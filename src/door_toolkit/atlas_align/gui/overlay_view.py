"""
Overlay view widget
===================

:class:`OverlayView` shows the user's 2D reference image with their
ROI outlines drawn on top. Supports:

* Setting/replacing the reference image (grayscale uint8/uint16/float).
* Drawing ROI contours in cyan for unassigned and a deterministic
  category colour once the IoU matcher has assigned them.
* Highlighting a selected ROI in bright yellow.
* Exposing its underlying :class:`pyqtgraph.ViewBox` so a companion
  :class:`~door_toolkit.atlas_align.gui.atlas_view.AtlasView` can lock
  pan/zoom state to it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from door_toolkit.atlas_align.config import get_logger
from door_toolkit.atlas_align.io.roi_loader import ROI


def _colorise_labels(
    label_projection: np.ndarray, label_lookup: Dict[int, str]
) -> np.ndarray:
    """Convert an integer label image to an RGBA uint8 image."""
    out = np.zeros((*label_projection.shape, 4), dtype=np.uint8)
    unique = np.unique(label_projection)
    for gid in unique:
        gid_int = int(gid)
        if gid_int == 0:
            continue
        name = label_lookup.get(gid_int, f"UNK_{gid_int}")
        qc = color_for_label(name)
        mask = label_projection == gid
        out[mask, 0] = qc.red()
        out[mask, 1] = qc.green()
        out[mask, 2] = qc.blue()
        out[mask, 3] = 255
    return out


def _compute_boundaries(label_projection: np.ndarray) -> np.ndarray:
    """Pixel-wise label boundary mask."""
    lp = label_projection
    changes = np.zeros_like(lp, dtype=bool)
    changes[:-1, :] |= lp[:-1, :] != lp[1:, :]
    changes[1:, :] |= lp[:-1, :] != lp[1:, :]
    changes[:, :-1] |= lp[:, :-1] != lp[:, 1:]
    changes[:, 1:] |= lp[:, :-1] != lp[:, 1:]
    changes &= lp > 0
    return changes

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


_UNASSIGNED_PEN = pg.mkPen(color=(0, 210, 255, 230), width=2)
_SELECTED_PEN = pg.mkPen(color=(255, 255, 0, 255), width=3)


def color_for_label(name: str) -> QtGui.QColor:
    """Deterministic colour per glomerulus name.

    Hashes the name to pick a stable hue on a large palette so the same
    glomerulus looks the same across the left and right panes.
    """
    digest = hashlib.md5(name.encode("utf-8")).digest()
    hue = digest[0] / 255.0 * 360.0
    sat = 180 + digest[1] % 60  # keep saturation reasonably high
    val = 200 + digest[2] % 55
    color = QtGui.QColor.fromHsv(int(hue), int(sat), int(val))
    return color


# ---------------------------------------------------------------------------
# View widget
# ---------------------------------------------------------------------------


@dataclass
class ROIDisplay:
    """Internal record for one rendered ROI contour."""

    roi: ROI
    plot_item: pg.PlotDataItem
    assigned_name: Optional[str] = None


class OverlayView(pg.GraphicsLayoutWidget):
    """pyqtgraph widget showing the reference image + ROI contours.

    Optionally renders a semi-transparent colour-coded atlas label
    projection *on top of* the reference image via
    :meth:`set_atlas_overlay`, so the user can visually check how well
    the current atlas pose + view lines up with their ROIs.
    """

    roi_selected = QtCore.pyqtSignal(int)  # roi_index, -1 = cleared
    # Emitted while the user drags the atlas with the mouse. Delta is in
    # image pixels (positive x → right, positive y → down).
    atlas_dragged = QtCore.pyqtSignal(float, float)  # dx_px, dy_px
    # Emitted on mouse wheel: positive step = zoom in, negative = zoom out.
    atlas_scale_requested = QtCore.pyqtSignal(float)  # factor (1.05 or 1/1.05)
    # Emitted when the user finishes drawing a polygon in draw-mode.
    # Payload: (xs, ys) numpy arrays of polygon vertices in image pixels.
    roi_drawn = QtCore.pyqtSignal(object, object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        logger.debug("OverlayView.__init__")
        self.setBackground("k")

        self._view_box: pg.ViewBox = self.addViewBox(row=0, col=0)
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(True)  # image origin at top-left

        # Layer order (bottom → top):
        #   reference image → atlas overlay (semi-transparent) → ROI outlines
        self._image_item = pg.ImageItem(axisOrder="row-major")
        self._view_box.addItem(self._image_item)

        self._atlas_overlay_item = pg.ImageItem(axisOrder="row-major")
        self._atlas_overlay_item.setOpacity(0.45)
        self._atlas_overlay_item.setZValue(5)
        self._view_box.addItem(self._atlas_overlay_item)

        self._atlas_boundary_item = pg.ImageItem(axisOrder="row-major")
        self._atlas_boundary_item.setOpacity(0.85)
        self._atlas_boundary_item.setZValue(6)
        self._view_box.addItem(self._atlas_boundary_item)

        self._roi_displays: List[ROIDisplay] = []
        self._selected_index: int = -1
        self._atlas_overlay_enabled: bool = True
        self._atlas_labels_enabled: bool = True
        self._atlas_label_items: List[pg.TextItem] = []
        self._drawn_label_items: List[pg.TextItem] = []
        self._drag_anchor: Optional[QtCore.QPointF] = None
        self._atlas_overlay_opacity: float = 0.45
        self._atlas_boundary_opacity: float = 0.90

        # Draw-mode state. When active, mouse clicks on the overlay add
        # polygon vertices instead of selecting ROIs.
        self._draw_mode: bool = False
        self._draw_points: List[Tuple[float, float]] = []
        self._draw_plot_item: Optional[pg.PlotDataItem] = None
        self._draw_markers: Optional[pg.ScatterPlotItem] = None

        # Disable pyqtgraph's built-in pan/zoom so left-mouse drag moves the
        # atlas rather than the ref image underneath. Also disable auto-range
        # so the zoom level we set in ``set_reference_image`` survives every
        # subsequent update — otherwise adding / removing glomerulus label
        # text items silently re-fits the view.
        self._view_box.setMouseEnabled(x=False, y=False)
        self._view_box.setMenuEnabled(False)
        self._view_box.disableAutoRange()

        # Install our own mouse handlers on the atlas overlay item so the
        # user can drag it around. Scroll wheel scales.
        self._atlas_overlay_item.mouseClickEvent = self._on_overlay_click  # type: ignore[assignment]
        self._atlas_overlay_item.mouseDragEvent = self._on_overlay_drag  # type: ignore[assignment]
        self._atlas_overlay_item.hoverEvent = lambda _e: None  # type: ignore[assignment]
        self._image_item.mouseClickEvent = self._on_click  # type: ignore[assignment]

    # ------------------------------------------------------------------ API

    @property
    def view_box(self) -> pg.ViewBox:
        """The underlying pyqtgraph :class:`ViewBox` — for link sharing."""
        return self._view_box

    def set_reference_image(self, image: np.ndarray) -> None:
        """Install the reference image and fit the view to it."""
        if image.ndim != 2:
            raise ValueError(
                f"reference image must be 2D, got shape {image.shape}"
            )
        # pyqtgraph accepts uint8 / uint16 / float, but we standardise to
        # float so levels behave predictably.
        img = image.astype(np.float32, copy=False)
        self._image_item.setImage(img, autoLevels=True)
        # Explicit setRange for the initial fit, then keep auto-range off so
        # subsequent projections / label changes don't reset the user's zoom.
        h, w = image.shape
        self._view_box.setRange(
            xRange=(0, w), yRange=(0, h), padding=0
        )
        self._view_box.disableAutoRange()
        logger.debug("set_reference_image: shape=%s", image.shape)

    def set_rois(self, rois: Sequence[ROI]) -> None:
        """Replace all ROI contours with the given list (unassigned style)."""
        self._clear_rois()
        for roi in rois:
            self._add_roi(roi)
        logger.debug("set_rois: %d ROIs", len(rois))

    def update_assignments(self, roi_to_name: Dict[int, Optional[str]]) -> None:
        """Recolour ROIs given a mapping ``{roi_index → glomerulus_name}``.

        A ``None`` value means the ROI is explicitly unassigned.
        """
        for idx, display in enumerate(self._roi_displays):
            name = roi_to_name.get(idx)
            display.assigned_name = name
            self._apply_roi_style(display, selected=(idx == self._selected_index))

    def select_roi(self, roi_index: int) -> None:
        """Programmatically select an ROI (or ``-1`` to clear)."""
        if self._selected_index == roi_index:
            return
        prev = self._selected_index
        self._selected_index = roi_index
        for i in (prev, roi_index):
            if 0 <= i < len(self._roi_displays):
                self._apply_roi_style(
                    self._roi_displays[i], selected=(i == self._selected_index)
                )
        self.roi_selected.emit(roi_index)

    def set_atlas_overlay(
        self,
        label_projection: Optional[np.ndarray],
        label_lookup: Optional[Dict[int, str]],
    ) -> None:
        """Draw the current atlas label projection on top of the reference.

        ``None`` values clear the overlay (colour, boundaries, and labels).

        The current zoom/pan state is captured before the update and
        restored after, so nothing we do internally (adding/removing
        labels, swapping overlay images, cycling views) can reset the
        user's view.
        """
        # Snapshot the current zoom so the update can't disturb it.
        xr, yr = self._view_box.viewRange()

        # Always clear existing text labels first so switching views doesn't
        # accumulate stale labels on screen.
        self._clear_atlas_labels()

        if label_projection is None or label_lookup is None:
            blank = np.zeros((1, 1, 4), dtype=np.uint8)
            self._atlas_overlay_item.setImage(blank, autoLevels=False)
            self._atlas_boundary_item.setImage(blank, autoLevels=False)
            self._view_box.setRange(
                xRange=xr, yRange=yr, padding=0
            )
            return

        color_rgba = _colorise_labels(label_projection, label_lookup)
        self._atlas_overlay_item.setImage(color_rgba, autoLevels=False)
        self._atlas_overlay_item.setOpacity(
            self._atlas_overlay_opacity if self._atlas_overlay_enabled else 0.0
        )

        # Magenta boundary lines for a sharper "where does glomerulus X sit?"
        boundaries = _compute_boundaries(label_projection)
        boundary_rgba = np.zeros((*boundaries.shape, 4), dtype=np.uint8)
        boundary_rgba[boundaries] = [255, 80, 255, 230]
        self._atlas_boundary_item.setImage(boundary_rgba, autoLevels=False)
        self._atlas_boundary_item.setOpacity(
            self._atlas_boundary_opacity if self._atlas_overlay_enabled else 0.0
        )

        # Glomerulus name labels at the centroid of each polygon — matches
        # the DoOR figure style (e.g. "DA1", "VA7m").
        self._draw_atlas_labels(label_projection, label_lookup)

        # Restore the zoom — adding items can sometimes re-range the
        # viewbox even with autoRange off.
        self._view_box.setRange(
            xRange=xr, yRange=yr, padding=0
        )

    def _clear_atlas_labels(self) -> None:
        for item in self._atlas_label_items:
            self._view_box.removeItem(item)
        self._atlas_label_items = []

    def _draw_atlas_labels(
        self,
        label_projection: np.ndarray,
        label_lookup: Dict[int, str],
    ) -> None:
        if not self._atlas_labels_enabled:
            return
        unique = np.unique(label_projection)
        for gid in unique:
            gid_int = int(gid)
            if gid_int == 0:
                continue
            mask = label_projection == gid_int
            if mask.sum() < 8:  # too small to label cleanly
                continue
            ys, xs = np.where(mask)
            cy = float(ys.mean())
            cx = float(xs.mean())
            name = label_lookup.get(gid_int, f"UNK_{gid_int}")

            # White fill with black outline for readability over any colour.
            html = (
                f"<div style='color: white; font-size: 11pt; "
                f"font-weight: bold; "
                f"text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, "
                f"-1px 1px 0 #000, 1px 1px 0 #000;'>{name}</div>"
            )
            text = pg.TextItem(html=html, anchor=(0.5, 0.5))
            text.setPos(cx, cy)
            text.setZValue(20)  # above overlays + ROI contours
            self._view_box.addItem(text)
            self._atlas_label_items.append(text)

    def toggle_atlas_labels(self, enabled: Optional[bool] = None) -> None:
        """Show/hide the glomerulus name text labels."""
        if enabled is None:
            self._atlas_labels_enabled = not self._atlas_labels_enabled
        else:
            self._atlas_labels_enabled = bool(enabled)
        for item in self._atlas_label_items:
            item.setVisible(self._atlas_labels_enabled)

    # ----------------------------------------------------- drawn-ROI labels

    def set_drawn_labels(
        self, labels: Sequence[Tuple[str, float, float]]
    ) -> None:
        """Replace the text labels painted above user-drawn polygons.

        ``labels`` is a sequence of ``(text, x_pixel, y_pixel)`` triples —
        one per drawn polygon, in the order they should appear. Called
        whenever a drawn ROI is added, deleted, or has its side changed
        (which updates the ``L1, L2 / R1, R2`` numbering).
        """
        # Snapshot current view so the relayout doesn't kick the zoom.
        xr, yr = self._view_box.viewRange()

        for item in self._drawn_label_items:
            self._view_box.removeItem(item)
        self._drawn_label_items = []

        for text_str, cx, cy in labels:
            if not text_str:
                continue
            html = (
                f"<div style='color: #ffe066; font-size: 12pt; "
                f"font-weight: bold; "
                f"text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, "
                f"-1px 1px 0 #000, 1px 1px 0 #000;'>{text_str}</div>"
            )
            text = pg.TextItem(html=html, anchor=(0.5, 0.5))
            text.setPos(float(cx), float(cy))
            text.setZValue(25)  # above atlas labels (20) and ROI contours (10)
            self._view_box.addItem(text)
            self._drawn_label_items.append(text)

        self._view_box.setRange(xRange=xr, yRange=yr, padding=0)

    # ----------------------------------------------------- draw-mode API

    def start_drawing(self) -> None:
        """Enter polygon draw-mode. Clicks append vertices."""
        self._draw_mode = True
        self._draw_points = []
        self._clear_draw_preview()
        logger.debug("OverlayView: draw mode ON")

    def finish_drawing(self) -> None:
        """Close the current polygon and emit ``roi_drawn``."""
        if not self._draw_mode:
            return
        pts = list(self._draw_points)
        self._draw_mode = False
        self._clear_draw_preview()
        if len(pts) >= 3:
            xs = np.asarray([p[0] for p in pts], dtype=np.float32)
            ys = np.asarray([p[1] for p in pts], dtype=np.float32)
            logger.debug("OverlayView: draw mode FINISHED with %d points", len(pts))
            self.roi_drawn.emit(xs, ys)
        else:
            logger.info(
                "OverlayView: draw mode finished with only %d points — ignored.",
                len(pts),
            )

    def cancel_drawing(self) -> None:
        """Abandon the current draw without emitting."""
        self._draw_mode = False
        self._draw_points = []
        self._clear_draw_preview()
        logger.debug("OverlayView: draw mode CANCELLED")

    @property
    def draw_mode_active(self) -> bool:
        return self._draw_mode

    def _clear_draw_preview(self) -> None:
        if self._draw_plot_item is not None:
            self._view_box.removeItem(self._draw_plot_item)
            self._draw_plot_item = None
        if self._draw_markers is not None:
            self._view_box.removeItem(self._draw_markers)
            self._draw_markers = None

    def _refresh_draw_preview(self) -> None:
        self._clear_draw_preview()
        if not self._draw_points:
            return
        xs = [p[0] for p in self._draw_points]
        ys = [p[1] for p in self._draw_points]
        # Dashed polyline; closes back to the first vertex if ≥ 3 pts.
        if len(xs) >= 3:
            xs_plot = xs + xs[:1]
            ys_plot = ys + ys[:1]
        else:
            xs_plot, ys_plot = xs, ys
        pen = pg.mkPen(color=(255, 220, 60, 230), width=2, style=QtCore.Qt.PenStyle.DashLine)
        self._draw_plot_item = pg.PlotDataItem(xs_plot, ys_plot, pen=pen)
        self._draw_plot_item.setZValue(30)
        self._view_box.addItem(self._draw_plot_item)
        self._draw_markers = pg.ScatterPlotItem(
            xs, ys, symbol="o", size=8,
            pen=pg.mkPen(color=(255, 220, 60, 255)),
            brush=pg.mkBrush(255, 220, 60, 180),
        )
        self._draw_markers.setZValue(31)
        self._view_box.addItem(self._draw_markers)

    def reset_zoom(self) -> None:
        """Return the view to fit the reference image edge-to-edge."""
        img = self._image_item.image
        if img is None:
            return
        h, w = img.shape[:2]
        self._view_box.setRange(
            xRange=(0, w), yRange=(0, h),
            padding=0, disableAutoRange=True,
        )

    def current_zoom_factor(self) -> float:
        """Return the current zoom as a fraction of "fit image" (1.0 = fit)."""
        img = self._image_item.image
        if img is None:
            return 1.0
        h, w = img.shape[:2]
        xr, _yr = self._view_box.viewRange()
        current_span = xr[1] - xr[0]
        if current_span <= 0:
            return 1.0
        return float(w / current_span)

    def set_zoom_factor(self, factor: float) -> None:
        """Set the zoom to a multiple of "fit image". 1.0 = fit; 2.0 = 2× in."""
        img = self._image_item.image
        if img is None:
            return
        h, w = img.shape[:2]
        factor = max(0.05, float(factor))
        xr, yr = self._view_box.viewRange()
        cx = (xr[0] + xr[1]) / 2.0
        cy = (yr[0] + yr[1]) / 2.0
        new_w = w / factor
        new_h = h / factor
        self._view_box.setRange(
            xRange=(cx - new_w / 2.0, cx + new_w / 2.0),
            yRange=(cy - new_h / 2.0, cy + new_h / 2.0),
            padding=0, disableAutoRange=True,
        )

    def toggle_atlas_overlay(self, enabled: Optional[bool] = None) -> None:
        """Show/hide the atlas overlay without losing its image data."""
        if enabled is None:
            self._atlas_overlay_enabled = not self._atlas_overlay_enabled
        else:
            self._atlas_overlay_enabled = bool(enabled)
        self._atlas_overlay_item.setOpacity(
            self._atlas_overlay_opacity if self._atlas_overlay_enabled else 0.0
        )
        self._atlas_boundary_item.setOpacity(
            self._atlas_boundary_opacity if self._atlas_overlay_enabled else 0.0
        )

    def set_atlas_overlay_opacity(self, value: float) -> None:
        """Set the atlas overlay opacity (0.0–1.0). Boundaries scale too."""
        value = max(0.0, min(1.0, float(value)))
        self._atlas_overlay_opacity = value
        # Boundary lines tend to look cleaner slightly stronger than the fill.
        self._atlas_boundary_opacity = min(1.0, value + 0.45)
        if self._atlas_overlay_enabled:
            self._atlas_overlay_item.setOpacity(self._atlas_overlay_opacity)
            self._atlas_boundary_item.setOpacity(self._atlas_boundary_opacity)

    # --------------------------------------------------------------- helpers

    def _clear_rois(self) -> None:
        for display in self._roi_displays:
            self._view_box.removeItem(display.plot_item)
        self._roi_displays.clear()
        self._selected_index = -1

    def _add_roi(self, roi: ROI) -> None:
        # Close the polygon by repeating the first point. We render only the
        # outline — no text labels — so the reference image stays readable.
        x = np.concatenate([roi.x, roi.x[:1]])
        y = np.concatenate([roi.y, roi.y[:1]])
        plot_item = pg.PlotDataItem(
            x=x, y=y, pen=_UNASSIGNED_PEN, antialias=True
        )
        plot_item.setZValue(10)  # keep ROIs on top of the atlas overlay
        self._view_box.addItem(plot_item)

        self._roi_displays.append(
            ROIDisplay(roi=roi, plot_item=plot_item)
        )

    def _apply_roi_style(
        self, display: ROIDisplay, selected: bool
    ) -> None:
        if selected:
            display.plot_item.setPen(_SELECTED_PEN)
        elif display.assigned_name:
            color = color_for_label(display.assigned_name)
            display.plot_item.setPen(pg.mkPen(color=color, width=2))
        else:
            display.plot_item.setPen(_UNASSIGNED_PEN)

    # -------------------------------------------------------------- events

    def _on_click(self, event) -> None:  # pragma: no cover - GUI interaction
        # In draw mode: left-click adds a vertex; right-click finishes the poly.
        if self._draw_mode:
            pos = event.pos()
            x, y = float(pos.x()), float(pos.y())
            if event.button() == QtCore.Qt.MouseButton.LeftButton:
                self._draw_points.append((x, y))
                self._refresh_draw_preview()
                event.accept()
                return
            if event.button() == QtCore.Qt.MouseButton.RightButton:
                self.finish_drawing()
                event.accept()
                return
        if event.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        pos = event.pos()
        x, y = pos.x(), pos.y()
        best_idx = -1
        best_dist = float("inf")
        for i, display in enumerate(self._roi_displays):
            cx, cy = display.roi.centroid
            d = (cx - x) ** 2 + (cy - y) ** 2
            if d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx >= 0:
            self.select_roi(best_idx)
        event.accept()

    def _on_overlay_click(self, event) -> None:  # pragma: no cover - GUI
        # In draw mode, treat a click on the atlas overlay the same as
        # one on the reference image — append a vertex / finish. Otherwise
        # swallow so it doesn't trigger ROI-select beneath.
        if self._draw_mode:
            self._on_click(event)
            return
        event.accept()

    def _on_overlay_drag(self, event) -> None:  # pragma: no cover - GUI
        """Handle left-mouse drag on the atlas overlay.

        * Plain left-drag → move the atlas (emit ``atlas_dragged``).
        * Ctrl + left-drag OR middle-drag → pan the viewbox (both ref + atlas).
        * In draw mode: ignore drags entirely so the user can click to
          place vertices without accidentally nudging the atlas.
        """
        if self._draw_mode:
            event.accept()
            return
        btn = event.button()
        mods = event.modifiers() if hasattr(event, "modifiers") else None
        is_pan = btn == QtCore.Qt.MouseButton.MiddleButton or (
            btn == QtCore.Qt.MouseButton.LeftButton
            and mods is not None
            and (mods & QtCore.Qt.KeyboardModifier.ControlModifier)
        )

        if btn not in (
            QtCore.Qt.MouseButton.LeftButton,
            QtCore.Qt.MouseButton.MiddleButton,
        ):
            event.ignore()
            return

        if event.isStart():
            self._drag_anchor = event.buttonDownPos()
            event.accept()
            return
        if event.isFinish():
            self._drag_anchor = None
            event.accept()
            return

        cur = event.pos()
        last = event.lastPos()
        dx = float(cur.x() - last.x())
        dy = float(cur.y() - last.y())

        if is_pan:
            # Scene pan: shift the view window by the cursor delta.
            self._view_box.translateBy(x=-dx, y=-dy)
        else:
            self.atlas_dragged.emit(dx, dy)
        event.accept()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:  # noqa: N802
        """Mouse wheel handling.

        * Ctrl+scroll → zoom the whole view (ref image AND atlas together
          stay locked in image-pixel coordinates, just drawn larger/smaller).
        * Plain scroll → scale the atlas only, leaving the ref image fixed.
        """
        delta = event.angleDelta().y()
        if delta == 0:
            return super().wheelEvent(event)
        factor = 1.15 if delta > 0 else (1.0 / 1.15)

        mods = event.modifiers()
        if mods & QtCore.Qt.KeyboardModifier.ControlModifier:
            # Zoom the viewbox around the mouse cursor so the point under
            # the cursor stays under the cursor.
            pos = event.position()  # QPointF in widget coords
            scene_pos = self.mapToScene(int(pos.x()), int(pos.y()))
            center_data = self._view_box.mapSceneToView(scene_pos)
            self._view_box.scaleBy(
                (1.0 / factor, 1.0 / factor), center=center_data
            )
            event.accept()
            return

        self.atlas_scale_requested.emit(factor)
        event.accept()
