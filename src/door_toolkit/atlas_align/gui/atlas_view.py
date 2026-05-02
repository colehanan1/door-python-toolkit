"""
Atlas view widget
=================

:class:`AtlasView` renders:

* The grayscale MIP of the currently-posed atlas as the background
  image.
* The label-projection boundaries as thin magenta lines on top.
* Per-glomerulus coloured overlays that match the colours used in
  :class:`~door_toolkit.atlas_align.gui.overlay_view.OverlayView`.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyqtgraph as pg
from PyQt6 import QtCore, QtGui, QtWidgets

from door_toolkit.atlas_align.config import get_logger
from door_toolkit.atlas_align.gui.overlay_view import color_for_label

logger = get_logger(__name__)


_BOUNDARY_PEN = pg.mkPen(color=(255, 80, 255, 220), width=1)


class AtlasView(pg.GraphicsLayoutWidget):
    """Grayscale MIP + label-projection boundary overlay."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        logger.debug("AtlasView.__init__")
        self.setBackground("k")

        self._view_box: pg.ViewBox = self.addViewBox(row=0, col=0)
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(True)

        self._grayscale_item = pg.ImageItem(axisOrder="row-major")
        self._view_box.addItem(self._grayscale_item)

        self._label_color_item = pg.ImageItem(axisOrder="row-major")
        self._label_color_item.setOpacity(0.35)
        self._view_box.addItem(self._label_color_item)

        self._boundary_item = pg.ImageItem(axisOrder="row-major")
        self._view_box.addItem(self._boundary_item)

        self._show_label_overlay: bool = True

    @property
    def view_box(self) -> pg.ViewBox:
        return self._view_box

    def link_to(self, other: pg.ViewBox) -> None:
        """Link this view's pan/zoom to another ViewBox (bidirectional)."""
        self._view_box.setXLink(other)
        self._view_box.setYLink(other)

    def set_projection(
        self,
        grayscale_mip: np.ndarray,
        label_projection: np.ndarray,
        label_lookup: Dict[int, str],
    ) -> None:
        """Install the latest grayscale + label projections."""
        if grayscale_mip.ndim != 2 or label_projection.ndim != 2:
            raise ValueError(
                "Both projections must be 2D; got "
                f"grayscale={grayscale_mip.shape} label={label_projection.shape}"
            )
        if grayscale_mip.shape != label_projection.shape:
            raise ValueError(
                "grayscale and label projection shapes differ: "
                f"{grayscale_mip.shape} vs {label_projection.shape}"
            )

        self._grayscale_item.setImage(
            grayscale_mip.astype(np.float32, copy=False),
            autoLevels=True,
            levels=None,
        )

        color_img = _colorise_labels(label_projection, label_lookup)
        self._label_color_item.setImage(
            color_img,
            autoLevels=False,
            opacity=0.35 if self._show_label_overlay else 0.0,
        )
        self._label_color_item.setOpacity(
            0.35 if self._show_label_overlay else 0.0
        )

        boundaries = _compute_boundaries(label_projection)
        boundary_rgba = np.zeros(
            (*boundaries.shape, 4), dtype=np.uint8
        )
        boundary_rgba[boundaries] = [255, 80, 255, 230]
        self._boundary_item.setImage(
            boundary_rgba,
            autoLevels=False,
            opacity=0.85 if self._show_label_overlay else 0.0,
        )
        self._boundary_item.setOpacity(
            0.85 if self._show_label_overlay else 0.0
        )

        self._view_box.autoRange()
        logger.debug(
            "AtlasView.set_projection shape=%s labels=%d",
            grayscale_mip.shape,
            int(label_projection.max()),
        )

    def toggle_label_overlay(self, enabled: Optional[bool] = None) -> None:
        """Show/hide the coloured label overlay + magenta boundaries."""
        if enabled is None:
            self._show_label_overlay = not self._show_label_overlay
        else:
            self._show_label_overlay = bool(enabled)
        alpha_colour = 0.35 if self._show_label_overlay else 0.0
        alpha_edge = 0.85 if self._show_label_overlay else 0.0
        self._label_color_item.setOpacity(alpha_colour)
        self._boundary_item.setOpacity(alpha_edge)


# ---------------------------------------------------------------------------
# Numpy helpers
# ---------------------------------------------------------------------------


def _colorise_labels(
    label_projection: np.ndarray, label_lookup: Dict[int, str]
) -> np.ndarray:
    """Convert an integer label image to an RGBA uint8 image.

    Uses :func:`color_for_label` so the colour matches what the overlay
    view shows for assigned ROIs.
    """
    out = np.zeros((*label_projection.shape, 4), dtype=np.uint8)
    unique = np.unique(label_projection)
    for gid in unique:
        gid_int = int(gid)
        if gid_int == 0:
            continue
        name = label_lookup.get(gid_int, f"UNK_{gid_int}")
        qc = color_for_label(name)
        out[label_projection == gid, 0] = qc.red()
        out[label_projection == gid, 1] = qc.green()
        out[label_projection == gid, 2] = qc.blue()
        out[label_projection == gid, 3] = 255
    return out


def _compute_boundaries(label_projection: np.ndarray) -> np.ndarray:
    """Pixel-wise label boundary mask (True where a label changes)."""
    lp = label_projection
    changes = np.zeros_like(lp, dtype=bool)
    changes[:-1, :] |= lp[:-1, :] != lp[1:, :]
    changes[1:, :] |= lp[:-1, :] != lp[1:, :]
    changes[:, :-1] |= lp[:, :-1] != lp[:, 1:]
    changes[:, 1:] |= lp[:, :-1] != lp[:, 1:]
    # Ignore transitions that are just background→background.
    changes &= lp > 0
    return changes
