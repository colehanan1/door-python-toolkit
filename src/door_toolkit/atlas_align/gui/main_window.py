"""
Main window wiring
==================

Assembles :class:`OverlayView`, :class:`AtlasView`,
:class:`PoseControlPanel` and :class:`IoUPanel` into a single
:class:`AtlasAlignMainWindow`, and drives a background
:class:`ProjectionWorker` that keeps the Qt event loop responsive while
the volume transform / projection / IoU matching cycle runs.

Pipeline on every pose change (debounced to
:data:`door_toolkit.atlas_align.config.PROJECTION_DEBOUNCE_MS` ms):

1. :func:`door_toolkit.atlas_align.core.volume_transform.transform_atlas`
   → resampled (grayscale, labelmap).
2. :func:`door_toolkit.atlas_align.core.projection.project_atlas` →
   grayscale MIP + label-projection.
3. Rasterise each ROI to a 2D mask (cached once).
4. :func:`door_toolkit.atlas_align.core.iou_matcher.assign_rois` →
   AssignmentResult.
5. Emit ``projection_ready`` to update both views + IoU panel on the
   main thread.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile
from PyQt6 import QtCore, QtGui, QtWidgets

from door_toolkit.atlas_align.config import (
    DEFAULT_IOU_THRESHOLD,
    PROJECTION_DEBOUNCE_MS,
    PROJECTION_WARN_MS,
    configure_logging,
    get_logger,
)
from door_toolkit.atlas_align.core.iou_matcher import (
    AssignmentResult,
    assign_rois,
)
from door_toolkit.atlas_align.core.projection import project_atlas
from door_toolkit.atlas_align.core.volume_transform import (
    Pose,
    transform_atlas,
)
from door_toolkit.atlas_align.gui.atlas_view import AtlasView
from door_toolkit.atlas_align.gui.iou_panel import IoUPanel
from door_toolkit.atlas_align.gui.overlay_view import OverlayView
from door_toolkit.atlas_align.gui.pose_controls import PoseControlPanel
from door_toolkit.atlas_align.io.atlas_loader import AtlasBundle, load_atlas_bundle
from door_toolkit.atlas_align.io.pose_io import (
    file_sha256,
    load_pose,
    save_pose,
)
from door_toolkit.atlas_align.io.roi_exporter import (
    export_assignments_csv,
    export_roi_zip,
)
from door_toolkit.atlas_align.io.roi_loader import ROI, ROISet, load_rois
from door_toolkit.atlas_align.io.dff_loader import (
    DFFBundle,
    load_dff_directory,
    match_rois_to_dff,
)
from door_toolkit.atlas_align.io.door_response import (
    DoorResponseBundle,
    cosine_similarity,
    load_door_responses,
    pearson_correlation,
    project_to_door_scale,
    rank_glomeruli_by_similarity,
)
from door_toolkit.atlas_align.io.activity_maps import (
    ActivityMaps,
    load_activity_maps,
)
from door_toolkit.atlas_align.io.drawn_roi_dff import (
    DrawnROIDff,
    DrawnROIDffExtractor,
    door_row_scale,
)
from door_toolkit.atlas_align.gui.drawn_roi_panel import (
    DrawnROI,
    DrawnROIPanel,
    load_drawn_rois_csv,
)
from door_toolkit.atlas_align.gui.response_panel import ResponsePanel

logger = get_logger(__name__)


def fit_reference_and_rois_to_atlas(
    reference: np.ndarray,
    roi_set: ROISet,
    atlas_shape_yx: tuple[int, int],
) -> tuple[np.ndarray, ROISet]:
    """Resize the reference image and re-scale ROIs to the atlas projection grid.

    The IoU matcher requires ROI masks and the atlas label projection to live
    on the same pixel grid. Atlas projection shape is the YX of the 3D
    labelmap (e.g. ``(566, 1210)`` for JRC2018F). This helper rescales the
    user's reference image (e.g. ``(2048, 2048)``) and their ROIs onto that
    grid so the pipeline can run.

    Aspect ratio may distort when source and target aspect ratios differ; this
    is expected — the geometry is consistent across all three artifacts so
    IoU remains meaningful, and the user's rigid pose (rotate + scale +
    translate) still places the atlas correctly on the resized reference.
    """
    from skimage.transform import resize as skimage_resize

    target_h, target_w = int(atlas_shape_yx[0]), int(atlas_shape_yx[1])
    src_h, src_w = int(reference.shape[0]), int(reference.shape[1])
    if (src_h, src_w) == (target_h, target_w):
        logger.debug("Reference already at atlas shape; no resize needed.")
        return reference, roi_set

    logger.info(
        "Resizing reference %s -> atlas projection shape %s and scaling ROIs.",
        (src_h, src_w), (target_h, target_w),
    )
    ref_resized = skimage_resize(
        reference.astype(np.float32, copy=False),
        (target_h, target_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32, copy=False)

    scale_y = target_h / src_h
    scale_x = target_w / src_w
    new_rois: list[ROI] = []
    for roi in roi_set:
        new_rois.append(
            ROI(
                name=roi.name,
                x=roi.x * scale_x,
                y=roi.y * scale_y,
                roi_type=roi.roi_type,
                index=roi.index,
            )
        )
    return ref_resized, ROISet(rois=new_rois, source=roi_set.source)


def _load_reference_image(path: Path) -> np.ndarray:
    """Read a 2D reference image (TIF / PNG). 3D TIFs are max-projected."""
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(str(path))
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            arr = arr[..., :3].mean(axis=-1)
        else:
            arr = arr.max(axis=0)
    if arr.ndim != 2:
        raise ValueError(
            f"Reference image must be 2D or reducible to 2D, got {arr.shape}"
        )
    return arr.astype(np.float32, copy=False)


class ProjectionPayload:
    """Container passed from worker thread to main thread."""

    __slots__ = (
        "pose",
        "grayscale_mip",
        "label_projection",
        "assignment",
        "elapsed_ms",
    )

    def __init__(
        self,
        pose: Pose,
        grayscale_mip: np.ndarray,
        label_projection: np.ndarray,
        assignment: AssignmentResult,
        elapsed_ms: float,
    ) -> None:
        self.pose = pose
        self.grayscale_mip = grayscale_mip
        self.label_projection = label_projection
        self.assignment = assignment
        self.elapsed_ms = elapsed_ms


class DrawnROIDffWorker(QtCore.QObject):
    """Runs :meth:`DrawnROIDffExtractor.compute_for_polygon` off the GUI thread.

    Must live at **module scope** — Qt's meta-object system can't
    ``moveToThread`` a class defined inside a method, which is why the
    inline version silently failed to run.
    """

    finished = QtCore.pyqtSignal(int, object)
    failed = QtCore.pyqtSignal(int, str)

    def __init__(
        self,
        extractor: "DrawnROIDffExtractor",
        drawn_index: int,
        xs,
        ys,
    ) -> None:
        super().__init__()
        self._extractor = extractor
        self._drawn_index = int(drawn_index)
        self._xs = xs
        self._ys = ys

    @QtCore.pyqtSlot()
    def run(self) -> None:
        logger.info(
            "DrawnROIDffWorker: starting compute for drawn ROI #%d (%d vertices)",
            self._drawn_index, len(self._xs),
        )
        try:
            result = self._extractor.compute_for_polygon(self._xs, self._ys)
            logger.info(
                "DrawnROIDffWorker: finished drawn ROI #%d (%d odors)",
                self._drawn_index, len(result.odor_order),
            )
            self.finished.emit(self._drawn_index, result)
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "DrawnROIDffWorker: drawn ROI #%d FAILED",
                self._drawn_index,
            )
            self.failed.emit(self._drawn_index, str(e))


class ProjectionWorker(QtCore.QObject):
    """Runs transform → project → IoU off the GUI thread."""

    projection_ready = QtCore.pyqtSignal(object)  # ProjectionPayload
    failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        bundle: AtlasBundle,
        roi_masks: List[np.ndarray],
        roi_names: List[str],
    ) -> None:
        super().__init__()
        self._bundle = bundle
        self._roi_masks = roi_masks
        self._roi_names = roi_names
        self._pending: Optional[Tuple[Pose, float, int]] = None
        self._lock = False

    @QtCore.pyqtSlot(object, float, int)
    def recompute(self, pose: Pose, threshold: float, slice_index: int) -> None:
        if self._lock:
            self._pending = (pose, threshold, slice_index)
            return
        self._lock = True
        try:
            while True:
                self._compute(pose, threshold, slice_index)
                if self._pending is None:
                    break
                pose, threshold, slice_index = self._pending
                self._pending = None
        finally:
            self._lock = False

    def _compute(self, pose: Pose, threshold: float, slice_index: int) -> None:
        t0 = time.perf_counter()
        try:
            # For multi-view atlases, transform and project only the active
            # slice so Z=5 DoOR stacks don't cost 5x memory/time each pose.
            Z = self._bundle.labelmap.shape[0]
            if 0 <= slice_index < Z and Z > 1:
                # Slice mode: wrap the 2D slice in a Z=1 volume so the
                # existing 3D pipeline still works, but the cost is cheap.
                gray_sub = self._bundle.grayscale[slice_index:slice_index + 1]
                lm_sub = self._bundle.labelmap[slice_index:slice_index + 1]
            else:
                gray_sub = self._bundle.grayscale
                lm_sub = self._bundle.labelmap

            transformed = transform_atlas(gray_sub, lm_sub, pose)
            projection = project_atlas(
                transformed.grayscale, transformed.labelmap
            )
            result = assign_rois(
                self._roi_masks,
                self._roi_names,
                projection.labelmap_projection,
                self._bundle.labels,
                threshold=threshold,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.projection_ready.emit(
                ProjectionPayload(
                    pose=pose,
                    grayscale_mip=projection.grayscale_mip,
                    label_projection=projection.labelmap_projection,
                    assignment=result,
                    elapsed_ms=elapsed_ms,
                )
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Projection worker failed")
            self.failed.emit(str(e))


class AtlasAlignMainWindow(QtWidgets.QMainWindow):
    """Top-level window orchestrating the atlas-alignment session."""

    request_recompute = QtCore.pyqtSignal(object, float, int)

    def __init__(
        self,
        bundle: AtlasBundle,
        reference_image: np.ndarray,
        roi_set: ROISet,
        *,
        initial_pose: Optional[Pose] = None,
        threshold: float = DEFAULT_IOU_THRESHOLD,
        reference_path: Optional[Path] = None,
        dff_bundle: Optional[DFFBundle] = None,
        activity_maps: Optional[ActivityMaps] = None,
        dff_extractor: Optional[DrawnROIDffExtractor] = None,
    ) -> None:
        super().__init__()
        logger.debug("AtlasAlignMainWindow.__init__")

        self.setWindowTitle("atlas_align")
        # Size to the available screen area (minus taskbar/menu) so the
        # window never exceeds the user's display.
        app = QtWidgets.QApplication.instance()
        if app is not None:
            screen = app.primaryScreen()
            if screen is not None:
                avail = screen.availableGeometry()
                self.resize(avail.width(), avail.height())
                self.move(avail.topLeft())

        self._bundle = bundle
        self._reference_image = reference_image
        self._roi_set = roi_set
        self._reference_path = reference_path
        self._last_payload: Optional[ProjectionPayload] = None

        self._roi_masks = [
            roi.to_mask(reference_image.shape[:2]) for roi in roi_set
        ]
        self._roi_names = [roi.name for roi in roi_set]
        # Active ROI state — used by the Response dock & DoOR suggestions.
        self._active_roi_index: Optional[int] = None
        # User-drawn ROIs (polygon coords in fullframe pixel space).
        self._activity_maps: Optional[ActivityMaps] = activity_maps
        self._dff_extractor: Optional[DrawnROIDffExtractor] = dff_extractor
        self._drawn_rois: List[DrawnROI] = []
        self._drawn_dff_results: Dict[int, DrawnROIDff] = {}
        self._next_drawn_index: int = 0
        self._active_drawn_index: Optional[int] = None

        # ΔF/F data (optional). Match FIJI ROIs to dff columns by nearest
        # centroid so clicking an ROI in the table / image looks up the
        # right trace.
        self._dff_bundle: Optional[DFFBundle] = dff_bundle
        self._fiji_to_dff: Dict[int, int] = {}
        self._door_bundle: Optional[DoorResponseBundle] = None
        if self._dff_bundle is not None and len(roi_set) > 0:
            self._fiji_to_dff = match_rois_to_dff(
                list(roi_set), self._dff_bundle, tolerance_px=30.0
            )

        # DoOR literature lookup — needed for both the dff path AND the
        # drawn-ROI / activity-map path. Load whenever we have some odor
        # ordering to key off.
        odor_order_for_door: Optional[List[str]] = None
        if self._dff_bundle is not None:
            odor_order_for_door = self._dff_bundle.odor_order
        elif self._activity_maps is not None:
            odor_order_for_door = self._activity_maps.odor_order
        elif self._dff_extractor is not None:
            odor_order_for_door = self._dff_extractor.odor_order
        if odor_order_for_door is not None:
            try:
                from door_toolkit.atlas_align import __file__ as _pkg
                mappings_dir = (
                    Path(_pkg).resolve().parent.parent.parent.parent
                    / "data" / "mappings"
                )
                self._door_bundle = load_door_responses(
                    odor_order=odor_order_for_door,
                    glomeruli=list(bundle.labels.values()),
                    mappings_dir=mappings_dir,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Failed to load DoOR response data; DoOR comparison disabled."
                )
                self._door_bundle = None

        # Multi-view atlas state. For DoOR atlases Z > 1 and each Z index
        # corresponds to one of the slice views; otherwise Z == 1 and
        # there is effectively one view. The same pose is shared across all
        # views — switching view (Space/Tab) doesn't reset the sliders.
        self._n_views = int(bundle.labelmap.shape[0])
        self._view_names: List[str] = list(
            bundle.manifest.get("view_names") or []
        )
        if len(self._view_names) != self._n_views:
            self._view_names = [f"slice_{i}" for i in range(self._n_views)]
        self._current_view: int = 0

        atlas_type = str(bundle.manifest.get("atlas_type", ""))
        self._is_door_2d = atlas_type.startswith("door_2d")

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.overlay_view = OverlayView()
        self.atlas_view = AtlasView()
        splitter.addWidget(self.overlay_view)
        splitter.addWidget(self.atlas_view)
        if self._is_door_2d:
            # The atlas is already overlaid on the reference image in the
            # left pane — hide the redundant right pane entirely so the
            # user isn't staring at two copies of the same projection.
            self.atlas_view.hide()
            splitter.setSizes([1600, 0])
        else:
            splitter.setSizes([800, 800])
        self._splitter = splitter
        self.setCentralWidget(splitter)

        pose_dock = QtWidgets.QDockWidget("Pose", self)
        self.pose_controls = PoseControlPanel(pose_dock)
        # Wrap the pose controls in a QScrollArea so the dock border is
        # freely draggable — even if the panel's natural height exceeds
        # the dock's current height, the user can still shrink the dock.
        pose_scroll = QtWidgets.QScrollArea(pose_dock)
        pose_scroll.setWidgetResizable(True)
        pose_scroll.setWidget(self.pose_controls)
        pose_scroll.setMinimumHeight(50)
        pose_dock.setWidget(pose_scroll)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, pose_dock
        )
        self._pose_dock = pose_dock

        iou_dock = QtWidgets.QDockWidget("Assignments", self)
        self.iou_panel = IoUPanel(iou_dock)
        self.iou_panel.set_threshold(threshold)
        # Seed the Manual-column dropdown with every glomerulus the atlas
        # knows about — lets the user pick any of them regardless of
        # which view is currently active.
        self.iou_panel.set_glomerulus_palette(list(bundle.labels.values()))
        iou_dock.setWidget(self.iou_panel)
        self.addDockWidget(
            QtCore.Qt.DockWidgetArea.RightDockWidgetArea, iou_dock
        )
        self._iou_dock = iou_dock

        # Drawn-ROI dock — user-drawn polygons + their activity-map /
        # ΔF/F responses. Added whenever we have either an activity-map
        # stack or a dff extractor.
        self._drawn_dock: Optional[QtWidgets.QDockWidget] = None
        if self._activity_maps is not None or self._dff_extractor is not None:
            drawn_dock = QtWidgets.QDockWidget("Drawn ROIs", self)
            self.drawn_panel = DrawnROIPanel(drawn_dock)
            self.drawn_panel.set_glomerulus_palette(list(bundle.labels.values()))
            self.drawn_panel.draw_requested.connect(self._on_start_drawing)
            self.drawn_panel.delete_requested.connect(self._on_delete_drawn)
            self.drawn_panel.roi_clicked.connect(self._on_drawn_clicked)
            self.drawn_panel.manual_assignment_changed.connect(
                self._on_drawn_manual_changed
            )
            self.drawn_panel.side_changed.connect(self._on_drawn_side_changed)
            self.drawn_panel.export_requested.connect(self._on_drawn_export)
            self.drawn_panel.load_requested.connect(self._on_drawn_load)
            drawn_dock.setWidget(self.drawn_panel)
            self.addDockWidget(
                QtCore.Qt.DockWidgetArea.RightDockWidgetArea, drawn_dock
            )
            self.tabifyDockWidget(iou_dock, drawn_dock)
            self._drawn_dock = drawn_dock
            # Hook draw-finished events from the overlay view.
            self.overlay_view.roi_drawn.connect(self._on_polygon_drawn)
        else:
            self.drawn_panel = None  # type: ignore[assignment]

        # Response dock — shows per-odor response data for the selected
        # ROI (either a loaded FIJI ROI matched against dff CSVs, or a
        # drawn ROI whose response came from the activity-map stack).
        # Appears whenever *either* source is available.
        self._response_dock: Optional[QtWidgets.QDockWidget] = None
        if (
            self._dff_bundle is not None
            or self._activity_maps is not None
            or self._dff_extractor is not None
        ):
            response_dock = QtWidgets.QDockWidget("Response", self)
            self.response_panel = ResponsePanel(response_dock)
            self.response_panel.suggestion_clicked.connect(
                self._on_door_suggestion_clicked
            )
            response_dock.setWidget(self.response_panel)
            self.addDockWidget(
                QtCore.Qt.DockWidgetArea.RightDockWidgetArea, response_dock
            )
            self.tabifyDockWidget(iou_dock, response_dock)
            iou_dock.raise_()
            self._response_dock = response_dock
        else:
            self.response_panel = None  # type: ignore[assignment]

        self._status = self.statusBar()
        self._status.showMessage("Ready.")

        # Only link the two panes' pan/zoom if the atlas pane is visible.
        # Linking propagates autoRange from the hidden atlas pane into the
        # overlay pane and kicks the user's zoom back to "fit" on every
        # projection update.
        if not self._is_door_2d:
            self.atlas_view.link_to(self.overlay_view.view_box)

        self.overlay_view.set_reference_image(reference_image)
        self.overlay_view.set_rois(list(roi_set))

        self._worker_thread = QtCore.QThread(self)
        self._worker = ProjectionWorker(
            bundle, self._roi_masks, self._roi_names
        )
        self._worker.moveToThread(self._worker_thread)
        self._worker.projection_ready.connect(self._on_projection_ready)
        self._worker.failed.connect(self._on_worker_failed)
        self.request_recompute.connect(self._worker.recompute)
        self._worker_thread.start()

        self._debounce = QtCore.QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.timeout.connect(self._fire_recompute)

        self.pose_controls.pose_changed.connect(self._on_pose_changed)
        self.pose_controls.auto_center_requested.connect(self._on_auto_center)
        self.pose_controls.save_pose_requested.connect(self._on_save_pose)
        self.pose_controls.load_pose_requested.connect(self._on_load_pose)
        self.iou_panel.threshold_changed.connect(self._on_threshold_changed)
        self.iou_panel.export_requested.connect(self._on_export)
        self.iou_panel.row_selected.connect(self.overlay_view.select_roi)
        self.iou_panel.row_selected.connect(self._on_roi_selected)
        self.overlay_view.roi_selected.connect(self._on_roi_selected)
        self.iou_panel.manual_assignment_changed.connect(
            self._on_manual_assignment_changed
        )
        # Mouse-drag the atlas: translate the drag delta into a pose update.
        self.overlay_view.atlas_dragged.connect(self._on_atlas_dragged)
        self.overlay_view.atlas_scale_requested.connect(self._on_atlas_scale)

        self._install_shortcuts()
        self._install_view_toolbar()

        self.pose_controls.set_pose(
            initial_pose if initial_pose is not None else Pose(),
            emit=True,
        )

    def _install_shortcuts(self) -> None:
        sc = QtGui.QShortcut

        def nudge_pose(fn):
            def _():
                fn(self.pose_controls)
                self.pose_controls._emit_pose_changed()  # type: ignore[attr-defined]

            return _

        bindings = [
            ("Left", lambda p: p.tx.nudge(-1.0)),
            ("Right", lambda p: p.tx.nudge(+1.0)),
            ("Up", lambda p: p.ty.nudge(-1.0)),
            ("Down", lambda p: p.ty.nudge(+1.0)),
            ("Shift+Left", lambda p: p.tx.nudge(-10.0)),
            ("Shift+Right", lambda p: p.tx.nudge(+10.0)),
            ("Shift+Up", lambda p: p.ty.nudge(-10.0)),
            ("Shift+Down", lambda p: p.ty.nudge(+10.0)),
            ("Q", lambda p: p.rx.nudge(+0.5)),
            ("E", lambda p: p.rx.nudge(-0.5)),
            ("R", lambda p: p.ry.nudge(+0.5)),
            ("T", lambda p: p.ry.nudge(-0.5)),
            ("Y", lambda p: p.rz.nudge(+0.5)),
            ("U", lambda p: p.rz.nudge(-0.5)),
            ("+", lambda p: p.scale.setValue(p.scale.value() * 1.02)),
            ("-", lambda p: p.scale.setValue(p.scale.value() / 1.02)),
        ]
        for key, fn in bindings:
            sc(QtGui.QKeySequence(key), self).activated.connect(nudge_pose(fn))

        def on_space():
            if self._n_views > 1:
                self._cycle_view(+1)
            else:
                self.atlas_view.toggle_label_overlay()

        def on_shift_space():
            if self._n_views > 1:
                self._cycle_view(-1)
            else:
                self.atlas_view.toggle_label_overlay()

        # ApplicationShortcut context so Space works no matter which widget
        # has focus (sliders, checkboxes, dock titles all swallow Space by
        # default with the WindowShortcut context).
        space_sc = sc(QtGui.QKeySequence("Space"), self)
        space_sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        space_sc.activated.connect(on_space)

        shift_space_sc = sc(QtGui.QKeySequence("Shift+Space"), self)
        shift_space_sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        shift_space_sc.activated.connect(on_shift_space)

        # Also expose the cycle on Tab / Shift+Tab as a backup — some users
        # reach for Tab naturally, and Tab is less likely to be grabbed by
        # slider focus.
        tab_sc = sc(QtGui.QKeySequence("Tab"), self)
        tab_sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        tab_sc.activated.connect(on_space)
        shift_tab_sc = sc(QtGui.QKeySequence("Shift+Tab"), self)
        shift_tab_sc.setContext(QtCore.Qt.ShortcutContext.ApplicationShortcut)
        shift_tab_sc.activated.connect(on_shift_space)
        def toggle_all_overlays():
            self.atlas_view.toggle_label_overlay()
            self.overlay_view.toggle_atlas_overlay()

        sc(QtGui.QKeySequence("O"), self).activated.connect(toggle_all_overlays)
        sc(QtGui.QKeySequence("L"), self).activated.connect(
            self.overlay_view.toggle_atlas_labels
        )

        def toggle_docks():
            # Simple "focus mode" — hide pose + IoU docks so the central
            # image fills the window. Press F again to bring them back.
            visible = self._pose_dock.isVisible()
            self._pose_dock.setVisible(not visible)
            self._iou_dock.setVisible(not visible)

        sc(QtGui.QKeySequence("F"), self).activated.connect(toggle_docks)

        # Reset zoom back to "fit image" — handy after aggressive Ctrl+scroll.
        zoom_reset_keys = (QtCore.Qt.Key.Key_Home, QtCore.Qt.Key.Key_0)
        for key in zoom_reset_keys:
            sc(QtGui.QKeySequence(key), self).activated.connect(
                self.overlay_view.reset_zoom
            )
        sc(QtGui.QKeySequence("Ctrl+S"), self).activated.connect(
            self._on_save_pose
        )
        sc(QtGui.QKeySequence("Ctrl+E"), self).activated.connect(
            self._on_export
        )

    def _on_pose_changed(self, pose: Pose) -> None:
        # Pose is shared across views. Switching views (Space/Tab) only
        # changes which Z-slice is rasterised; the transform is whatever
        # the user has dialled in on the sliders right now.
        self._debounce.start(PROJECTION_DEBOUNCE_MS)

    def _on_atlas_dragged(self, dx_px: float, dy_px: float) -> None:
        """Shift the atlas by a mouse-drag pixel delta."""
        self.pose_controls.tx.nudge(dx_px)
        self.pose_controls.ty.nudge(dy_px)

    def _on_atlas_scale(self, factor: float) -> None:
        """Resize the atlas by the given multiplicative factor."""
        self.pose_controls.scale.setValue(
            self.pose_controls.scale.value() * float(factor)
        )

    def _fire_recompute(self) -> None:
        pose = self.pose_controls.pose()
        self.request_recompute.emit(
            pose, self.iou_panel.threshold(), self._current_view
        )

    def _cycle_view(self, step: int = 1) -> None:
        if self._n_views <= 1:
            return
        new_view = (self._current_view + step) % self._n_views
        self._current_view = new_view
        name = self._view_names[new_view] if new_view < len(self._view_names) \
            else f"slice_{new_view}"
        self._status.showMessage(
            f"View {new_view + 1}/{self._n_views}: {name}", 3000
        )
        if hasattr(self, "_view_combo"):
            self._view_combo.blockSignals(True)
            self._view_combo.setCurrentIndex(new_view)
            self._view_combo.blockSignals(False)
        # Pose stays put — just trigger a re-projection against the new slice.
        self._debounce.start(PROJECTION_DEBOUNCE_MS)

    def _install_view_toolbar(self) -> None:
        """Add a visible toolbar with prev/next view buttons + dropdown.

        Keyboard shortcuts can be swallowed by focused widgets; a toolbar
        button is unambiguous — click it and the view changes.
        """
        if self._n_views <= 1:
            return
        toolbar = self.addToolBar("View")
        toolbar.setMovable(False)

        prev_btn = QtGui.QAction("◀ Prev view", self)
        prev_btn.setShortcut(QtGui.QKeySequence("PgUp"))
        prev_btn.triggered.connect(lambda: self._cycle_view(-1))
        toolbar.addAction(prev_btn)

        self._view_combo = QtWidgets.QComboBox(toolbar)
        for i, n in enumerate(self._view_names):
            self._view_combo.addItem(f"{i + 1}. {n}", userData=i)
        self._view_combo.setCurrentIndex(self._current_view)
        self._view_combo.currentIndexChanged.connect(self._on_view_combo_changed)
        toolbar.addWidget(self._view_combo)

        next_btn = QtGui.QAction("Next view ▶", self)
        next_btn.setShortcut(QtGui.QKeySequence("PgDown"))
        next_btn.triggered.connect(lambda: self._cycle_view(+1))
        toolbar.addAction(next_btn)

        toolbar.addSeparator()

        show_right = QtGui.QAction("Show atlas-only pane", self)
        show_right.setCheckable(True)
        show_right.setChecked(not self._is_door_2d)
        show_right.triggered.connect(self._on_show_right_pane)
        toolbar.addAction(show_right)

        toolbar.addSeparator()

        # Explicit zoom slider/spinbox so the user can see & set the current
        # zoom factor. 1.0 = fit-image; higher = zoomed in.
        toolbar.addWidget(QtWidgets.QLabel(" Zoom"))
        self._zoom_spin = QtWidgets.QDoubleSpinBox()
        self._zoom_spin.setRange(0.1, 20.0)
        self._zoom_spin.setSingleStep(0.1)
        self._zoom_spin.setDecimals(2)
        self._zoom_spin.setValue(1.0)
        self._zoom_spin.setToolTip(
            "1.0 = fit image. Higher = more zoomed in. "
            "Also: Ctrl+scroll / Home / 0 / mouse wheel."
        )
        self._zoom_spin.valueChanged.connect(self._on_zoom_spin_changed)
        toolbar.addWidget(self._zoom_spin)

        reset_zoom = QtGui.QAction("Reset zoom", self)
        reset_zoom.triggered.connect(self.overlay_view.reset_zoom)
        reset_zoom.triggered.connect(lambda: self._sync_zoom_spin())
        toolbar.addAction(reset_zoom)

        # Keep the spinbox in sync if the user zooms via mouse wheel.
        self._zoom_sync_timer = QtCore.QTimer(self)
        self._zoom_sync_timer.setInterval(120)
        self._zoom_sync_timer.timeout.connect(self._sync_zoom_spin)
        self._zoom_sync_timer.start()

        toolbar.addSeparator()

        # Opacity slider for the atlas overlay.
        toolbar.addWidget(QtWidgets.QLabel(" Atlas opacity"))
        self._opacity_slider = QtWidgets.QSlider(
            QtCore.Qt.Orientation.Horizontal, toolbar
        )
        self._opacity_slider.setMinimum(0)
        self._opacity_slider.setMaximum(100)
        self._opacity_slider.setValue(45)
        self._opacity_slider.setFixedWidth(140)
        self._opacity_slider.valueChanged.connect(
            lambda v: self.overlay_view.set_atlas_overlay_opacity(v / 100.0)
        )
        toolbar.addWidget(self._opacity_slider)

        toolbar.addSeparator()

        # "Draw ROI" toolbar action — any time a drawn-ROI response
        # source is available (activity maps OR ΔF/F extractor).
        if self._activity_maps is not None or self._dff_extractor is not None:
            draw_action = QtGui.QAction("Draw ROI", self)
            draw_action.setShortcut(QtGui.QKeySequence("D"))
            draw_action.setToolTip(
                "Left-click to add vertices, right-click to finish, "
                "Esc to cancel."
            )
            draw_action.triggered.connect(self._on_start_drawing)
            toolbar.addAction(draw_action)

        toolbar.addSeparator()

        # Log-file pointer so the user knows where to grab error output.
        from door_toolkit.atlas_align.config import LOG_FILE
        log_action = QtGui.QAction("Open log folder", self)
        log_action.triggered.connect(lambda: self._open_log_folder(LOG_FILE))
        toolbar.addAction(log_action)

        toolbar.addSeparator()
        hint = QtWidgets.QLabel(
            "  Space / Tab = next view   ·   O = overlay   ·   L = labels   ·   F = hide docks  "
        )
        toolbar.addWidget(hint)

    def _on_zoom_spin_changed(self, value: float) -> None:
        self.overlay_view.set_zoom_factor(value)

    def _sync_zoom_spin(self) -> None:
        """Pull the current zoom from the view and push to the spinbox."""
        if not hasattr(self, "_zoom_spin"):
            return
        factor = float(self.overlay_view.current_zoom_factor())
        current = self._zoom_spin.value()
        if abs(factor - current) < 0.005:
            return
        self._zoom_spin.blockSignals(True)
        self._zoom_spin.setValue(factor)
        self._zoom_spin.blockSignals(False)

    def _open_log_folder(self, log_file: "Path") -> None:
        import subprocess
        folder = Path(log_file).parent
        folder.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", str(folder)])
        except Exception:  # noqa: BLE001
            self._status.showMessage(f"Log folder: {folder}", 5000)

    def _on_show_right_pane(self, checked: bool) -> None:
        if checked:
            self.atlas_view.show()
            self._splitter.setSizes([1000, 600])
            # Link + push a fresh projection now that the pane is visible.
            self.atlas_view.link_to(self.overlay_view.view_box)
            if self._last_payload is not None:
                self.atlas_view.set_projection(
                    self._last_payload.grayscale_mip,
                    self._last_payload.label_projection,
                    self._bundle.labels,
                )
        else:
            # Unlink so the hidden pane's autoRange can't reach back into
            # the overlay pane.
            self.atlas_view.view_box.setXLink(None)
            self.atlas_view.view_box.setYLink(None)
            self.atlas_view.hide()
            self._splitter.setSizes([1600, 0])

    def _on_view_combo_changed(self, idx: int) -> None:
        if idx == self._current_view or idx < 0:
            return
        step = (idx - self._current_view) % max(self._n_views, 1)
        self._cycle_view(step)

    def _on_threshold_changed(self, _value: float) -> None:
        self._debounce.start(PROJECTION_DEBOUNCE_MS)

    def _on_roi_selected(self, roi_index: int) -> None:
        """Push the selected ROI's ΔF/F summary + DoOR comparison into the
        Response dock.

        ``roi_index`` is an index into the overlay's combined display list
        (loaded FIJI ROIs first, then drawn polygons). If the click lands
        on a drawn polygon we route it to :meth:`_on_drawn_clicked` so the
        Drawn-ROIs dock gets selected and the Response dock updates.
        """
        n_fiji = len(self._roi_set)
        if roi_index >= n_fiji:
            drawn_pos = roi_index - n_fiji
            if 0 <= drawn_pos < len(self._drawn_rois):
                drawn_idx = self._drawn_rois[drawn_pos].index
                if self.drawn_panel is not None:
                    self.drawn_panel.select_by_drawn_index(drawn_idx)
                self._on_drawn_clicked(drawn_idx)
            return
        if self.response_panel is None or self._dff_bundle is None:
            return
        if not (0 <= roi_index < n_fiji):
            self.response_panel.clear()
            return
        self._active_roi_index = roi_index
        roi = self._roi_set[roi_index]
        cx, cy = roi.centroid
        header = f"{roi.name}   [FIJI #{roi_index}]"
        subheader = f"centroid (x={cx:.0f}, y={cy:.0f})"
        dff_idx = self._fiji_to_dff.get(roi_index)
        if dff_idx is None:
            self.response_panel.show_no_dff_data(header)
            return
        matched_col = self._dff_bundle.roi_columns[dff_idx]
        subheader += (
            f"   ·   dff match: {matched_col.header} "
            f"(x={matched_col.x:.0f}, y={matched_col.y:.0f})"
        )
        summary = self._dff_bundle.summary_for_roi(dff_idx)
        self.response_panel.show_roi_summary(
            roi_name=header,
            roi_extra=subheader,
            summary=summary,
            **self._door_comparison_for_roi(roi_index, dff_idx),
        )

    def _door_comparison_for_roi(
        self, fiji_roi_index: int, dff_idx: int
    ) -> dict:
        """Return the DoOR-related kwargs that :meth:`ResponsePanel.show_roi_summary`
        expects for the given ROI."""
        if self._door_bundle is None:
            return {}
        import numpy as np

        user_max = self._dff_bundle.max_vector_for_roi(dff_idx)
        # DoOR responses for the ROI's manual glomerulus pick (if any).
        manuals = self.iou_panel.manual_assignments()
        manual_glom = manuals.get(fiji_roi_index)
        door_responses = None
        manual_sim = None
        manual_cosine = None
        manual_receptor = None
        if manual_glom:
            door_vec = self._door_bundle.response_for_glomerulus(manual_glom)
            if door_vec is not None:
                door_responses = {
                    odor: (float(door_vec[i]) if np.isfinite(door_vec[i]) else float("nan"))
                    for i, odor in enumerate(self._door_bundle.odor_order)
                }
                # Pearson = pattern similarity (what you actually want).
                manual_sim = pearson_correlation(user_max, door_vec)
                # Cosine shown alongside for reference / debugging.
                manual_cosine = cosine_similarity(user_max, door_vec)
                manual_receptor = (
                    self._door_bundle.receptor_by_glomerulus.get(manual_glom)
                )
        # Top-3 ranked by Pearson (cosine is too lenient on positive-only data).
        suggestions = rank_glomeruli_by_similarity(
            user_max, self._door_bundle, top_n=3, metric="pearson"
        )
        # DoOR-calibrated per-odor scalar for this ROI — row / (max - min).
        scaled = door_row_scale(np.asarray(user_max, dtype=np.float64))
        user_door_scaled = {
            odor: float(scaled[i])
            for i, odor in enumerate(self._door_bundle.odor_order)
            if i < scaled.size
        }
        # DoOR-projected: monotonic-fit port of project_points/back_project
        # against the manually-assigned glomerulus' DoOR vector.
        user_door_projected: Optional[Dict[str, float]] = None
        projection_model: Optional[str] = None
        projection_rms: Optional[float] = None
        if manual_glom is not None:
            dvec_manual = self._door_bundle.response_for_glomerulus(manual_glom)
            if dvec_manual is not None:
                proj = project_to_door_scale(
                    np.asarray(user_max, dtype=np.float64),
                    np.asarray(dvec_manual, dtype=np.float64),
                )
                if proj.ok:
                    user_door_projected = {
                        odor: float(proj.projected[i])
                        for i, odor in enumerate(self._door_bundle.odor_order)
                        if i < proj.projected.size
                        and np.isfinite(proj.projected[i])
                    }
                    projection_model = proj.model_name
                    projection_rms = proj.rms
        return dict(
            door_responses=door_responses,
            manual_glomerulus=manual_glom,
            manual_receptor=manual_receptor,
            manual_similarity=manual_sim,
            manual_cosine=manual_cosine,
            suggestions=suggestions,
            user_door_scaled=user_door_scaled,
            user_door_projected=user_door_projected,
            projection_model=projection_model,
            projection_rms=projection_rms,
        )

    # ---------------------------------------------- drawn-ROI workflow

    def _on_start_drawing(self) -> None:
        """Enter polygon-draw mode on the overlay view."""
        if self._activity_maps is None and self._dff_extractor is None:
            return
        self.overlay_view.start_drawing()
        self._status.showMessage(
            "Draw mode: left-click to add vertices, right-click to finish, Esc to cancel.",
            5000,
        )

    def _on_polygon_drawn(self, xs, ys) -> None:  # xs, ys np.ndarray
        """Called when the user finishes drawing a polygon."""
        if self._activity_maps is None and self._dff_extractor is None:
            return
        idx = self._next_drawn_index
        self._next_drawn_index += 1
        # Fast initial response from the activity map (if we have one) —
        # the extractor's real ΔF/F overwrites this when it finishes.
        initial_response: Optional[np.ndarray] = None
        if self._activity_maps is not None:
            initial_response = self._activity_maps.response_for_polygon(xs, ys)
        roi = DrawnROI(index=idx, xs=xs, ys=ys, response=initial_response)
        self._drawn_rois.append(roi)
        if self.drawn_panel is not None:
            self.drawn_panel.add_roi(roi)
        self._redraw_drawn_overlays()
        self._renumber_drawn_rois()
        self._active_drawn_index = idx
        self._refresh_response_for_drawn(idx)

        # If the real-ΔF/F extractor is wired up, kick off a background
        # worker — bleach-corrected ΔF/F is slow (first draw loads the
        # memmap cache). When it finishes, the ROI's response vector is
        # replaced with real peak ΔF/F per odor and the panels refresh.
        if self._dff_extractor is not None:
            self._set_dff_status(
                idx,
                state="loading",
                text=(
                    f"⏳ Drawn ROI #{idx}: computing bleach-corrected "
                    f"ΔF/F from raw stacks…  (first-draw loads all 7 "
                    f"odor movies; subsequent draws are fast)"
                ),
            )
            self._run_dff_extraction_async(idx, xs, ys)
        else:
            self._status.showMessage(
                f"Drawn ROI #{idx} — {len(xs)} vertices  ·  "
                f"response computed across {self._activity_maps.n_odors} odors.",
                5000,
            )

    def _set_dff_status(
        self,
        drawn_index: int,
        *,
        state: str,
        text: str,
    ) -> None:
        """Keep the user posted on where ΔF/F extraction is for a drawn ROI.

        ``state`` is one of ``loading`` / ``done`` / ``failed`` — controls
        the colour used for the status-bar message and the Drawn ROIs
        table's Top-1 column.
        """
        colour_css = {
            "loading": "background-color: #3a3a00; color: #ffe066;",
            "done":    "background-color: #103a10; color: #66ff66;",
            "failed":  "background-color: #3a1010; color: #ff8080;",
        }.get(state, "")
        self._status.setStyleSheet(colour_css)
        # 0ms = persistent until replaced; finite ms = auto-clear.
        timeout_ms = 0 if state == "loading" else 5000
        self._status.showMessage(text, timeout_ms)
        # Also annotate the Drawn ROIs table's current state marker.
        if self.drawn_panel is not None:
            marker = {
                "loading": "⏳ loading…",
                "done":    "✓ ΔF/F",
                "failed":  "✗ failed",
            }.get(state, "")
            if marker:
                self.drawn_panel.set_top_match(drawn_index, marker)

    # --------------------------------------------- async ΔF/F extraction

    def _run_dff_extraction_async(
        self, drawn_index: int, xs, ys,
    ) -> None:
        """Compute real ΔF/F for the given polygon off the GUI thread.

        Uses the same ``DrawnROIDffExtractor`` instance across all
        drawn ROIs — its memmap cache means subsequent polygons don't
        pay the stack-load cost.
        """
        extractor = self._dff_extractor
        if extractor is None:
            return

        thread = QtCore.QThread(self)
        worker = DrawnROIDffWorker(extractor, drawn_index, xs, ys)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_dff_extraction_done)
        worker.failed.connect(self._on_dff_extraction_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(thread.deleteLater)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        # Keep a reference so the thread + worker objects aren't GC'd.
        self._drawn_dff_threads = getattr(self, "_drawn_dff_threads", [])
        self._drawn_dff_threads.append((thread, worker))
        logger.info(
            "Dispatching ΔF/F worker thread for drawn ROI #%d", drawn_index
        )
        thread.start()

    def _on_dff_extraction_done(
        self, drawn_index: int, result: DrawnROIDff
    ) -> None:
        # Store the full trace result for potential export / plotting.
        self._drawn_dff_results[drawn_index] = result
        # Overwrite the ROI's response vector with real peak ΔF/F per odor.
        for roi in self._drawn_rois:
            if roi.index == drawn_index:
                roi.response = result.scalar_dff_vector()
                break
        if self._active_drawn_index == drawn_index:
            self._refresh_response_for_drawn(drawn_index)
        self._set_dff_status(
            drawn_index,
            state="done",
            text=(
                f"✓ Drawn ROI #{drawn_index}: bleach-corrected ΔF/F ready "
                f"({len(result.odor_order)} odors). Response dock + DoOR "
                f"suggestions have refreshed."
            ),
        )

    def _on_dff_extraction_failed(
        self, drawn_index: int, message: str
    ) -> None:
        self._set_dff_status(
            drawn_index,
            state="failed",
            text=f"✗ ΔF/F extraction failed for drawn ROI #{drawn_index}: {message}",
        )
        logger.error("Drawn ROI #%d ΔF/F extraction failed: %s", drawn_index, message)

    def _redraw_drawn_overlays(self) -> None:
        """Update the overlay's visible ROI set to include drawn ROIs after
        the loaded FIJI ROIs (so they appear on top of the reference)."""
        # Build a combined list of ROI-like objects for OverlayView.
        from door_toolkit.atlas_align.io.roi_loader import ROI as FijiROI

        combined: List[FijiROI] = list(self._roi_set)
        for d in self._drawn_rois:
            combined.append(FijiROI(
                name=f"drawn_{d.index}",
                x=d.xs,
                y=d.ys,
                roi_type="polygon",
                index=len(combined),
            ))
        self.overlay_view.set_rois(combined)

    def _on_delete_drawn(self, drawn_index: int) -> None:
        self._drawn_rois = [
            r for r in self._drawn_rois if r.index != drawn_index
        ]
        if self.drawn_panel is not None:
            self.drawn_panel.remove_roi(drawn_index)
        self._redraw_drawn_overlays()
        self._renumber_drawn_rois()

    def _on_drawn_clicked(self, drawn_index: int) -> None:
        self._active_drawn_index = drawn_index
        self._refresh_response_for_drawn(drawn_index)

    def _on_drawn_manual_changed(
        self, drawn_index: int, glomerulus_name: str
    ) -> None:
        self._refresh_response_for_drawn(drawn_index)

    def _on_drawn_side_changed(self, drawn_index: int, side: str) -> None:
        """User toggled L/R in the Drawn ROIs table — renumber + relabel."""
        new_side = side if side in ("L", "R") else None
        for roi in self._drawn_rois:
            if roi.index == drawn_index:
                roi.side = new_side
                break
        self._renumber_drawn_rois()

    def _renumber_drawn_rois(self) -> None:
        """Compute per-side 1-indexed display labels and push them to the
        Drawn-ROIs table + the text-label overlay.

        Numbering is independent per side: L-ROIs are ``L1, L2, …``,
        R-ROIs are ``R1, R2, …``, and ROIs with no side get plain
        ``1, 2, …`` (in draw order within their group).
        """
        counters: Dict[Optional[str], int] = {"L": 0, "R": 0, None: 0}
        overlay_labels: List[Tuple[str, float, float]] = []
        for roi in self._drawn_rois:
            side = roi.side if roi.side in ("L", "R") else None
            counters[side] = counters[side] + 1
            n = counters[side]
            label = f"{side}{n}" if side else f"{n}"
            roi.display_label = label
            cx, cy = float(roi.xs.mean()), float(roi.ys.mean())
            overlay_labels.append((label, cx, cy))
            if self.drawn_panel is not None:
                self.drawn_panel.set_display_label(roi.index, label)
        self.overlay_view.set_drawn_labels(overlay_labels)

    def _on_drawn_load(self) -> None:
        """Load previously-exported drawn ROIs from a CSV and re-compute
        their ΔF/F from the raw stacks.

        The saved polygon geometry + side + manual glomerulus pick are
        read back, but any saved per-odor response values in the CSV
        are **intentionally discarded** — each loaded polygon is fed
        to the same :class:`DrawnROIDffWorker` that handles freshly-drawn
        ROIs, so the Response dock is populated by a fresh bleach-corrected
        ΔF/F + DoOR scalar extraction from the current raw movies.
        """
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load drawn ROIs", "", "CSV (*.csv)"
        )
        if not path:
            return
        try:
            loaded, _odor_cols = load_drawn_rois_csv(Path(path))
        except Exception as e:  # noqa: BLE001
            logger.exception("Failed to load drawn-ROIs CSV %s", path)
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))
            return
        if not loaded:
            self._status.showMessage(f"No usable rows found in {path}", 5000)
            return

        first_idx: Optional[int] = None
        n_added = 0
        for loaded_roi in loaded:
            idx = self._next_drawn_index
            self._next_drawn_index += 1
            # Seed the new ROI's response from the activity map if we have
            # one — cheap & instant so the Response dock has something to
            # show while the real ΔF/F worker is still loading the stack.
            initial_response: Optional[np.ndarray] = None
            if self._activity_maps is not None:
                initial_response = self._activity_maps.response_for_polygon(
                    loaded_roi.xs, loaded_roi.ys
                )
            roi = DrawnROI(
                index=idx,
                xs=loaded_roi.xs,
                ys=loaded_roi.ys,
                response=initial_response,
                manual_glomerulus=loaded_roi.manual_glomerulus,
                side=loaded_roi.side,
            )
            self._drawn_rois.append(roi)
            if self.drawn_panel is not None:
                self.drawn_panel.add_roi(roi)
            if first_idx is None:
                first_idx = idx
            n_added += 1

        self._redraw_drawn_overlays()
        self._renumber_drawn_rois()

        # Kick off real bleach-corrected ΔF/F extraction per loaded ROI.
        # The worker cache means the first polygon pays the 7× stack-load
        # cost and subsequent polygons are fast.
        if self._dff_extractor is not None:
            for roi in self._drawn_rois[-n_added:]:
                self._set_dff_status(
                    roi.index,
                    state="loading",
                    text=(
                        f"⏳ Drawn ROI #{roi.index}: recomputing bleach-corrected "
                        f"ΔF/F from raw stacks…"
                    ),
                )
                self._run_dff_extraction_async(roi.index, roi.xs, roi.ys)
            self._status.showMessage(
                f"Loaded {n_added} drawn ROIs from {path} — "
                f"re-running ΔF/F extraction in background.",
                6000,
            )
        else:
            self._status.showMessage(
                f"Loaded {n_added} drawn ROIs from {path} "
                f"(no ΔF/F extractor wired up — response uses activity map).",
                6000,
            )

        if first_idx is not None:
            self._active_drawn_index = first_idx
            self._refresh_response_for_drawn(first_idx)

    def _refresh_response_for_drawn(self, drawn_index: int) -> None:
        """Push the drawn ROI's response + DoOR comparison into the
        Response dock (reusing the existing panel infrastructure)."""
        if self.response_panel is None:
            return
        if self._activity_maps is None and self._dff_extractor is None:
            return
        roi = next(
            (r for r in self._drawn_rois if r.index == drawn_index), None
        )
        if roi is None:
            return

        # Resolve odor order + data-source description.
        if drawn_index in self._drawn_dff_results:
            extract = self._drawn_dff_results[drawn_index]
            odor_order = extract.odor_order
            source_desc = f"bleach-corrected ΔF/F from raw stacks"
        elif self._activity_maps is not None:
            odor_order = self._activity_maps.odor_order
            source_desc = f"activity map: {self._activity_maps.source_dir.name}"
        elif self._dff_extractor is not None:
            odor_order = self._dff_extractor.ready_odors
            source_desc = "ΔF/F pending (worker still loading stacks)…"
        else:
            odor_order = []
            source_desc = ""

        cx, cy = roi.centroid
        header = f"Drawn ROI #{drawn_index}"
        subheader = (
            f"centroid (x={cx:.0f}, y={cy:.0f})  ·  "
            f"{len(roi.xs)} vertices  ·  {source_desc}"
        )

        if roi.response is None or len(odor_order) == 0:
            self.response_panel.show_no_dff_data(header)
            return
        summary = {
            odor: (0.0, float(val))
            for odor, val in zip(odor_order, roi.response)
        }
        # DoOR comparison.
        manual_glom = roi.manual_glomerulus
        door_responses = None
        manual_sim = None
        manual_cosine = None
        manual_receptor = None
        suggestions: List[Tuple[str, float]] = []
        if self._door_bundle is not None:
            if manual_glom:
                dvec = self._door_bundle.response_for_glomerulus(manual_glom)
                if dvec is not None:
                    door_responses = {
                        o: (float(dvec[i]) if np.isfinite(dvec[i]) else float("nan"))
                        for i, o in enumerate(self._door_bundle.odor_order)
                    }
                    manual_sim = pearson_correlation(roi.response, dvec)
                    manual_cosine = cosine_similarity(roi.response, dvec)
                    manual_receptor = (
                        self._door_bundle.receptor_by_glomerulus.get(manual_glom)
                    )
            suggestions = rank_glomeruli_by_similarity(
                roi.response, self._door_bundle, top_n=3, metric="pearson"
            )
            if suggestions and self.drawn_panel is not None:
                top_name, _top_r = suggestions[0]
                self.drawn_panel.set_top_match(drawn_index, top_name)

        # DoOR-calibrated per-odor vector for this drawn ROI (pattern only).
        scaled_vec = door_row_scale(
            np.asarray(roi.response, dtype=np.float64)
        )
        user_door_scaled = {
            odor: float(scaled_vec[i])
            for i, odor in enumerate(odor_order)
            if i < scaled_vec.size
        }

        # DoOR-projected: requires the manual glomerulus pick. We align
        # our odor_order with DoOR's per-receptor vector so the monotonic
        # fit sees matched odor pairs.
        user_door_projected: Optional[Dict[str, float]] = None
        projection_model: Optional[str] = None
        projection_rms: Optional[float] = None
        if manual_glom is not None and self._door_bundle is not None:
            dvec_manual = self._door_bundle.response_for_glomerulus(manual_glom)
            if dvec_manual is not None:
                # Map our drawn-ROI odor_order into DoOR's odor index space.
                door_order = self._door_bundle.odor_order
                idx_in_door = [
                    door_order.index(o) if o in door_order else -1
                    for o in odor_order
                ]
                if any(i >= 0 for i in idx_in_door):
                    door_aligned = np.full(
                        len(odor_order), np.nan, dtype=np.float64
                    )
                    for k, i in enumerate(idx_in_door):
                        if i >= 0:
                            v = float(dvec_manual[i])
                            if np.isfinite(v):
                                door_aligned[k] = v
                    proj = project_to_door_scale(
                        np.asarray(roi.response, dtype=np.float64),
                        door_aligned,
                    )
                    if proj.ok:
                        user_door_projected = {
                            odor: float(proj.projected[i])
                            for i, odor in enumerate(odor_order)
                            if i < proj.projected.size
                            and np.isfinite(proj.projected[i])
                        }
                        projection_model = proj.model_name
                        projection_rms = proj.rms

        self.response_panel.show_roi_summary(
            roi_name=header,
            roi_extra=subheader,
            summary=summary,
            door_responses=door_responses,
            manual_glomerulus=manual_glom,
            manual_receptor=manual_receptor,
            manual_similarity=manual_sim,
            manual_cosine=manual_cosine,
            suggestions=suggestions,
            user_door_scaled=user_door_scaled,
            user_door_projected=user_door_projected,
            projection_model=projection_model,
            projection_rms=projection_rms,
        )

    def _on_drawn_export(self) -> None:
        if self.drawn_panel is None:
            return
        if self._activity_maps is not None:
            odor_order = self._activity_maps.odor_order
        elif self._dff_extractor is not None:
            odor_order = self._dff_extractor.ready_odors
        else:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export drawn ROIs", "drawn_rois.csv", "CSV (*.csv)"
        )
        if not path:
            return
        n = self.drawn_panel.write_csv(Path(path), odor_order)
        self._status.showMessage(f"Exported {n} drawn ROIs → {path}", 5000)

    def _on_door_suggestion_clicked(self, glomerulus_name: str) -> None:
        """Apply a clicked DoOR suggestion to the active ROI's manual
        assignment. Refuses to overwrite a locked row."""
        # If the active ROI is a drawn polygon, route the pick to the
        # Drawn-ROI panel instead of the Assignments panel.
        drawn_idx = getattr(self, "_active_drawn_index", None)
        if drawn_idx is not None and any(
            r.index == drawn_idx for r in self._drawn_rois
        ):
            if self.drawn_panel is None:
                return
            if self.drawn_panel.is_locked(drawn_idx):
                self._status.showMessage(
                    f"Drawn ROI #{drawn_idx} is locked — uncheck the Lock "
                    f"column to apply a new DoOR suggestion.",
                    5000,
                )
                return
            if not self.drawn_panel.try_set_manual(drawn_idx, glomerulus_name):
                return
            self._refresh_response_for_drawn(drawn_idx)
            return

        idx = getattr(self, "_active_roi_index", None)
        if idx is None:
            return
        if self.iou_panel.is_locked(int(idx)):
            self._status.showMessage(
                f"ROI #{idx} is locked — uncheck the Lock column to apply "
                f"a new DoOR suggestion.",
                5000,
            )
            return
        # Let IoUPanel know — this also fires ``manual_assignment_changed``
        # which re-colours the overlay.
        current = self.iou_panel.manual_assignments()
        current[int(idx)] = glomerulus_name
        self.iou_panel.set_manual_assignments(current)
        # Force the panel to refresh so the combobox reflects the new pick.
        if self._last_payload is not None:
            self.iou_panel.update_result(self._last_payload.assignment)
        # Refresh Response dock with the new manual pick.
        self._on_roi_selected(int(idx))
        # Recolour the overlay immediately.
        self._on_manual_assignment_changed(int(idx), glomerulus_name)

    def _on_manual_assignment_changed(
        self, roi_index: int, glomerulus_name: str
    ) -> None:
        """React to the user picking a glomerulus in the Manual column.

        Updates the overlay colour for that ROI immediately so the user
        sees the effect of their pick without waiting for the next
        projection, and refreshes the Response dock's DoOR comparison if
        the active ROI is the one that changed.
        """
        if (
            getattr(self, "_active_roi_index", None) == roi_index
            and self.response_panel is not None
        ):
            # Re-build the Response panel with fresh DoOR kwargs.
            self._on_roi_selected(roi_index)
        manuals = self.iou_panel.manual_assignments()
        roi_to_name: Dict[int, Optional[str]] = {}
        for idx in range(len(self._roi_set)):
            roi_to_name[idx] = manuals.get(idx) or None
        # Only overwrite per-ROI colours with manual picks when they exist;
        # ROIs not yet picked keep their auto IoU colour (from last payload).
        if self._last_payload is not None:
            for a in self._last_payload.assignment.assignments:
                if a.roi_index in manuals:
                    continue  # manual wins
                if a.above_threshold:
                    roi_to_name[a.roi_index] = a.glomerulus_name
                else:
                    roi_to_name[a.roi_index] = None
        self.overlay_view.update_assignments(roi_to_name)

    def _on_projection_ready(self, payload: ProjectionPayload) -> None:
        self._last_payload = payload
        # Only push the projection into the right-pane atlas view when it's
        # actually visible — otherwise its autoRange propagates via the
        # view-link and nukes the user's zoom on the overlay pane.
        if self.atlas_view.isVisible():
            self.atlas_view.set_projection(
                payload.grayscale_mip,
                payload.label_projection,
                self._bundle.labels,
            )
        # Overlay the atlas on top of the reference image so the user can
        # judge the fit in one pane.
        self.overlay_view.set_atlas_overlay(
            payload.label_projection, self._bundle.labels
        )
        self.iou_panel.update_result(payload.assignment)

        # Manual picks override the IoU argmax whenever the user has chosen
        # a glomerulus for that ROI in the "Manual" column.
        manuals = self.iou_panel.manual_assignments()
        roi_to_name: Dict[int, Optional[str]] = {}
        for a in payload.assignment.assignments:
            if a.roi_index in manuals:
                roi_to_name[a.roi_index] = manuals[a.roi_index]
            elif a.above_threshold:
                roi_to_name[a.roi_index] = a.glomerulus_name
            else:
                roi_to_name[a.roi_index] = None
        self.overlay_view.update_assignments(roi_to_name)

        if self._n_views > 1:
            view_name = self._view_names[self._current_view]
            view_prefix = (
                f"view {self._current_view + 1}/{self._n_views} "
                f"'{view_name}'   "
            )
        else:
            view_prefix = ""

        if payload.elapsed_ms > PROJECTION_WARN_MS:
            self._status.showMessage(
                f"{view_prefix}slow projection: {payload.elapsed_ms:.0f} ms "
                f"(mean IoU {payload.assignment.mean_iou:.3f})"
            )
        else:
            self._status.showMessage(
                f"{view_prefix}mean IoU {payload.assignment.mean_iou:.3f} "
                f"({payload.assignment.n_above_threshold} assigned / "
                f"{payload.assignment.n_below_threshold} unassigned)  "
                f"{payload.elapsed_ms:.0f} ms"
            )

    def _on_worker_failed(self, message: str) -> None:
        logger.error("Worker reported failure: %s", message)
        self._status.showMessage(f"Projection failed: {message}")
        QtWidgets.QMessageBox.critical(self, "Projection failed", message)

    def _on_auto_center(self) -> None:
        lm = self._bundle.labelmap
        nz = np.argwhere(lm > 0)
        if nz.size == 0:
            return
        cz, cy, cx = nz.mean(axis=0)
        ref_h, ref_w = self._reference_image.shape
        target_cx = (ref_w - 1) / 2.0
        target_cy = (ref_h - 1) / 2.0
        dx = target_cx - cx
        dy = target_cy - cy
        pose = self.pose_controls.pose()
        new_pose = Pose(
            tx=pose.tx + dx,
            ty=pose.ty + dy,
            tz=pose.tz,
            rx=pose.rx, ry=pose.ry, rz=pose.rz,
            sx=pose.sx, sy=pose.sy, sz=pose.sz,
            flip_x=pose.flip_x, flip_y=pose.flip_y, flip_z=pose.flip_z,
        )
        self.pose_controls.set_pose(new_pose, emit=True)

    def _on_save_pose(self) -> None:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save pose", filter="JSON (*.json)"
        )
        if not path:
            return
        self._write_pose(Path(path))

    def _write_pose(self, path: Path) -> None:
        atlas_hash = None
        if self._bundle.atlas_dir is not None:
            tif = self._bundle.atlas_dir / "flywire_al_labelmap.tif"
            if tif.is_file():
                atlas_hash = file_sha256(tif)
        ref_hash = (
            file_sha256(self._reference_path)
            if self._reference_path and self._reference_path.is_file()
            else None
        )
        save_pose(
            path,
            self.pose_controls.pose(),
            threshold=self.iou_panel.threshold(),
            atlas_hash=atlas_hash,
            reference_hash=ref_hash,
        )

    def _on_load_pose(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load pose", filter="JSON (*.json)"
        )
        if not path:
            return
        pose, meta = load_pose(Path(path))
        self.pose_controls.set_pose(pose, emit=True)
        if meta.get("threshold") is not None:
            self.iou_panel.set_threshold(meta["threshold"])

    def _on_export(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose export directory"
        )
        if not directory:
            return
        self.export_to_directory(Path(directory))

    def export_to_directory(self, out_dir: Path) -> None:
        """Write rois_assigned.zip + assignments.csv + manual_assignments.csv
        + pose.json into ``out_dir``."""
        if self._last_payload is None:
            self._status.showMessage("Projection not ready yet; try again.")
            return
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            export_roi_zip(
                out_dir / "rois_assigned.zip",
                list(self._roi_set),
                self._last_payload.assignment,
            )
            export_assignments_csv(
                out_dir / "assignments.csv", self._last_payload.assignment
            )
            self._write_manual_csv(out_dir / "manual_assignments.csv")
            self._write_pose(out_dir / "pose.json")
            self._status.showMessage(f"Exported to {out_dir}")
        except Exception as e:  # noqa: BLE001
            logger.exception("Export failed")
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def _write_manual_csv(self, path: Path) -> None:
        """Write a minimal manual-picks CSV: three columns, one row per
        ROI the user manually assigned a glomerulus to.

        Columns:
            roi_index, roi_original_name, manually_assigned_glomerulus
        """
        import csv

        manuals = self.iou_panel.manual_assignments()
        fields = [
            "roi_index",
            "roi_original_name",
            "manually_assigned_glomerulus",
        ]
        # Sort by ROI index so the file is stable/diff-friendly.
        rows = sorted(manuals.items(), key=lambda kv: int(kv[0]))
        with Path(path).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for roi_idx, glom in rows:
                writer.writerow({
                    "roi_index": int(roi_idx),
                    "roi_original_name": self._roi_names[roi_idx]
                        if 0 <= roi_idx < len(self._roi_names) else "",
                    "manually_assigned_glomerulus": glom,
                })
        logger.info("Wrote manual-assignments CSV: %s (%d rows)", path, len(rows))

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Catch Space / Tab / Escape no matter which child has focus."""
        key = event.key()
        mods = event.modifiers()
        if self.overlay_view.draw_mode_active:
            if key == QtCore.Qt.Key.Key_Escape:
                self.overlay_view.cancel_drawing()
                event.accept()
                return
            if key in (
                QtCore.Qt.Key.Key_Return,
                QtCore.Qt.Key.Key_Enter,
            ):
                self.overlay_view.finish_drawing()
                event.accept()
                return
        if key == QtCore.Qt.Key.Key_Space and self._n_views > 1:
            if mods & QtCore.Qt.KeyboardModifier.ShiftModifier:
                self._cycle_view(-1)
            else:
                self._cycle_view(+1)
            event.accept()
            return
        if key in (
            QtCore.Qt.Key.Key_Tab, QtCore.Qt.Key.Key_Backtab
        ) and self._n_views > 1:
            step = -1 if (
                key == QtCore.Qt.Key.Key_Backtab
                or (mods & QtCore.Qt.KeyboardModifier.ShiftModifier)
            ) else +1
            self._cycle_view(step)
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:  # noqa: N802
        try:
            self._worker_thread.quit()
            self._worker_thread.wait(2000)
        finally:
            super().closeEvent(event)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the atlas_align GUI."
    )
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument(
        "--reference", type=Path, default=None,
        help=(
            "Reference image the ROIs were drawn on. Optional when "
            "--activity-maps is given — the fullframe PNR is used as "
            "the canvas in that case."
        ),
    )
    parser.add_argument(
        "--rois", type=Path, default=None,
        help=(
            "Optional FIJI RoiManager .zip. Omit to launch with no "
            "pre-existing ROIs (useful when you just want to draw new "
            "ones on top of the activity map)."
        ),
    )
    parser.add_argument("--pose", type=Path, default=None)
    parser.add_argument(
        "--threshold", type=float, default=DEFAULT_IOU_THRESHOLD
    )
    parser.add_argument(
        "--dff-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing per-odor ΔF/F CSVs "
            "(dff_<letter>_trial_*.csv). When given, a Response dock is "
            "added that shows per-odor max/min ΔF/F for the selected ROI."
        ),
    )
    parser.add_argument(
        "--activity-maps",
        type=Path,
        default=None,
        help=(
            "Optional directory containing PNR_per_trial.tif + "
            "PNR_mean_fullframe.tif. Enables the 'Drawn ROIs' dock and "
            "lets you draw polygons on the fullframe map whose per-odor "
            "responses get computed from the activity stack."
        ),
    )
    parser.add_argument(
        "--fly-dir",
        type=Path,
        default=None,
        help=(
            "Optional top-level fly folder (e.g. /home/ramanlab/Lightsheet/fly_4) "
            "containing trial_*_OFM_* subdirs + analysis/al_roi/al_meta.json. "
            "When given, every drawn polygon triggers a real ΔF/F "
            "extraction from the cropped movie stacks (first trial per "
            "odor), bleach-corrected, with the per-odor peak fed into "
            "the DoOR comparison."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    configure_logging(verbose=args.verbose)

    # Route unhandled exceptions to the log file so every previous-run
    # failure is recoverable after the fact.
    def _excepthook(exc_type, exc_value, exc_tb):
        logger.error(
            "Unhandled exception",
            exc_info=(exc_type, exc_value, exc_tb),
        )
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _excepthook

    logger.info(
        "Launching atlas_align GUI: atlas=%s reference=%s rois=%s",
        args.atlas, args.reference, args.rois,
    )
    from door_toolkit.atlas_align.config import LOG_FILE
    logger.info("Log file: %s", LOG_FILE)

    bundle = load_atlas_bundle(args.atlas)

    # ROIs are now optional. Missing = start with an empty set so the
    # user can draw their own on the activity map.
    if args.rois is not None:
        roi_set = load_rois(args.rois)
    else:
        roi_set = ROISet(rois=[], source=Path())
        logger.info("No --rois supplied; starting with an empty ROI set.")

    # Reference image is optional when --activity-maps is given — we
    # fall back to the fullframe PNR once activity maps have loaded.
    # If neither is supplied, that's a usage error.
    reference: Optional[np.ndarray] = None
    if args.reference is not None:
        reference = _load_reference_image(args.reference)
    elif args.activity_maps is None:
        raise SystemExit(
            "Either --reference <tif> or --activity-maps <dir> must be "
            "provided so there is a canvas to display."
        )

    initial_pose: Optional[Pose] = None
    if args.pose is not None:
        initial_pose, _meta = load_pose(args.pose)

    dff_bundle: Optional[DFFBundle] = None
    if args.dff_dir is not None:
        try:
            dff_bundle = load_dff_directory(args.dff_dir)
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to load --dff-dir %s; Response dock will be disabled.",
                args.dff_dir,
            )

    activity_maps: Optional[ActivityMaps] = None
    if args.activity_maps is not None:
        target_order = dff_bundle.odor_order if dff_bundle is not None else None
        try:
            activity_maps = load_activity_maps(
                args.activity_maps, target_odor_order=target_order
            )
            # If no explicit --reference was given, use the fullframe PNR
            # as the display canvas; otherwise the user's reference wins.
            if reference is None and activity_maps.reference_fullframe is not None:
                reference = activity_maps.reference_fullframe.astype(
                    np.float32, copy=False
                )
                logger.info(
                    "Using fullframe activity-map reference (%s) as the display canvas.",
                    reference.shape,
                )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to load --activity-maps %s; Drawn-ROIs dock disabled.",
                args.activity_maps,
            )

    if reference is None:
        raise SystemExit(
            "--activity-maps did not contain a usable PNR_mean_fullframe.tif "
            "and no --reference was supplied. Nothing to display."
        )

    # DoOR multi-view atlases ship already at the reference shape (if the
    # builder was pointed at --reference). Single-shape atlases get the
    # old resize-reference path so the IoU grids still align.
    atlas_type = str(bundle.manifest.get("atlas_type", ""))
    atlas_shape_yx = (bundle.labelmap.shape[1], bundle.labelmap.shape[2])
    if reference.shape[:2] != atlas_shape_yx:
        if atlas_type.startswith("door_2d"):
            logger.warning(
                "Reference shape %s != atlas shape %s. Rebuild the DoOR "
                "atlas with --reference <ref.tif> so the grids align "
                "without resize distortion.",
                reference.shape[:2], atlas_shape_yx,
            )
        reference, roi_set = fit_reference_and_rois_to_atlas(
            reference, roi_set, atlas_shape_yx
        )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(
        sys.argv
    )
    dff_extractor: Optional[DrawnROIDffExtractor] = None
    if args.fly_dir is not None:
        try:
            odor_order_for_extractor = (
                dff_bundle.odor_order if dff_bundle is not None
                else (activity_maps.odor_order if activity_maps is not None
                      else list(["Apple_Cider_Vinegar","Benzaldehyde","Citral",
                                 "Ethyl_Butyrate","Hexanol","Linalool","3-Octanol"]))
            )
            dff_extractor = DrawnROIDffExtractor(
                fly_dir=args.fly_dir,
                odor_order=odor_order_for_extractor,
            )
        except Exception:  # noqa: BLE001
            logger.exception(
                "Failed to set up drawn-ROI ΔF/F extractor from --fly-dir %s.",
                args.fly_dir,
            )

    window = AtlasAlignMainWindow(
        bundle=bundle,
        reference_image=reference,
        roi_set=roi_set,
        initial_pose=initial_pose,
        threshold=args.threshold,
        reference_path=args.reference,
        dff_bundle=dff_bundle,
        activity_maps=activity_maps,
        dff_extractor=dff_extractor,
    )
    window.show()

    # PyQt event-loop entrypoint — bound instance method, takes no args.
    run_loop = getattr(app, "exec")
    return int(run_loop())


if __name__ == "__main__":
    sys.exit(main())
