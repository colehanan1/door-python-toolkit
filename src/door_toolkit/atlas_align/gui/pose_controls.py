"""
Pose controls widget
====================

:class:`PoseControlPanel` exposes the 10 DOFs of a
:class:`~door_toolkit.atlas_align.core.volume_transform.Pose` as
slider + spinbox pairs, plus flip toggles, reset / save / load, and an
"Auto-centre" button.

Signals:
    ``pose_changed(Pose)``: emitted whenever any control changes.
    ``auto_center_requested()``: user clicked the Auto-centre button.
    ``save_pose_requested()``: user clicked Save pose.
    ``load_pose_requested()``: user clicked Load pose.
"""

from __future__ import annotations

from typing import Optional

from PyQt6 import QtCore, QtWidgets

from door_toolkit.atlas_align.config import IDENTITY_POSE, get_logger
from door_toolkit.atlas_align.core.volume_transform import Pose

logger = get_logger(__name__)


# Slider ranges. Sliders store integers; we scale to float via _SLIDER_SCALE.
_SLIDER_SCALE = 10  # 1 slider tick = 0.1 in float units
_TRANS_RANGE = (-500.0, 500.0)
_ROT_RANGE = (-180.0, 180.0)
_SCALE_RANGE = (0.1, 5.0)


class _SliderSpinRow(QtWidgets.QWidget):
    """A horizontal (label, slider, spinbox) triplet bound to one float value."""

    value_changed = QtCore.pyqtSignal(float)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        initial: float,
        step: float = 0.1,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent=parent)
        self._minimum = minimum
        self._maximum = maximum
        self._step = step

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        self._label = QtWidgets.QLabel(label)
        self._label.setMinimumWidth(30)
        layout.addWidget(self._label)

        self._slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self._slider.setMinimum(int(minimum / step))
        self._slider.setMaximum(int(maximum / step))
        self._slider.setValue(int(initial / step))
        self._slider.setSingleStep(1)
        self._slider.setPageStep(max(1, int(10 * step / step)))
        layout.addWidget(self._slider, stretch=4)

        self._spin = QtWidgets.QDoubleSpinBox()
        self._spin.setDecimals(2)
        self._spin.setRange(minimum, maximum)
        self._spin.setSingleStep(step)
        self._spin.setValue(initial)
        self._spin.setMinimumWidth(80)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)

        self._suppress = False

    # ---- internal sync ----

    def _on_slider_changed(self, int_val: int) -> None:
        if self._suppress:
            return
        value = int_val * self._step
        self._suppress = True
        self._spin.setValue(value)
        self._suppress = False
        self.value_changed.emit(value)

    def _on_spin_changed(self, value: float) -> None:
        if self._suppress:
            return
        self._suppress = True
        self._slider.setValue(int(round(value / self._step)))
        self._suppress = False
        self.value_changed.emit(value)

    # ---- public ----

    def value(self) -> float:
        return self._spin.value()

    def setValue(self, value: float) -> None:
        clamped = max(self._minimum, min(self._maximum, float(value)))
        self._suppress = True
        self._spin.setValue(clamped)
        self._slider.setValue(int(round(clamped / self._step)))
        self._suppress = False
        self.value_changed.emit(clamped)

    def nudge(self, delta: float) -> None:
        self.setValue(self.value() + delta)


class PoseControlPanel(QtWidgets.QWidget):
    """10-DOF pose editor panel."""

    pose_changed = QtCore.pyqtSignal(object)  # emits Pose
    auto_center_requested = QtCore.pyqtSignal()
    save_pose_requested = QtCore.pyqtSignal()
    load_pose_requested = QtCore.pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        logger.debug("PoseControlPanel.__init__")

        # Let the pose dock be freely resized — the panel should shrink
        # with its container, not enforce a minimum height that blocks
        # the user from dragging the dock border smaller.
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Policy.Preferred,
            QtWidgets.QSizePolicy.Policy.Minimum,
        )
        self.setMinimumHeight(0)

        self._emit_suppressed = False

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        root.addWidget(self._build_translation_group())
        root.addWidget(self._build_rotation_group())
        root.addWidget(self._build_scale_group())
        root.addWidget(self._build_flip_group())
        root.addLayout(self._build_button_row())
        root.addStretch(1)

        self._wire_up_signals()

    # --------------------------------------------------------- group builders

    def _build_translation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Translation (voxels)")
        layout = QtWidgets.QVBoxLayout(group)
        self.tx = _SliderSpinRow("tx", *_TRANS_RANGE, initial=IDENTITY_POSE.tx)
        self.ty = _SliderSpinRow("ty", *_TRANS_RANGE, initial=IDENTITY_POSE.ty)
        self.tz = _SliderSpinRow("tz", *_TRANS_RANGE, initial=IDENTITY_POSE.tz)
        for row in (self.tx, self.ty, self.tz):
            layout.addWidget(row)
        return group

    def _build_rotation_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Rotation (degrees, intrinsic ZYX)")
        layout = QtWidgets.QVBoxLayout(group)
        self.rx = _SliderSpinRow(
            "rx", *_ROT_RANGE, initial=IDENTITY_POSE.rx, step=0.5
        )
        self.ry = _SliderSpinRow(
            "ry", *_ROT_RANGE, initial=IDENTITY_POSE.ry, step=0.5
        )
        self.rz = _SliderSpinRow(
            "rz", *_ROT_RANGE, initial=IDENTITY_POSE.rz, step=0.5
        )
        for row in (self.rx, self.ry, self.rz):
            layout.addWidget(row)
        return group

    def _build_scale_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Scale")
        layout = QtWidgets.QVBoxLayout(group)

        # Link-checkbox: when checked (default), X and Y scale together via
        # the ``scale`` slider; ``sx`` / ``sy`` rows are disabled. Uncheck
        # to stretch one axis independently.
        self.link_scale = QtWidgets.QCheckBox("Link X/Y scale")
        self.link_scale.setChecked(True)
        layout.addWidget(self.link_scale)

        # Uniform "scale" slider (used when linked).
        self.scale = _SliderSpinRow(
            "scale", *_SCALE_RANGE, initial=IDENTITY_POSE.sx, step=0.01
        )
        layout.addWidget(self.scale)

        # Independent X/Y scale sliders (used when unlinked).
        self.sx = _SliderSpinRow(
            "sx", *_SCALE_RANGE, initial=IDENTITY_POSE.sx, step=0.01
        )
        self.sy = _SliderSpinRow(
            "sy", *_SCALE_RANGE, initial=IDENTITY_POSE.sy, step=0.01
        )
        layout.addWidget(self.sx)
        layout.addWidget(self.sy)

        # Hidden, always 1.0 — kept as a dummy so ``pose()`` still reads sz.
        self.sz = _SliderSpinRow(
            "sz", *_SCALE_RANGE, initial=1.0, step=0.01
        )
        self.sz.hide()

        self.link_scale.toggled.connect(self._on_scale_link_toggled)
        # When linked, keep sx/sy in sync with the linked scale slider.
        self.scale.value_changed.connect(self._on_scale_slider_changed)
        # When unlinked, the uniform-scale slider is disabled — but if the
        # user scrubs one of sx/sy we want to mirror the change into the
        # linked slider so re-enabling "Link" uses a sensible value.
        self.sx.value_changed.connect(self._on_sx_sy_changed)
        self.sy.value_changed.connect(self._on_sx_sy_changed)

        # Initial enabled state without emitting (flip checkboxes haven't
        # been built yet by the time this builder runs).
        self.scale.setEnabled(True)
        self.sx.setEnabled(False)
        self.sy.setEnabled(False)
        return group

    def _on_scale_link_toggled(self, linked: bool) -> None:
        self.scale.setEnabled(linked)
        self.sx.setEnabled(not linked)
        self.sy.setEnabled(not linked)
        if linked:
            # Collapse sx and sy back to the uniform value when re-linking.
            self._emit_suppressed = True
            try:
                v = self.scale.value()
                self.sx.setValue(v)
                self.sy.setValue(v)
            finally:
                self._emit_suppressed = False
            # Only emit pose_changed if the rest of the panel is already
            # constructed (flip checkboxes etc.); otherwise the caller
            # hasn't wired up the panel yet.
            if hasattr(self, "flip_x"):
                self._emit_pose_changed()

    def _on_scale_slider_changed(self, value: float) -> None:
        if not self.link_scale.isChecked():
            return
        # Mirror the uniform value into sx/sy with signal suppression so
        # each one doesn't emit its own pose_changed — then emit once at
        # the end so the projection worker sees the new scale.
        self._emit_suppressed = True
        try:
            self.sx.setValue(value)
            self.sy.setValue(value)
        finally:
            self._emit_suppressed = False
        if hasattr(self, "flip_x"):  # guard: during panel construction
            self._emit_pose_changed()

    def _on_sx_sy_changed(self, _value: float) -> None:
        if self.link_scale.isChecked():
            return
        # Keep the uniform slider reflecting the geometric mean so that
        # re-linking "averages" the two axes rather than snapping.
        geo = (self.sx.value() * self.sy.value()) ** 0.5
        self._emit_suppressed = True
        try:
            self.scale.setValue(geo)
        finally:
            self._emit_suppressed = False

    def _build_flip_group(self) -> QtWidgets.QGroupBox:
        group = QtWidgets.QGroupBox("Flip")
        layout = QtWidgets.QHBoxLayout(group)
        self.flip_x = QtWidgets.QCheckBox("flip X")
        self.flip_y = QtWidgets.QCheckBox("flip Y")
        self.flip_z = QtWidgets.QCheckBox("flip Z")
        for cb in (self.flip_x, self.flip_y, self.flip_z):
            layout.addWidget(cb)
        return group

    def _build_button_row(self) -> QtWidgets.QHBoxLayout:
        layout = QtWidgets.QHBoxLayout()
        self.reset_btn = QtWidgets.QPushButton("Reset pose")
        self.auto_center_btn = QtWidgets.QPushButton("Auto-centre")
        self.save_btn = QtWidgets.QPushButton("Save pose…")
        self.load_btn = QtWidgets.QPushButton("Load pose…")
        for btn in (
            self.reset_btn,
            self.auto_center_btn,
            self.save_btn,
            self.load_btn,
        ):
            layout.addWidget(btn)
        return layout

    # ------------------------------------------------------------------ wiring

    def _wire_up_signals(self) -> None:
        # Translation + rotation rows all emit pose_changed directly. Scale
        # rows are wired separately inside ``_build_scale_group`` because
        # they need link-checkbox-aware behaviour. ``sx`` / ``sy`` are the
        # source of truth for scale so ``pose()`` reads them directly.
        for row in (self.tx, self.ty, self.tz, self.rx, self.ry, self.rz):
            row.value_changed.connect(self._emit_pose_changed)
        self.sx.value_changed.connect(self._emit_pose_changed)
        self.sy.value_changed.connect(self._emit_pose_changed)
        for cb in (self.flip_x, self.flip_y, self.flip_z):
            cb.stateChanged.connect(self._emit_pose_changed)

        self.reset_btn.clicked.connect(self.reset_pose)
        self.auto_center_btn.clicked.connect(self.auto_center_requested)
        self.save_btn.clicked.connect(self.save_pose_requested)
        self.load_btn.clicked.connect(self.load_pose_requested)

    # --------------------------------------------------------------- public

    def pose(self) -> Pose:
        # Read from the sx/sy sliders directly — they are the source of
        # truth whether linked or not (the linked "scale" slider simply
        # feeds them both).
        return Pose(
            tx=self.tx.value(),
            ty=self.ty.value(),
            tz=self.tz.value(),
            rx=self.rx.value(),
            ry=self.ry.value(),
            rz=self.rz.value(),
            sx=self.sx.value(),
            sy=self.sy.value(),
            sz=1.0,
            flip_x=self.flip_x.isChecked(),
            flip_y=self.flip_y.isChecked(),
            flip_z=self.flip_z.isChecked(),
        )

    def set_pose(self, pose: Pose, emit: bool = True) -> None:
        self._emit_suppressed = True
        try:
            self.tx.setValue(pose.tx)
            self.ty.setValue(pose.ty)
            self.tz.setValue(pose.tz)
            self.rx.setValue(pose.rx)
            self.ry.setValue(pose.ry)
            self.rz.setValue(pose.rz)
            self.sx.setValue(pose.sx)
            self.sy.setValue(pose.sy)
            # Auto-engage "Link X/Y" when the two axes match — feels
            # natural for poses loaded from disk.
            linked = abs(pose.sx - pose.sy) < 1e-4
            self.link_scale.setChecked(linked)
            if linked:
                self.scale.setValue(pose.sx)
            else:
                # Reflect the geometric mean so re-linking is sensible.
                self.scale.setValue((pose.sx * pose.sy) ** 0.5)
            self.flip_x.setChecked(pose.flip_x)
            self.flip_y.setChecked(pose.flip_y)
            self.flip_z.setChecked(pose.flip_z)
        finally:
            self._emit_suppressed = False
        if emit:
            self._emit_pose_changed()

    def reset_pose(self) -> None:
        self.set_pose(Pose(), emit=True)

    # --------------------------------------------------------------- internal

    def _emit_pose_changed(self, *_args) -> None:
        if self._emit_suppressed:
            return
        pose = self.pose()
        logger.debug("PoseControlPanel emit: %s", pose)
        self.pose_changed.emit(pose)
