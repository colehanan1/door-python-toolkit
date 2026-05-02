"""
IoU summary panel
=================

:class:`IoUPanel` shows the running IoU assignment: a sortable table of
``roi_name | assigned_glomerulus | iou | alternates…``, a live mean-IoU
label, a threshold spinbox, and an **Assign & Export** button.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from PyQt6 import QtCore, QtGui, QtWidgets

from door_toolkit.atlas_align.config import DEFAULT_IOU_THRESHOLD, get_logger
from door_toolkit.atlas_align.core.iou_matcher import AssignmentResult

logger = get_logger(__name__)

_COLUMNS = [
    "ROI",
    "IoU top",
    "IoU",
    "Lock",
    "Manual",
    "Alt 1",
    "Alt 1 IoU",
    "Alt 2",
    "Alt 2 IoU",
    "Alt 3",
    "Alt 3 IoU",
]

#: Sentinel text used when the user has not yet picked a manual assignment.
_MANUAL_UNSET = "—"


class IoUPanel(QtWidgets.QWidget):
    """Right-dock panel showing per-ROI assignments and summary stats.

    In addition to the IoU-based automatic matching, the "Manual" column
    lets the user override each ROI's glomerulus assignment. The current
    manual overrides are exported alongside the IoU assignments when the
    user clicks *Assign & Export*.
    """

    threshold_changed = QtCore.pyqtSignal(float)
    export_requested = QtCore.pyqtSignal()
    row_selected = QtCore.pyqtSignal(int)  # roi_index
    # Emitted whenever a Manual-column dropdown changes. ``""`` means the
    # user unset the assignment.
    manual_assignment_changed = QtCore.pyqtSignal(int, str)  # roi_index, glom_name

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        logger.debug("IoUPanel.__init__")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # --- summary row ----------------------------------------------------
        summary_row = QtWidgets.QHBoxLayout()
        self._mean_label = QtWidgets.QLabel("mean IoU: —")
        font = QtGui.QFont()
        font.setBold(True)
        self._mean_label.setFont(font)
        summary_row.addWidget(self._mean_label)

        summary_row.addStretch(1)

        summary_row.addWidget(QtWidgets.QLabel("threshold"))
        self._threshold_spin = QtWidgets.QDoubleSpinBox()
        self._threshold_spin.setRange(0.0, 1.0)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setDecimals(2)
        self._threshold_spin.setValue(DEFAULT_IOU_THRESHOLD)
        summary_row.addWidget(self._threshold_spin)

        layout.addLayout(summary_row)

        self._counts_label = QtWidgets.QLabel("0 above / 0 below")
        layout.addWidget(self._counts_label)

        # --- table ----------------------------------------------------------
        self._table = QtWidgets.QTableWidget(0, len(_COLUMNS), self)
        self._table.setHorizontalHeaderLabels(_COLUMNS)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setSortingEnabled(True)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.ResizeToContents
        )
        layout.addWidget(self._table, stretch=1)

        # --- export button --------------------------------------------------
        self._export_btn = QtWidgets.QPushButton("Assign && Export")
        layout.addWidget(self._export_btn)

        # --- signals --------------------------------------------------------
        self._threshold_spin.valueChanged.connect(self.threshold_changed)
        self._export_btn.clicked.connect(self.export_requested)
        self._table.itemSelectionChanged.connect(self._on_row_selected)

        # Full list of all glomerulus names (across every view), populated
        # once by :meth:`set_glomerulus_palette`. Used for the Manual-column
        # dropdown.
        self._all_glomerulus_names: List[str] = []
        # roi_index -> manually-chosen glomerulus name (or "" if unset)
        self._manual_assignments: Dict[int, str] = {}
        # roi_index -> True if the user locked this row's manual pick
        # (the Manual combobox is disabled while locked).
        self._locked_rois: Dict[int, bool] = {}
        # roi_index -> the live QComboBox for that row, so programmatic
        # callers (e.g. "Apply DoOR suggestion") can respect the lock.
        self._manual_combos: Dict[int, QtWidgets.QComboBox] = {}

    # --------------------------------------------------------------- public

    def threshold(self) -> float:
        return self._threshold_spin.value()

    def set_threshold(self, value: float) -> None:
        self._threshold_spin.blockSignals(True)
        self._threshold_spin.setValue(float(value))
        self._threshold_spin.blockSignals(False)

    # ---------------------------------------------------------- palette

    def set_glomerulus_palette(self, names: List[str]) -> None:
        """Provide the full list of possible glomerulus names for the Manual
        dropdown. Call once at startup from the GUI with
        ``bundle.labels.values()``."""
        self._all_glomerulus_names = sorted(set(names))

    def manual_assignments(self) -> Dict[int, str]:
        """Snapshot of ``{roi_index → manually-chosen-glomerulus-name}``.

        Entries are present only for ROIs the user has explicitly picked
        a glomerulus for.
        """
        return {
            idx: name
            for idx, name in self._manual_assignments.items()
            if name
        }

    def set_manual_assignments(self, mapping: Dict[int, str]) -> None:
        self._manual_assignments = dict(mapping)

    # ---------------------------------------------------------- table

    def update_result(self, result: AssignmentResult) -> None:
        """Refresh the table and summary from a new assignment result."""
        self._mean_label.setText(f"mean IoU: {result.mean_iou:.3f}")
        self._counts_label.setText(
            f"{result.n_above_threshold} above / {result.n_below_threshold} below"
        )

        # Disable sorting while we swap rows; otherwise setCellWidget and
        # sorting cross paths and rows jumble.
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(result.assignments))
        lock_col = _COLUMNS.index("Lock")
        manual_col = _COLUMNS.index("Manual")
        self._manual_combos.clear()
        for row, a in enumerate(result.assignments):
            alt = (a.alternates + [(0, "", 0.0)] * 3)[:3]
            top_iou_name = (
                a.glomerulus_name if a.above_threshold else "— (below thr)"
            )
            # Numeric IoU columns (for UserRole sort values) live at the
            # fixed indices 2 (top), 6 (alt1), 8 (alt2), 10 (alt3) once
            # the Lock column is inserted. Precompute once per row.
            col_values = {
                0: a.roi_name,
                1: top_iou_name,
                2: f"{a.iou:.3f}",
                5: alt[0][1],
                6: f"{alt[0][2]:.3f}" if alt[0][1] else "",
                7: alt[1][1],
                8: f"{alt[1][2]:.3f}" if alt[1][1] else "",
                9: alt[2][1],
                10: f"{alt[2][2]:.3f}" if alt[2][1] else "",
            }
            numeric_cols = {2, 6, 8, 10}
            for col, val in col_values.items():
                item = QtWidgets.QTableWidgetItem(str(val))
                if col in numeric_cols:
                    try:
                        item.setData(
                            QtCore.Qt.ItemDataRole.UserRole, float(val)
                        )
                    except (TypeError, ValueError):
                        pass
                item.setData(QtCore.Qt.ItemDataRole.UserRole + 1, a.roi_index)
                if not a.above_threshold:
                    item.setForeground(QtGui.QBrush(QtGui.QColor(180, 180, 180)))
                self._table.setItem(row, col, item)

            # Lock checkbox: when checked, the Manual combobox is
            # disabled so the assignment can't be changed (by the user
            # clicking, by the DoOR-suggestion shortcut, or by any other
            # programmatic caller that routes through set_manual_locked).
            lock_cb = QtWidgets.QCheckBox()
            lock_cb.setToolTip(
                "Lock this row's manual assignment. Uncheck to allow changes."
            )
            lock_cb.setChecked(bool(self._locked_rois.get(a.roi_index, False)))
            lock_cb.stateChanged.connect(
                lambda state, rid=a.roi_index: self._on_lock_toggled(
                    rid, state == QtCore.Qt.CheckState.Checked.value
                )
            )
            # Centre the checkbox inside the cell.
            lock_holder = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(lock_holder)
            h.addWidget(lock_cb)
            h.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            h.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(row, lock_col, lock_holder)

            # Manual dropdown (editable per-row unless locked).
            combo = QtWidgets.QComboBox()
            combo.addItem(_MANUAL_UNSET)
            combo.addItems(self._all_glomerulus_names)
            existing = self._manual_assignments.get(a.roi_index, "")
            if existing and existing in self._all_glomerulus_names:
                combo.setCurrentText(existing)
            else:
                combo.setCurrentText(_MANUAL_UNSET)
            combo.currentTextChanged.connect(
                lambda text, rid=a.roi_index: self._on_manual_changed(rid, text)
            )
            combo.setEnabled(not self._locked_rois.get(a.roi_index, False))
            self._table.setCellWidget(row, manual_col, combo)
            self._manual_combos[a.roi_index] = combo

        # Leave sorting off — the Manual column's combobox widgets don't
        # sort cleanly; ROIs remain in their source-file order which tends
        # to be what users expect anyway.
        self._table.setSortingEnabled(False)

    def _on_manual_changed(self, roi_index: int, text: str) -> None:
        # Belt-and-suspenders: the combobox is disabled when locked, but
        # guard emission too in case a refresh races a signal.
        if self._locked_rois.get(roi_index, False):
            return
        if text == _MANUAL_UNSET:
            self._manual_assignments.pop(roi_index, None)
            self.manual_assignment_changed.emit(roi_index, "")
        else:
            self._manual_assignments[roi_index] = text
            self.manual_assignment_changed.emit(roi_index, text)

    # ----------------------------------------------------- lock API

    def _on_lock_toggled(self, roi_index: int, locked: bool) -> None:
        self._locked_rois[roi_index] = bool(locked)
        combo = self._manual_combos.get(roi_index)
        if combo is not None:
            combo.setEnabled(not locked)

    def is_locked(self, roi_index: int) -> bool:
        """True if this ROI's manual pick is locked against changes."""
        return bool(self._locked_rois.get(roi_index, False))

    def locked_assignments(self) -> Dict[int, bool]:
        """Snapshot of ``{roi_index → True}`` for every locked row."""
        return {rid: True for rid, v in self._locked_rois.items() if v}

    def try_set_manual(self, roi_index: int, glomerulus: str) -> bool:
        """Programmatic setter that refuses to change a locked row.

        Returns ``True`` if the change was applied, ``False`` if the row
        was locked and the caller should show a "locked" message.
        """
        if self._locked_rois.get(roi_index, False):
            return False
        combo = self._manual_combos.get(roi_index)
        if combo is not None:
            combo.setCurrentText(
                glomerulus if glomerulus else _MANUAL_UNSET
            )
        else:
            # No live combo (table not yet rendered) — store directly.
            if glomerulus:
                self._manual_assignments[roi_index] = glomerulus
            else:
                self._manual_assignments.pop(roi_index, None)
            self.manual_assignment_changed.emit(roi_index, glomerulus or "")
        return True

    # --------------------------------------------------------------- private

    def _on_row_selected(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        row = rows[0].row()
        item = self._table.item(row, 0)
        if item is None:
            return
        roi_index = item.data(QtCore.Qt.ItemDataRole.UserRole + 1)
        if roi_index is None:
            return
        try:
            self.row_selected.emit(int(roi_index))
        except (TypeError, ValueError):
            pass
