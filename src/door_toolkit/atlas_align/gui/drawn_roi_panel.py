"""
Drawn-ROI dock
==============

Lists user-drawn polygons created by :meth:`OverlayView.start_drawing`
and lets the user:

* click into the list to update the Response dock with that ROI's
  activity-map response + DoOR comparison,
* tag each polygon with an antennal-lobe side (L / R) — the display
  label then reads ``L1, L2, …`` / ``R1, R2, …``,
* assign a glomerulus manually via a per-row dropdown,
* load previously-exported drawn ROIs from CSV (polygons + side +
  manual pick + saved responses), and
* export the drawn ROIs + manual assignments to a CSV.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from door_toolkit.atlas_align.config import get_logger

logger = get_logger(__name__)

_MANUAL_UNSET = "—"
_SIDE_UNSET = "—"
_SIDES = (_SIDE_UNSET, "L", "R")


@dataclass
class DrawnROI:
    """One user-drawn polygon + optional computed response + assignment."""

    index: int
    xs: np.ndarray
    ys: np.ndarray
    response: Optional[np.ndarray] = None  # (n_odors,) float32
    manual_glomerulus: Optional[str] = None
    side: Optional[str] = None             # "L" / "R" / None
    display_label: str = ""                # e.g. "L1", "R2", or "1" when side unset

    @property
    def centroid(self) -> Tuple[float, float]:
        return float(self.xs.mean()), float(self.ys.mean())


# ---------------------------------------------------------------------------
# CSV import / export helpers — module-level so both the panel and the main
# window can call them.
# ---------------------------------------------------------------------------


def load_drawn_rois_csv(
    path: Path,
) -> Tuple[List[DrawnROI], List[str]]:
    """Parse a previously-exported drawn-ROIs CSV.

    Returns:
        A pair ``(rois, odor_columns)`` where ``odor_columns`` is the list
        of odors whose per-odor ``response_<odor>`` values were stored (in
        file order). Each :class:`DrawnROI` has its ``index`` field left at
        ``-1`` so the caller can assign a fresh persistent index.

    Accepts both the old (pre-side) and the new (with side) column layouts.
    """
    path = Path(path).expanduser().resolve()
    rois: List[DrawnROI] = []
    odor_cols: List[str] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        field_names = reader.fieldnames or []
        odor_cols = [
            fn[len("response_"):] for fn in field_names
            if fn.startswith("response_")
        ]
        for row in reader:
            xs_raw = (row.get("vertices_x") or "").strip()
            ys_raw = (row.get("vertices_y") or "").strip()
            if not xs_raw or not ys_raw:
                continue
            try:
                xs = np.asarray(
                    [float(v) for v in xs_raw.split(";") if v != ""],
                    dtype=np.float32,
                )
                ys = np.asarray(
                    [float(v) for v in ys_raw.split(";") if v != ""],
                    dtype=np.float32,
                )
            except ValueError:
                logger.warning("Skipping malformed row in %s: %r", path, row)
                continue
            if xs.size < 3 or ys.size < 3 or xs.size != ys.size:
                continue

            side_raw = (row.get("side") or "").strip().upper()
            side = side_raw if side_raw in ("L", "R") else None

            manual_raw = (row.get("manual_assigned_glomerulus") or "").strip()
            manual = manual_raw if manual_raw else None

            # Reconstruct response vector in odor_cols order if any are present.
            response: Optional[np.ndarray] = None
            if odor_cols:
                vals: List[float] = []
                for o in odor_cols:
                    v_raw = (row.get(f"response_{o}") or "").strip()
                    if v_raw == "":
                        vals.append(float("nan"))
                    else:
                        try:
                            vals.append(float(v_raw))
                        except ValueError:
                            vals.append(float("nan"))
                if any(np.isfinite(v) for v in vals):
                    response = np.asarray(vals, dtype=np.float32)

            rois.append(DrawnROI(
                index=-1,
                xs=xs, ys=ys,
                response=response,
                manual_glomerulus=manual,
                side=side,
                display_label=(row.get("display_label") or "").strip(),
            ))
    logger.info("Loaded %d drawn ROIs from %s", len(rois), path)
    return rois, odor_cols


class DrawnROIPanel(QtWidgets.QWidget):
    """Dock content for managing user-drawn ROIs."""

    draw_requested = QtCore.pyqtSignal()           # user wants to start drawing
    delete_requested = QtCore.pyqtSignal(int)      # drawn_index
    roi_clicked = QtCore.pyqtSignal(int)           # drawn_index (for Response panel)
    manual_assignment_changed = QtCore.pyqtSignal(int, str)  # idx, glomerulus
    side_changed = QtCore.pyqtSignal(int, str)     # idx, side ("L"/"R"/"")
    export_requested = QtCore.pyqtSignal()
    load_requested = QtCore.pyqtSignal()           # main window opens file dialog

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        hint = QtWidgets.QLabel(
            "Start drawing, then left-click to add polygon vertices. "
            "Enter closes and saves. ESC cancels. Set the Side column "
            "(L/R) and the label above each polygon updates to L1, R2, …"
        )
        hint.setStyleSheet("color: #aaa;")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        btn_row = QtWidgets.QHBoxLayout()
        self._start_btn = QtWidgets.QPushButton("Start drawing")
        self._start_btn.clicked.connect(self.draw_requested)
        self._delete_btn = QtWidgets.QPushButton("Delete selected")
        self._delete_btn.clicked.connect(self._on_delete_clicked)
        self._load_btn = QtWidgets.QPushButton("Load drawn ROIs…")
        self._load_btn.clicked.connect(self.load_requested)
        self._export_btn = QtWidgets.QPushButton("Export drawn ROIs…")
        self._export_btn.clicked.connect(self.export_requested)
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._delete_btn)
        btn_row.addWidget(self._load_btn)
        btn_row.addWidget(self._export_btn)
        layout.addLayout(btn_row)

        # Columns: Label | Side | Centroid | Top-1 DoOR | Lock | Manual
        self._table = QtWidgets.QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(
            ["Label", "Side", "Centroid (x, y)", "Top-1 DoOR match",
             "Lock", "Manual"]
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.itemSelectionChanged.connect(self._on_selection)
        layout.addWidget(self._table, stretch=1)

        self._rois: List[DrawnROI] = []
        self._palette: List[str] = []
        self._top_match_for: Dict[int, str] = {}
        # drawn_index -> True if locked; the Manual combobox on that row
        # is disabled and programmatic setters refuse to overwrite it.
        self._locked_rois: Dict[int, bool] = {}
        # drawn_index -> the live Manual QComboBox, so we can enable /
        # disable it from _on_lock_toggled without re-running setCellWidget.
        self._manual_combos: Dict[int, QtWidgets.QComboBox] = {}

    # ----------------------------------------------------- palette / data

    def set_glomerulus_palette(self, names: List[str]) -> None:
        self._palette = sorted(set(names))

    def add_roi(self, roi: DrawnROI) -> None:
        self._rois.append(roi)
        self._append_table_row(roi)

    def remove_roi(self, index: int) -> None:
        for i, r in enumerate(self._rois):
            if r.index == index:
                self._rois.pop(i)
                self._table.removeRow(i)
                break
        self._locked_rois.pop(index, None)
        self._manual_combos.pop(index, None)

    def rois(self) -> List[DrawnROI]:
        return list(self._rois)

    def set_top_match(self, drawn_index: int, glomerulus: str) -> None:
        """Update the "Top-1 DoOR match" cell and stash for export."""
        self._top_match_for[drawn_index] = glomerulus
        for row in range(self._table.rowCount()):
            if self._roi_at_row(row).index == drawn_index:
                self._table.item(row, 3).setText(glomerulus or "—")
                return

    def update_manual(self, drawn_index: int, glomerulus: str) -> None:
        for roi in self._rois:
            if roi.index == drawn_index:
                roi.manual_glomerulus = glomerulus or None
                break

    def set_display_label(self, drawn_index: int, label: str) -> None:
        """Update the "Label" cell for ``drawn_index`` in-place."""
        for i, roi in enumerate(self._rois):
            if roi.index == drawn_index:
                roi.display_label = label
                item = self._table.item(i, 0)
                if item is not None:
                    item.setText(label)
                return

    # ------------------------------------------------------------ helpers

    def _append_table_row(self, roi: DrawnROI) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)

        label_item = QtWidgets.QTableWidgetItem(roi.display_label or str(roi.index + 1))
        label_item.setData(QtCore.Qt.ItemDataRole.UserRole, roi.index)
        self._table.setItem(row, 0, label_item)

        side_combo = QtWidgets.QComboBox()
        side_combo.addItems(list(_SIDES))
        if roi.side in ("L", "R"):
            side_combo.setCurrentText(roi.side)
        else:
            side_combo.setCurrentText(_SIDE_UNSET)
        side_combo.currentTextChanged.connect(
            lambda text, di=roi.index: self._on_side_changed(di, text)
        )
        self._table.setCellWidget(row, 1, side_combo)

        cx, cy = roi.centroid
        self._table.setItem(
            row, 2, QtWidgets.QTableWidgetItem(f"({cx:.0f}, {cy:.0f})")
        )
        self._table.setItem(row, 3, QtWidgets.QTableWidgetItem("—"))

        # Lock checkbox — when checked, the Manual combobox is disabled
        # and programmatic setters refuse to overwrite the pick.
        lock_cb = QtWidgets.QCheckBox()
        lock_cb.setToolTip(
            "Lock this ROI's manual glomerulus. Uncheck to allow changes."
        )
        lock_cb.setChecked(bool(self._locked_rois.get(roi.index, False)))
        lock_cb.stateChanged.connect(
            lambda state, di=roi.index: self._on_lock_toggled(
                di, state == QtCore.Qt.CheckState.Checked.value
            )
        )
        lock_holder = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(lock_holder)
        h.addWidget(lock_cb)
        h.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        h.setContentsMargins(0, 0, 0, 0)
        self._table.setCellWidget(row, 4, lock_holder)

        manual_combo = QtWidgets.QComboBox()
        manual_combo.addItem(_MANUAL_UNSET)
        manual_combo.addItems(self._palette)
        if roi.manual_glomerulus:
            manual_combo.setCurrentText(roi.manual_glomerulus)
        manual_combo.currentTextChanged.connect(
            lambda text, di=roi.index: self._on_manual_changed(di, text)
        )
        manual_combo.setEnabled(not self._locked_rois.get(roi.index, False))
        self._table.setCellWidget(row, 5, manual_combo)
        self._manual_combos[roi.index] = manual_combo

    def _roi_at_row(self, row: int) -> DrawnROI:
        return self._rois[row]

    def _on_selection(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        drawn_index = self._rois[rows[0].row()].index
        self.roi_clicked.emit(drawn_index)

    def _on_manual_changed(self, drawn_index: int, text: str) -> None:
        # Belt-and-suspenders: the combobox is disabled when locked, but
        # guard emission too in case a refresh races a signal.
        if self._locked_rois.get(drawn_index, False):
            return
        picked = "" if text == _MANUAL_UNSET else text
        self.update_manual(drawn_index, picked)
        self.manual_assignment_changed.emit(drawn_index, picked)

    # ----------------------------------------------------- lock API

    def _on_lock_toggled(self, drawn_index: int, locked: bool) -> None:
        self._locked_rois[drawn_index] = bool(locked)
        combo = self._manual_combos.get(drawn_index)
        if combo is not None:
            combo.setEnabled(not locked)

    def is_locked(self, drawn_index: int) -> bool:
        """True if the Manual column for this drawn ROI is locked."""
        return bool(self._locked_rois.get(drawn_index, False))

    def try_set_manual(self, drawn_index: int, glomerulus: str) -> bool:
        """Programmatic setter that respects the lock.

        Returns ``True`` if the pick was applied, ``False`` when the row
        is locked and the caller should surface a "locked" status message.
        """
        if self._locked_rois.get(drawn_index, False):
            return False
        combo = self._manual_combos.get(drawn_index)
        if combo is not None:
            combo.setCurrentText(glomerulus if glomerulus else _MANUAL_UNSET)
        else:
            self.update_manual(drawn_index, glomerulus)
            self.manual_assignment_changed.emit(drawn_index, glomerulus or "")
        return True

    def _on_side_changed(self, drawn_index: int, text: str) -> None:
        picked = "" if text == _SIDE_UNSET else text
        for roi in self._rois:
            if roi.index == drawn_index:
                roi.side = picked if picked in ("L", "R") else None
                break
        self.side_changed.emit(drawn_index, picked)

    def _on_delete_clicked(self) -> None:
        rows = self._table.selectionModel().selectedRows()
        if not rows:
            return
        drawn_index = self._rois[rows[0].row()].index
        self.delete_requested.emit(drawn_index)

    def select_by_drawn_index(self, drawn_index: int) -> None:
        """Highlight the row matching ``drawn_index`` without re-emitting."""
        for i, roi in enumerate(self._rois):
            if roi.index == drawn_index:
                self._table.blockSignals(True)
                self._table.selectRow(i)
                self._table.blockSignals(False)
                return

    # ------------------------------------------------------------ export

    def write_csv(self, path: Path, odor_order: List[str]) -> int:
        """Write the drawn ROIs to a CSV keyed by drawn-roi index.

        Columns:
            drawn_index, display_label, side,
            centroid_x, centroid_y,
            manual_assigned_glomerulus, top_door_match,
            vertices_x, vertices_y,
            <one column per odor: response_{odor}>

        Returns:
            Row count written.
        """
        cols = [
            "drawn_index",
            "display_label",
            "side",
            "centroid_x",
            "centroid_y",
            "manual_assigned_glomerulus",
            "top_door_match",
            "vertices_x",
            "vertices_y",
        ] + [f"response_{o}" for o in odor_order]
        with Path(path).open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=cols)
            writer.writeheader()
            for roi in self._rois:
                cx, cy = roi.centroid
                row: Dict[str, object] = {
                    "drawn_index": roi.index,
                    "display_label": roi.display_label,
                    "side": roi.side or "",
                    "centroid_x": f"{cx:.2f}",
                    "centroid_y": f"{cy:.2f}",
                    "manual_assigned_glomerulus": roi.manual_glomerulus or "",
                    "top_door_match": self._top_match_for.get(roi.index, ""),
                    "vertices_x": ";".join(f"{v:.1f}" for v in roi.xs),
                    "vertices_y": ";".join(f"{v:.1f}" for v in roi.ys),
                }
                if roi.response is not None:
                    for i, o in enumerate(odor_order):
                        if i < len(roi.response):
                            row[f"response_{o}"] = f"{float(roi.response[i]):.6f}"
                        else:
                            row[f"response_{o}"] = ""
                else:
                    for o in odor_order:
                        row[f"response_{o}"] = ""
                writer.writerow(row)
        logger.info("Wrote drawn-ROIs CSV: %s (%d rows)", path, len(self._rois))
        return len(self._rois)
