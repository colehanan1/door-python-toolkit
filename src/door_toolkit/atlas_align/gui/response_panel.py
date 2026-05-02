"""
Response dock
=============

Shows per-odor max / min ΔF/F for the currently selected ROI, alongside
the DoOR literature response (and similarity score) for the manually
assigned glomerulus, plus top-3 best-matching glomeruli suggested by
cosine similarity between the user's peak-ΔF/F vector and every DoOR
response vector.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from PyQt6 import QtCore, QtGui, QtWidgets

from door_toolkit.atlas_align.config import get_logger

logger = get_logger(__name__)


def _score_color(sim: float) -> QtGui.QColor:
    """Map a Pearson correlation score (-1..+1) to a traffic-light colour.

    Thresholds are chosen so a score needs to clear +0.6 to register as
    green — that is, the two vectors' *patterns* (ups vs downs across
    odors) must agree pretty strongly, not just overlap in a generic
    positive-valued sense.
    """
    if sim >= 0.70:
        return QtGui.QColor(110, 240, 120)   # green — strong pattern agreement
    if sim >= 0.40:
        return QtGui.QColor(240, 220, 80)    # yellow — weak agreement
    if sim >= 0.10:
        return QtGui.QColor(240, 170, 80)    # orange — negligible
    return QtGui.QColor(240, 120, 120)        # red — unrelated / anti-correlated


class ResponsePanel(QtWidgets.QWidget):
    """Panel that displays ΔF/F stats + DoOR comparison for the active ROI."""

    #: Emitted when the user clicks a suggestion → main_window can flip the
    #: Manual column's combobox to that glomerulus.
    suggestion_clicked = QtCore.pyqtSignal(str)  # glomerulus_name

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent=parent)
        logger.debug("ResponsePanel.__init__")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        self._header = QtWidgets.QLabel("No ROI selected")
        font = QtGui.QFont()
        font.setBold(True)
        self._header.setFont(font)
        layout.addWidget(self._header)

        self._subheader = QtWidgets.QLabel("")
        self._subheader.setStyleSheet("color: #aaa;")
        layout.addWidget(self._subheader)

        # Main table: Odor / min / max / DoOR-scaled (row-norm) /
        #             DoOR-projected (monotonic fit) / DoOR / agree?
        self._table = QtWidgets.QTableWidget(0, 7)
        self._table.setHorizontalHeaderLabels(
            ["Odor", "min ΔF/F", "max ΔF/F",
             "yours (DoOR-scaled)", "yours (DoOR-projected)",
             "DoOR", "agree?"]
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.NoSelection
        )
        self._table.horizontalHeader().setSectionResizeMode(
            QtWidgets.QHeaderView.ResizeMode.Stretch
        )
        self._table.verticalHeader().setVisible(False)
        layout.addWidget(self._table, stretch=1)

        # Match summary.
        self._match_label = QtWidgets.QLabel("")
        self._match_label.setWordWrap(True)
        font_b = QtGui.QFont()
        font_b.setBold(True)
        self._match_label.setFont(font_b)
        layout.addWidget(self._match_label)

        # Top-3 suggestions.
        suggestions_box = QtWidgets.QGroupBox(
            "Top-3 DoOR matches for your ΔF/F pattern (Pearson r)"
        )
        sug_layout = QtWidgets.QVBoxLayout(suggestions_box)
        self._suggestions_list = QtWidgets.QListWidget()
        self._suggestions_list.itemClicked.connect(self._on_suggestion_clicked)
        sug_layout.addWidget(self._suggestions_list)
        self._suggestions_hint = QtWidgets.QLabel(
            "Click a suggestion to set it as the Manual pick for this ROI."
        )
        self._suggestions_hint.setStyleSheet("color: #aaa; font-style: italic;")
        sug_layout.addWidget(self._suggestions_hint)
        layout.addWidget(suggestions_box)

    # --------------------------------------------------------------- API

    def clear(self) -> None:
        self._header.setText("No ROI selected")
        self._subheader.setText("")
        self._table.setRowCount(0)
        self._match_label.setText("")
        self._suggestions_list.clear()

    def show_roi_summary(
        self,
        roi_name: str,
        roi_extra: str,
        summary: Dict[str, Tuple[float, float]],
        door_responses: Optional[Dict[str, float]] = None,
        manual_glomerulus: Optional[str] = None,
        manual_receptor: Optional[str] = None,
        manual_similarity: Optional[float] = None,
        manual_cosine: Optional[float] = None,
        suggestions: Optional[List[Tuple[str, float]]] = None,
        user_door_scaled: Optional[Dict[str, float]] = None,
        user_door_projected: Optional[Dict[str, float]] = None,
        projection_model: Optional[str] = None,
        projection_rms: Optional[float] = None,
    ) -> None:
        """Populate the whole panel for one ROI.

        Args:
            roi_name: ROI header text.
            roi_extra: Supporting info (centroid, matched dff column, …).
            summary: ``{odor → (min, max)}`` — the user's raw ΔF/F stats.
            door_responses: Optional ``{odor → DoOR response 0–1}`` for the
                manually-assigned glomerulus. ``NaN`` values mean DoOR has
                no data for that receptor/odor pair.
            manual_glomerulus: Which glomerulus the user picked (if any).
            manual_receptor: DoOR receptor backing that glomerulus.
            manual_similarity: Cosine similarity score for the manual pick.
            suggestions: Top-N (glomerulus, cosine) pairs for the ROI's
                ΔF/F pattern across all DoOR-mapped glomeruli.
            user_door_scaled: Optional ``{odor → DoOR-calibrated scalar}``
                for the current ROI. Comes from
                ``door_row_scale`` (``row / (max - min)``) so excitation
                and inhibition land on DoOR's per-receptor [-1, +1] scale
                and are directly comparable to ``door_responses``. When
                provided, the "agree?" column compares on THIS scale.
        """
        self._header.setText(f"ROI: {roi_name}")
        self._subheader.setText(roi_extra)

        def color_for_unit_scale(val: float) -> QtGui.QColor:
            """Traffic-light colour for a value on DoOR's [0, 1] scale."""
            if val >= 0.5:
                return QtGui.QColor(255, 90, 90)
            if val >= 0.2:
                return QtGui.QColor(255, 180, 60)
            if val >= 0.05:
                return QtGui.QColor(240, 230, 110)
            if val > -0.05:
                return QtGui.QColor(140, 140, 140)
            # Inhibition
            return QtGui.QColor(120, 170, 240)

        odors = list(summary.keys())
        self._table.setRowCount(len(odors))
        for row, odor in enumerate(odors):
            vmin, vmax = summary[odor]
            door_val = None
            if door_responses is not None:
                door_val = door_responses.get(odor)
            user_scaled = (
                user_door_scaled.get(odor)
                if user_door_scaled is not None else None
            )
            user_projected = (
                user_door_projected.get(odor)
                if user_door_projected is not None else None
            )

            self._table.setItem(row, 0, QtWidgets.QTableWidgetItem(odor))
            self._table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(f"{vmin:+.3f}")
            )
            max_item = QtWidgets.QTableWidgetItem(f"{vmax:+.3f}")
            if vmax >= 0.5:
                max_item.setForeground(QtGui.QBrush(QtGui.QColor(255, 90, 90)))
            elif vmax >= 0.2:
                max_item.setForeground(QtGui.QBrush(QtGui.QColor(255, 180, 60)))
            elif vmax >= 0.05:
                max_item.setForeground(QtGui.QBrush(QtGui.QColor(240, 230, 110)))
            else:
                max_item.setForeground(QtGui.QBrush(QtGui.QColor(140, 140, 140)))
            self._table.setItem(row, 2, max_item)

            # --- yours (DoOR-scaled) ----------------------------------
            if user_scaled is None or not (
                user_scaled == user_scaled  # NaN check
            ):
                scaled_item = QtWidgets.QTableWidgetItem("—")
                scaled_item.setForeground(
                    QtGui.QBrush(QtGui.QColor(140, 140, 140))
                )
            else:
                scaled_item = QtWidgets.QTableWidgetItem(f"{user_scaled:+.2f}")
                scaled_item.setForeground(
                    QtGui.QBrush(color_for_unit_scale(user_scaled))
                )
            self._table.setItem(row, 3, scaled_item)

            # --- yours (DoOR-projected) ------------------------------
            # Only populated when a manual glomerulus is picked AND at
            # least 5 overlapping odors exist to fit a monotonic model.
            if user_projected is None or not (
                user_projected == user_projected  # NaN check
            ):
                proj_item = QtWidgets.QTableWidgetItem("—")
                proj_item.setForeground(
                    QtGui.QBrush(QtGui.QColor(140, 140, 140))
                )
            else:
                proj_item = QtWidgets.QTableWidgetItem(f"{user_projected:+.2f}")
                proj_item.setForeground(
                    QtGui.QBrush(color_for_unit_scale(user_projected))
                )
            self._table.setItem(row, 4, proj_item)

            # --- DoOR consensus --------------------------------------
            if door_val is None:
                door_item = QtWidgets.QTableWidgetItem("—")
                agree_item = QtWidgets.QTableWidgetItem("")
            elif door_val != door_val:  # NaN
                door_item = QtWidgets.QTableWidgetItem("no data")
                door_item.setForeground(
                    QtGui.QBrush(QtGui.QColor(140, 140, 140))
                )
                agree_item = QtWidgets.QTableWidgetItem("")
            else:
                door_item = QtWidgets.QTableWidgetItem(f"{door_val:+.2f}")
                door_item.setForeground(
                    QtGui.QBrush(color_for_unit_scale(door_val))
                )

                # Agreement: prefer the DoOR-projected user value (on
                # DoOR's own magnitude scale) over the pattern-only
                # DoOR-scaled value. Fall back to raw vmax last.
                if user_projected is not None and (
                    user_projected == user_projected
                ):
                    cmp_val = user_projected
                elif user_scaled is not None:
                    cmp_val = user_scaled
                else:
                    cmp_val = vmax
                user_high = cmp_val >= 0.10
                door_high = door_val >= 0.10
                user_inh = cmp_val <= -0.10
                door_inh = door_val <= -0.10
                if user_inh and door_inh:
                    agree_item = QtWidgets.QTableWidgetItem("✓ (both inhib)")
                    agree_item.setForeground(
                        QtGui.QBrush(QtGui.QColor(110, 240, 120))
                    )
                elif user_high == door_high and user_inh == door_inh:
                    agree_item = QtWidgets.QTableWidgetItem("✓")
                    agree_item.setForeground(
                        QtGui.QBrush(QtGui.QColor(110, 240, 120))
                    )
                else:
                    if user_high and not door_high:
                        txt = "✗ (you: hi, DoOR: lo)"
                    elif door_high and not user_high:
                        txt = "✗ (you: lo, DoOR: hi)"
                    elif user_inh and not door_inh:
                        txt = "✗ (you: inhib, DoOR: +)"
                    elif door_inh and not user_inh:
                        txt = "✗ (you: +, DoOR: inhib)"
                    else:
                        txt = "✗"
                    agree_item = QtWidgets.QTableWidgetItem(txt)
                    agree_item.setForeground(
                        QtGui.QBrush(QtGui.QColor(240, 120, 120))
                    )
            self._table.setItem(row, 5, door_item)
            self._table.setItem(row, 6, agree_item)

        # Match summary label.
        if manual_glomerulus is None:
            self._match_label.setText(
                "No manual glomerulus picked. Pick one in the Assignments "
                "dock to compare with DoOR."
            )
            self._match_label.setStyleSheet("color: #aaa;")
        elif manual_similarity is None or manual_similarity != manual_similarity:
            receptor_note = (
                f" (receptor {manual_receptor} — not de-orphanised in "
                f"DoOR v2.0.0)"
                if manual_receptor else ""
            )
            self._match_label.setText(
                f"Manual: {manual_glomerulus}{receptor_note}"
                f"   —   no DoOR response data available."
            )
            self._match_label.setStyleSheet("color: #aaa;")
        else:
            tier = (
                "EXCELLENT" if manual_similarity >= 0.80 else
                "STRONG"    if manual_similarity >= 0.60 else
                "WEAK"      if manual_similarity >= 0.30 else
                "POOR"      if manual_similarity >= 0.00 else
                "ANTI-CORRELATED"
            )
            receptor = f" (receptor {manual_receptor})" if manual_receptor else ""
            cosine_note = (
                f"   ·   cosine = {manual_cosine:+.2f}"
                if manual_cosine is not None else ""
            )
            self._match_label.setText(
                f"Manual: {manual_glomerulus}{receptor}   "
                f"Pearson r = {manual_similarity:+.2f}  [{tier} MATCH]"
                + cosine_note
            )
            color = _score_color(manual_similarity)
            self._match_label.setStyleSheet(
                f"color: rgb({color.red()},{color.green()},{color.blue()}); "
                "font-weight: bold;"
            )

        # Suggestions list.
        self._suggestions_list.clear()
        if suggestions:
            for rank, (glom, sim) in enumerate(suggestions, start=1):
                marker = "  ★" if glom == manual_glomerulus else ""
                item = QtWidgets.QListWidgetItem(
                    f"{rank}. {glom}{marker}   r = {sim:+.3f}"
                )
                color = _score_color(sim)
                item.setForeground(QtGui.QBrush(color))
                item.setData(QtCore.Qt.ItemDataRole.UserRole, glom)
                self._suggestions_list.addItem(item)

    def show_no_dff_data(self, roi_name: str) -> None:
        self._header.setText(f"ROI: {roi_name}")
        self._subheader.setText(
            "No ΔF/F signal matched for this ROI "
            "(nearest CSV centroid is outside tolerance)."
        )
        self._table.setRowCount(0)
        self._match_label.setText("")
        self._suggestions_list.clear()

    # --------------------------------------------------------------- events

    def _on_suggestion_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        glom = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if glom:
            self.suggestion_clicked.emit(str(glom))
