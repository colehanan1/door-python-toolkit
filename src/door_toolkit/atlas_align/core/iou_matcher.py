"""
ROI ↔ glomerulus IoU matching
=============================

Given

* a set of ROI binary masks (``N_rois`` masks of shape ``(H, W)``) from the
  user's FIJI ROI archive, and
* the label projection of the currently-posed atlas
  (``(H, W)`` uint16 image where each non-zero value is a glomerulus id),

this module:

1. Computes the full ``N_rois × N_glomeruli`` intersection-over-union
   matrix vectorised as a single scan.
2. Resolves the best assignment with a greedy-stable algorithm that
   prevents two ROIs from claiming the same glomerulus:

   * Each ROI points at its argmax candidate.
   * If two or more ROIs point at the same glomerulus, the one with the
     highest IoU keeps it; the losers re-point to their next-best
     candidate.
   * Repeat until stable or ``MAX_ASSIGNMENT_ITERATIONS`` reached.
3. Exposes the top-3 candidates per ROI so the UI can show alternates.
4. Flags ROIs below a configurable IoU threshold as unassigned
   (``glomerulus_id=0``, name prefixed ``UNK_`` downstream).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from door_toolkit.atlas_align.config import (
    DEFAULT_IOU_THRESHOLD,
    MAX_ASSIGNMENT_ITERATIONS,
    get_logger,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Assignment:
    """Result of assigning a single ROI to a glomerulus.

    Attributes:
        roi_index: Index of the ROI in the input ROISet.
        roi_name: Original ROI name from FIJI.
        glomerulus_id: Assigned label id (0 = unassigned / below threshold).
        glomerulus_name: Human-readable label, or ``UNK_<name>`` if
            unassigned.
        iou: IoU of the assigned glomerulus (0.0 if unassigned).
        alternates: Top ``k`` next-best (glomerulus_id, name, iou) tuples
            excluding the assigned one. At most 3 entries.
        above_threshold: Whether the top-scoring IoU was ≥ threshold.
    """

    roi_index: int
    roi_name: str
    glomerulus_id: int
    glomerulus_name: str
    iou: float
    alternates: List[Tuple[int, str, float]] = field(default_factory=list)
    above_threshold: bool = False


@dataclass
class AssignmentResult:
    """Full output of :func:`assign_rois`."""

    assignments: List[Assignment]
    iou_matrix: np.ndarray  # (N_rois, N_glomeruli)
    glomerulus_ids: List[int]  # column order of iou_matrix
    mean_iou: float
    n_above_threshold: int
    n_below_threshold: int
    iterations: int
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Vectorised IoU
# ---------------------------------------------------------------------------


def iou_matrix(
    roi_masks: Sequence[np.ndarray],
    label_projection: np.ndarray,
    glomerulus_ids: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, List[int]]:
    """Compute the full ``(N_rois × N_glomeruli)`` IoU matrix.

    Args:
        roi_masks: Iterable of 2D boolean arrays, all with the same
            shape as ``label_projection``.
        label_projection: 2D integer array of glomerulus ids (``0`` =
            background). Typically comes from
            :func:`door_toolkit.atlas_align.core.projection.project_atlas`.
        glomerulus_ids: Optional explicit list of glomerulus ids in the
            column order to use. If ``None``, takes ``sorted(unique)``
            of the non-zero values in ``label_projection``.

    Returns:
        Tuple ``(iou, glomerulus_ids)`` where ``iou`` is an
        ``(N_rois, N_glomeruli)`` float32 array and ``glomerulus_ids``
        is the list of ids in column order.
    """
    label_projection = np.asarray(label_projection)
    if label_projection.ndim != 2:
        raise ValueError(
            f"label_projection must be 2D, got shape {label_projection.shape}"
        )

    if glomerulus_ids is None:
        present = np.unique(label_projection)
        glomerulus_ids = [int(g) for g in present if g != 0]
    else:
        glomerulus_ids = [int(g) for g in glomerulus_ids if g != 0]

    N_rois = len(roi_masks)
    N_g = len(glomerulus_ids)
    iou = np.zeros((N_rois, N_g), dtype=np.float32)

    if N_rois == 0 or N_g == 0:
        return iou, glomerulus_ids

    # Precompute per-glomerulus masks and their areas.
    glom_masks = {
        gid: (label_projection == gid) for gid in glomerulus_ids
    }
    glom_areas = {
        gid: int(m.sum()) for gid, m in glom_masks.items()
    }

    for i, roi_mask in enumerate(roi_masks):
        rm = np.asarray(roi_mask, dtype=bool)
        if rm.shape != label_projection.shape:
            raise ValueError(
                f"ROI {i} mask shape {rm.shape} != "
                f"label_projection shape {label_projection.shape}"
            )
        roi_area = int(rm.sum())
        if roi_area == 0:
            continue
        for j, gid in enumerate(glomerulus_ids):
            gm = glom_masks[gid]
            inter = int(np.logical_and(rm, gm).sum())
            if inter == 0:
                continue
            union = roi_area + glom_areas[gid] - inter
            iou[i, j] = inter / union if union > 0 else 0.0
    return iou, glomerulus_ids


# ---------------------------------------------------------------------------
# Greedy stable assignment
# ---------------------------------------------------------------------------


def _greedy_resolve(
    iou: np.ndarray,
    threshold: float,
    max_iter: int,
) -> Tuple[np.ndarray, int]:
    """Return ``best_col_per_row`` after greedy conflict resolution.

    Negative values indicate an unassigned ROI.

    Args:
        iou: (N_rois, N_glom) IoU matrix.
        threshold: Rows whose final best IoU is < threshold are marked -1.
        max_iter: Cap on the swap iteration count.

    Returns:
        ``(best_col, iterations)``: ``best_col[i]`` is the column index
        assigned to row ``i`` (``-1`` if none).
    """
    N, G = iou.shape
    if N == 0 or G == 0:
        return np.full(N, -1, dtype=np.int64), 0

    # Pre-sort each row's candidate columns by IoU descending. A row's
    # current pointer marches down this list whenever it loses a tie.
    sorted_cols = np.argsort(-iou, axis=1, kind="stable")  # (N, G)
    pointers = np.zeros(N, dtype=np.int64)

    iterations = 0
    for iterations in range(1, max_iter + 1):
        # Current choice per row.
        current = np.where(
            pointers < G,
            sorted_cols[np.arange(N), np.minimum(pointers, G - 1)],
            -1,
        )

        # Only rows whose best option is still above zero participate.
        active = current >= 0

        # Pick columns that multiple rows want; keep only the highest-IoU row.
        conflict_changed = False
        col_to_rows: Dict[int, List[int]] = {}
        for row_idx in np.where(active)[0]:
            col = int(current[row_idx])
            if iou[row_idx, col] <= 0:
                # No meaningful overlap left — mark as exhausted.
                pointers[row_idx] = G
                conflict_changed = True
                continue
            col_to_rows.setdefault(col, []).append(int(row_idx))

        for col, rows in col_to_rows.items():
            if len(rows) <= 1:
                continue
            # Winner = row with highest IoU for this col.
            winner = max(rows, key=lambda r: iou[r, col])
            for r in rows:
                if r != winner:
                    pointers[r] += 1
                    conflict_changed = True

        if not conflict_changed:
            break
    else:  # pragma: no cover — only reached if loop exhausts
        logger.warning(
            "IoU assignment did not converge within %d iterations.", max_iter
        )

    best_col = np.where(
        pointers < G,
        sorted_cols[np.arange(N), np.minimum(pointers, G - 1)],
        -1,
    ).astype(np.int64)

    # Apply threshold (only after resolution — so losers who fall below
    # are consistently flagged).
    for i in range(N):
        c = best_col[i]
        if c < 0 or iou[i, c] < threshold:
            best_col[i] = -1

    return best_col, iterations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def assign_rois(
    roi_masks: Sequence[np.ndarray],
    roi_names: Sequence[str],
    label_projection: np.ndarray,
    label_lookup: Dict[int, str],
    *,
    threshold: float = DEFAULT_IOU_THRESHOLD,
    max_iter: int = MAX_ASSIGNMENT_ITERATIONS,
) -> AssignmentResult:
    """Assign each ROI to its best-matching glomerulus.

    Args:
        roi_masks: List of 2D boolean arrays (one per ROI).
        roi_names: Original ROI names, same length as ``roi_masks``.
        label_projection: 2D uint16 array of glomerulus ids (0 = bg).
        label_lookup: ``{id → name}`` mapping from the
            :class:`~door_toolkit.atlas_align.io.AtlasBundle`.
        threshold: IoU below which an ROI is flagged unassigned.
        max_iter: Cap on greedy-resolution iterations.

    Returns:
        :class:`AssignmentResult` bundling per-ROI assignments plus
        summary statistics.
    """
    if len(roi_masks) != len(roi_names):
        raise ValueError(
            f"len(roi_masks) ({len(roi_masks)}) != len(roi_names) "
            f"({len(roi_names)})"
        )

    t0 = time.perf_counter()
    iou, glom_ids = iou_matrix(roi_masks, label_projection)
    best_col, iterations = _greedy_resolve(iou, threshold, max_iter)

    assignments: List[Assignment] = []
    for i, roi_name in enumerate(roi_names):
        col = int(best_col[i])
        if col < 0:
            # Unassigned — but still record top-3 candidates for the UI.
            if iou.shape[1] > 0:
                top_indices = np.argsort(-iou[i])[:3]
            else:
                top_indices = np.array([], dtype=np.int64)
            alternates = [
                (
                    glom_ids[j],
                    label_lookup.get(glom_ids[j], f"UNK_{glom_ids[j]}"),
                    float(iou[i, j]),
                )
                for j in top_indices
                if iou[i, j] > 0.0
            ]
            assignments.append(
                Assignment(
                    roi_index=i,
                    roi_name=roi_name,
                    glomerulus_id=0,
                    glomerulus_name=f"UNK_{roi_name}",
                    iou=float(iou[i].max()) if iou.shape[1] else 0.0,
                    alternates=alternates,
                    above_threshold=False,
                )
            )
        else:
            gid = glom_ids[col]
            # Top-3 excluding the assigned column.
            top_indices = np.argsort(-iou[i])
            alternates = []
            for j in top_indices:
                if int(j) == col:
                    continue
                if iou[i, j] <= 0.0:
                    break
                alternates.append(
                    (
                        glom_ids[j],
                        label_lookup.get(glom_ids[j], f"UNK_{glom_ids[j]}"),
                        float(iou[i, j]),
                    )
                )
                if len(alternates) >= 3:
                    break
            assignments.append(
                Assignment(
                    roi_index=i,
                    roi_name=roi_name,
                    glomerulus_id=int(gid),
                    glomerulus_name=label_lookup.get(
                        int(gid), f"UNK_{int(gid)}"
                    ),
                    iou=float(iou[i, col]),
                    alternates=alternates,
                    above_threshold=True,
                )
            )

    above = sum(1 for a in assignments if a.above_threshold)
    mean_iou = float(np.mean([a.iou for a in assignments])) if assignments else 0.0

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "assign_rois: %d ROIs, mean IoU %.3f, %d above / %d below "
        "threshold=%.2f, %d iterations, %.1f ms",
        len(assignments), mean_iou, above, len(assignments) - above,
        threshold, iterations, elapsed_ms,
    )

    return AssignmentResult(
        assignments=assignments,
        iou_matrix=iou,
        glomerulus_ids=glom_ids,
        mean_iou=mean_iou,
        n_above_threshold=above,
        n_below_threshold=len(assignments) - above,
        iterations=iterations,
        elapsed_ms=elapsed_ms,
    )
