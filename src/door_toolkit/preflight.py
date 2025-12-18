"""
Preflight utilities for training sanity checks.

These helpers are intentionally pure / lightweight so they can be unit-tested
without running full model training.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.metrics import balanced_accuracy_score


def optimal_balanced_accuracy_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> Tuple[float, float]:
    """
    Choose the probability threshold that maximizes balanced accuracy.

    Notes:
    - Uses `y_pred = (y_prob >= threshold)` convention.
    - If `y_true` has only one class, returns (0.5, 0.5).
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_true_arr.ndim != 1 or y_prob_arr.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")

    if np.unique(y_true_arr).size < 2:
        return 0.5, 0.5

    unique_scores = np.unique(y_prob_arr)
    if unique_scores.size == 1:
        threshold = float(unique_scores[0])
        bal_acc = float(balanced_accuracy_score(y_true_arr, y_prob_arr >= threshold))
        return threshold, bal_acc

    midpoints = (unique_scores[:-1] + unique_scores[1:]) / 2.0
    thresholds = np.concatenate(([0.0], midpoints, [1.0]))

    best_threshold = 0.5
    best_bal_acc = -np.inf
    for threshold in thresholds:
        bal_acc = balanced_accuracy_score(y_true_arr, y_prob_arr >= threshold)
        if (bal_acc > best_bal_acc) or (bal_acc == best_bal_acc and float(threshold) < best_threshold):
            best_bal_acc = float(bal_acc)
            best_threshold = float(threshold)

    return best_threshold, float(best_bal_acc)


def build_overfit_diagnostics(
    *,
    tiny_subset_flies: Sequence[str],
    per_fly_class_counts: Mapping[str, Mapping[str, int]],
    fixed_threshold: float,
    optimized_threshold: float,
    loss_gate_passed: bool,
    optimized_balacc_gate_passed: bool,
    final_passed: bool,
) -> Dict[str, object]:
    """Build a JSON-serializable diagnostics block for the overfit preflight."""
    selected_counts: Dict[str, Dict[str, int]] = {}
    for fly_id in tiny_subset_flies:
        counts = per_fly_class_counts.get(fly_id, {})
        selected_counts[str(fly_id)] = {
            "pos": int(counts.get("pos", 0)),
            "neg": int(counts.get("neg", 0)),
        }

    tiny_subset_pos = int(sum(c["pos"] for c in selected_counts.values()))
    tiny_subset_neg = int(sum(c["neg"] for c in selected_counts.values()))
    tiny_subset_trials = int(tiny_subset_pos + tiny_subset_neg)

    return {
        "tiny_subset_flies": [str(f) for f in tiny_subset_flies],
        "tiny_subset_trials": tiny_subset_trials,
        "tiny_subset_pos": tiny_subset_pos,
        "tiny_subset_neg": tiny_subset_neg,
        "per_fly_class_counts": selected_counts,
        "overfit_thresholds": {
            "fixed_threshold": float(fixed_threshold),
            "optimized_threshold": float(optimized_threshold),
        },
        "overfit_gate": {
            "loss_gate_passed": bool(loss_gate_passed),
            "optimized_balacc_gate_passed": bool(optimized_balacc_gate_passed),
            "final_passed": bool(final_passed),
        },
    }


def select_tiny_overfit_flies(
    per_fly_class_counts: Mapping[str, Mapping[str, int]],
    *,
    initial_k: int = 3,
    min_pos: int = 5,
    min_neg: int = 5,
    max_k: int = 8,
) -> Tuple[List[str], int]:
    """
    Deterministically select a small set of flies with enough positives/negatives.

    Selection rule:
    - Try `k = initial_k, initial_k+1, ..., max_k`
    - Return the first subset found with total_pos >= min_pos and total_neg >= min_neg
    - If none is found, return the best-effort first `max_k` flies (deterministic ordering)

    Ordering heuristic (deterministic):
    - Prefer flies with both classes (pos>0 and neg>0)
    - Prefer higher `min(pos, neg)` (more balanced)
    - Prefer higher `pos`, then higher `neg`
    - Tie-break by `fly_id`
    """
    if initial_k <= 0:
        raise ValueError("initial_k must be > 0")
    if max_k < initial_k:
        raise ValueError("max_k must be >= initial_k")

    def _get_pos_neg(fly_id: str) -> Tuple[int, int]:
        counts = per_fly_class_counts[fly_id]
        pos = int(counts.get("pos", 0))
        neg = int(counts.get("neg", 0))
        return pos, neg

    fly_ids = list(per_fly_class_counts.keys())
    if not fly_ids:
        return [], 0

    fly_ids_sorted = sorted(
        fly_ids,
        key=lambda fid: (
            -int(_get_pos_neg(fid)[0] > 0 and _get_pos_neg(fid)[1] > 0),
            -min(_get_pos_neg(fid)[0], _get_pos_neg(fid)[1]),
            -_get_pos_neg(fid)[0],
            -_get_pos_neg(fid)[1],
            fid,
        ),
    )

    def _meets_constraints(selected: List[str]) -> bool:
        total_pos = sum(_get_pos_neg(fid)[0] for fid in selected)
        total_neg = sum(_get_pos_neg(fid)[1] for fid in selected)
        return total_pos >= min_pos and total_neg >= min_neg

    def _dfs_choose(
        k: int,
        start_idx: int,
        selected: List[str],
        pos_sum: int,
        neg_sum: int,
    ) -> List[str] | None:
        if len(selected) == k:
            return selected if (pos_sum >= min_pos and neg_sum >= min_neg) else None

        remaining_needed = k - len(selected)
        remaining = fly_ids_sorted[start_idx:]
        if len(remaining) < remaining_needed:
            return None

        # Prune if even the best possible remaining flies cannot meet constraints.
        remaining_pos = sorted((_get_pos_neg(fid)[0] for fid in remaining), reverse=True)
        remaining_neg = sorted((_get_pos_neg(fid)[1] for fid in remaining), reverse=True)
        if pos_sum + sum(remaining_pos[:remaining_needed]) < min_pos:
            return None
        if neg_sum + sum(remaining_neg[:remaining_needed]) < min_neg:
            return None

        for i in range(start_idx, len(fly_ids_sorted)):
            fid = fly_ids_sorted[i]
            pos_i, neg_i = _get_pos_neg(fid)
            result = _dfs_choose(
                k,
                i + 1,
                selected + [fid],
                pos_sum + pos_i,
                neg_sum + neg_i,
            )
            if result is not None:
                return result
        return None

    for k in range(initial_k, max_k + 1):
        selection = _dfs_choose(k, 0, [], 0, 0)
        if selection is not None and _meets_constraints(selection):
            return selection, k

    fallback_k = min(max_k, len(fly_ids_sorted))
    return fly_ids_sorted[:fallback_k], fallback_k
