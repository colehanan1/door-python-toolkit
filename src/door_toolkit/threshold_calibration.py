"""
Threshold calibration utilities (validation → test, no leakage).

Decision → Evidence → Implementation
-----------------------------------
Decision: Choose a decision threshold using ONLY validation predictions/labels,
          then apply that threshold once on the held-out test set.
Evidence: On imbalanced datasets, a fixed threshold (e.g. 0.5) can yield
          degenerate all-negative predictions even when AUROC is strong; selecting
          the threshold on the test set would be data leakage.
Implementation: Use the same deterministic balanced-accuracy threshold search as
                the training preflight: evaluate candidate thresholds at
                midpoints between unique probabilities, plus {0.0, 1.0}, and
                tie-break by choosing the lowest threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

from door_toolkit.preflight import optimal_balanced_accuracy_threshold


def compute_optimal_threshold(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    metric: str = "balanced_accuracy",
) -> Tuple[float, float]:
    """
    Compute an optimal probability threshold for a binary classifier.

    Decision → Evidence → Implementation
    -----------------------------------
    Decision: Default to balanced accuracy as the optimization metric.
    Evidence: Balanced accuracy is robust to class imbalance and matches the
              project's primary reported thresholded metric.
    Implementation: For `metric='balanced_accuracy'`, delegate to
                    `optimal_balanced_accuracy_threshold()` (midpoint grid, deterministic
                    tie-break on lowest threshold).

    Returns:
        (threshold, best_score)
    """
    if metric != "balanced_accuracy":
        raise ValueError(f"Unsupported metric: {metric} (supported: 'balanced_accuracy')")

    threshold, best_score = optimal_balanced_accuracy_threshold(y_true, y_prob)
    return float(threshold), float(best_score)


def compute_thresholded_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
    *,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute classification metrics at a given probability threshold.

    Notes:
    - AUROC/AUPRC are threshold-independent but included for convenience so that
      `at_0.5` and `at_thr_opt_from_val` blocks are self-contained.
    - `pos_rate` is the fraction of examples predicted positive at `threshold`.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if y_true_arr.ndim != 1 or y_prob_arr.ndim != 1:
        raise ValueError("y_true and y_prob must be 1D")
    if y_true_arr.shape[0] != y_prob_arr.shape[0]:
        raise ValueError("y_true and y_prob must have the same length")

    if y_true_arr.size == 0:
        raise ValueError("y_true/y_prob must be non-empty")

    preds = (y_prob_arr >= float(threshold)).astype(int)
    pos_rate = float(np.mean(preds))

    if np.unique(y_true_arr).size < 2:
        auroc = 0.5
        auprc = float(np.mean(y_true_arr))
        bal_acc = 0.5
    else:
        auroc = float(roc_auc_score(y_true_arr, y_prob_arr))
        auprc = float(average_precision_score(y_true_arr, y_prob_arr))
        bal_acc = float(balanced_accuracy_score(y_true_arr, preds))

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "balanced_acc": float(bal_acc),
        "pos_rate": float(pos_rate),
    }


def build_threshold_calibration_eval(
    y_val_true: Sequence[int] | np.ndarray,
    y_val_prob: Sequence[float] | np.ndarray,
    y_test_true: Sequence[int] | np.ndarray,
    y_test_prob: Sequence[float] | np.ndarray,
    *,
    fixed_threshold: float = 0.5,
    metric: str = "balanced_accuracy",
) -> Dict[str, Any]:
    """
    Build a JSON-serializable val→test threshold calibration evaluation report.

    Decision → Evidence → Implementation
    -----------------------------------
    Decision: Report test metrics at BOTH `threshold=0.5` and `threshold=thr_opt_from_val`.
    Evidence: This preserves backwards compatibility while enabling fair comparisons
              when a fixed 0.5 threshold yields degenerate predictions.
    Implementation: Compute `thr_opt_from_val` on validation only, then apply it on test.
    """
    y_val_true_arr = np.asarray(y_val_true, dtype=int)
    y_val_prob_arr = np.asarray(y_val_prob, dtype=float)
    y_test_true_arr = np.asarray(y_test_true, dtype=int)
    y_test_prob_arr = np.asarray(y_test_prob, dtype=float)

    thr_opt, val_best = compute_optimal_threshold(
        y_val_true_arr,
        y_val_prob_arr,
        metric=metric,
    )

    report: Dict[str, Any] = {
        "metric": metric,
        "fixed_threshold": float(fixed_threshold),
        "thr_opt_from_val": float(thr_opt),
        "val": {
            "n_samples": int(y_val_true_arr.size),
            "pos_rate_true": float(np.mean(y_val_true_arr == 1)),
            "best_metric_value": float(val_best),
            "at_0.5": compute_thresholded_metrics(y_val_true_arr, y_val_prob_arr, threshold=fixed_threshold),
            "at_thr_opt_from_val": compute_thresholded_metrics(
                y_val_true_arr, y_val_prob_arr, threshold=thr_opt
            ),
        },
        "test": {
            "n_samples": int(y_test_true_arr.size),
            "pos_rate_true": float(np.mean(y_test_true_arr == 1)),
            "thr_from_val": float(thr_opt),
            "at_0.5": compute_thresholded_metrics(y_test_true_arr, y_test_prob_arr, threshold=fixed_threshold),
            "at_thr_opt_from_val": compute_thresholded_metrics(
                y_test_true_arr, y_test_prob_arr, threshold=thr_opt
            ),
        },
    }

    return report

