import numpy as np

from door_toolkit.threshold_calibration import (
    build_threshold_calibration_eval,
    compute_optimal_threshold,
    compute_thresholded_metrics,
)


def test_compute_optimal_threshold_balanced_accuracy_midpoint_grid():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4])

    threshold, best = compute_optimal_threshold(y_true, y_prob)

    assert np.isclose(threshold, 0.25)
    assert np.isclose(best, 1.0)


def test_thresholded_metrics_differs_between_fixed_and_optimal_threshold():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.10, 0.10, 0.10, 0.20, 0.20, 0.20])

    fixed = compute_thresholded_metrics(y_true, y_prob, threshold=0.5)
    assert fixed["pos_rate"] == 0.0
    assert np.isclose(fixed["balanced_acc"], 0.5)

    thr_opt, _ = compute_optimal_threshold(y_true, y_prob)
    opt = compute_thresholded_metrics(y_true, y_prob, threshold=thr_opt)

    assert thr_opt < 0.5
    assert opt["pos_rate"] > 0.0
    assert opt["balanced_acc"] > fixed["balanced_acc"]


def test_build_threshold_calibration_eval_uses_validation_threshold_only():
    # Validation suggests a low threshold (~0.405) for perfect separation.
    y_val_true = np.array([0, 1])
    y_val_prob = np.array([0.40, 0.41])

    # Test would prefer a much higher threshold (~0.905) if (incorrectly) optimized on test.
    y_test_true = np.array([0, 1])
    y_test_prob = np.array([0.90, 0.91])

    report = build_threshold_calibration_eval(
        y_val_true,
        y_val_prob,
        y_test_true,
        y_test_prob,
    )

    assert np.isclose(report["thr_opt_from_val"], 0.405)
    assert np.isclose(report["test"]["thr_from_val"], 0.405)

    # With the val-derived threshold, both test examples are predicted positive -> BalAcc=0.5.
    assert np.isclose(report["test"]["at_thr_opt_from_val"]["balanced_acc"], 0.5)

