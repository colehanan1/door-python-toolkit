import numpy as np
from sklearn.metrics import balanced_accuracy_score

from door_toolkit.preflight import (
    build_overfit_diagnostics,
    optimal_balanced_accuracy_threshold,
    select_tiny_overfit_flies,
)


def test_optimal_balanced_accuracy_threshold_matches_bruteforce_candidates():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.05, 0.10, 0.20, 0.21, 0.22, 0.23])

    threshold, best_balacc = optimal_balanced_accuracy_threshold(y_true, y_prob)

    unique = np.unique(y_prob)
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    candidates = np.concatenate(([0.0], midpoints, [1.0]))
    brute_best = max(balanced_accuracy_score(y_true, y_prob >= t) for t in candidates)

    assert 0.0 <= threshold <= 1.0
    assert np.isclose(best_balacc, brute_best)
    assert np.isclose(balanced_accuracy_score(y_true, y_prob >= threshold), brute_best)


def test_select_tiny_overfit_flies_meets_constraints_when_possible():
    per_fly = {
        "fly_a": {"pos": 0, "neg": 10},
        "fly_b": {"pos": 2, "neg": 8},
        "fly_c": {"pos": 2, "neg": 8},
        "fly_d": {"pos": 1, "neg": 9},
        "fly_e": {"pos": 5, "neg": 5},
    }

    selected, k = select_tiny_overfit_flies(per_fly, initial_k=3, min_pos=5, min_neg=5, max_k=8)
    selected2, k2 = select_tiny_overfit_flies(per_fly, initial_k=3, min_pos=5, min_neg=5, max_k=8)

    assert selected == selected2
    assert k == k2
    assert len(selected) == 3
    assert k == 3

    total_pos = sum(per_fly[f]["pos"] for f in selected)
    total_neg = sum(per_fly[f]["neg"] for f in selected)
    assert total_pos >= 5
    assert total_neg >= 5


def test_select_tiny_overfit_flies_auto_increases_k_when_needed():
    per_fly = {
        "fly_a": {"pos": 1, "neg": 9},
        "fly_b": {"pos": 1, "neg": 9},
        "fly_c": {"pos": 1, "neg": 9},
        "fly_d": {"pos": 2, "neg": 8},
    }

    selected, k = select_tiny_overfit_flies(per_fly, initial_k=3, min_pos=5, min_neg=5, max_k=4)

    assert len(selected) == 4
    assert k == 4
    assert sum(per_fly[f]["pos"] for f in selected) >= 5
    assert sum(per_fly[f]["neg"] for f in selected) >= 5


def test_build_overfit_diagnostics_includes_required_fields():
    per_fly = {
        "fly_1": {"pos": 2, "neg": 8},
        "fly_2": {"pos": 3, "neg": 7},
    }

    diag = build_overfit_diagnostics(
        tiny_subset_flies=["fly_1", "fly_2"],
        per_fly_class_counts=per_fly,
        fixed_threshold=0.5,
        optimized_threshold=0.25,
        loss_gate_passed=True,
        optimized_balacc_gate_passed=False,
        final_passed=False,
    )

    assert set(diag.keys()) == {
        "tiny_subset_flies",
        "tiny_subset_trials",
        "tiny_subset_pos",
        "tiny_subset_neg",
        "per_fly_class_counts",
        "overfit_thresholds",
        "overfit_gate",
    }
    assert diag["overfit_thresholds"]["fixed_threshold"] == 0.5
    assert diag["overfit_thresholds"]["optimized_threshold"] == 0.25
    assert diag["overfit_gate"]["loss_gate_passed"] is True
    assert diag["overfit_gate"]["optimized_balacc_gate_passed"] is False
    assert diag["overfit_gate"]["final_passed"] is False

