"""Tests for multi-condition leave-one-odor-out LOOCV regression."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from door_toolkit.multicond_loocv import (
    aggregate_weights,
    build_X_for_odors,
    detect_odor_columns,
    export_all,
    load_per_csv,
    loocv_fit_predict,
    make_targets,
    run_multicond_loocv,
)

FIXTURE_CSV = Path(__file__).parent / "data" / "multicond_loocv_fixture.csv"

# 7 odor fixture columns
FIXTURE_ODORS = [
    "Odor_A", "Odor_B", "Odor_C", "Odor_D", "Odor_E", "Odor_F", "Odor_G",
]


# ---------------------------------------------------------------------------
# Mock feature builder for tests (deterministic, 7 odors -> 3 features)
# ---------------------------------------------------------------------------

def _mock_feature_builder(odors):
    """Deterministic 7-odor feature matrix for stable sparse fits."""
    n = len(odors)
    rng = np.random.RandomState(42)
    X = rng.rand(n, 3).astype(np.float64)
    feature_names = ["feat_0", "feat_1", "feat_2"]
    return X, feature_names, {"source": "unit_test"}


# ---------------------------------------------------------------------------
# CSV loading and odor detection
# ---------------------------------------------------------------------------


class TestLoadAndDetect:
    def test_load_per_csv(self):
        df = load_per_csv(str(FIXTURE_CSV))
        assert "opto_AIR" in df.index
        assert len(df) == 5

    def test_detect_odor_columns_returns_7(self):
        df = load_per_csv(str(FIXTURE_CSV))
        odors = detect_odor_columns(df)
        assert len(odors) == 7
        assert odors == FIXTURE_ODORS

    def test_detect_odor_columns_errors_on_wrong_count(self, tmp_path):
        # CSV with only 3 odor columns
        csv_path = tmp_path / "bad.csv"
        csv_path.write_text(
            "dataset,O1,O2,O3\nopto_AIR,0.1,0.2,0.3\n"
        )
        df = load_per_csv(str(csv_path))
        with pytest.raises(ValueError, match="Expected exactly 7"):
            detect_odor_columns(df)

    def test_detect_excludes_mostly_nan_columns(self, tmp_path):
        """Columns with >50% NaN should be excluded."""
        csv_path = tmp_path / "sparse.csv"
        lines = [
            "dataset,O1,O2,O3,O4,O5,O6,O7,BadCol",
            "r1,0.1,0.2,0.3,0.4,0.5,0.6,0.7,",
            "r2,0.2,0.3,0.4,0.5,0.6,0.7,0.8,",
            "r3,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.1",
            "r4,0.4,0.5,0.6,0.7,0.8,0.9,1.0,",
            "r5,0.5,0.6,0.7,0.8,0.9,1.0,0.1,",
        ]
        csv_path.write_text("\n".join(lines) + "\n")
        df = load_per_csv(str(csv_path))
        odors = detect_odor_columns(df)
        assert "BadCol" not in odors
        assert len(odors) == 7


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


class TestMakeTargets:
    def test_control_target_uses_raw_centered_per(self):
        df = load_per_csv(str(FIXTURE_CSV))
        odors = detect_odor_columns(df)
        y_raw, y_centered, cmean = make_targets(df, odors, "opto_AIR", "opto_AIR")

        # y_raw should be the raw PER values for opto_AIR
        expected_raw = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
        np.testing.assert_allclose(y_raw, expected_raw)

        # Mean-centered
        assert abs(cmean - np.mean(expected_raw)) < 1e-12
        np.testing.assert_allclose(
            y_centered, expected_raw - np.mean(expected_raw), atol=1e-12
        )
        # Sum of centered should be ~0
        assert abs(np.sum(y_centered)) < 1e-10

    def test_trained_target_uses_delta_centered(self):
        df = load_per_csv(str(FIXTURE_CSV))
        odors = detect_odor_columns(df)
        y_raw, y_centered, cmean = make_targets(df, odors, "opto_AIR", "opto_EB")

        # y_raw should be ΔPER = opto_EB - opto_AIR
        control = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70])
        trained = np.array([0.15, 0.25, 0.35, 0.50, 0.55, 0.65, 0.80])
        expected_delta = trained - control
        np.testing.assert_allclose(y_raw, expected_delta)

        # Then mean-centered
        np.testing.assert_allclose(
            y_centered, expected_delta - np.mean(expected_delta), atol=1e-12
        )
        assert abs(np.sum(y_centered)) < 1e-10

    def test_control_target_is_not_delta(self):
        """Control must use raw PER, NOT delta (which would be all zeros)."""
        df = load_per_csv(str(FIXTURE_CSV))
        odors = detect_odor_columns(df)
        y_raw, _, _ = make_targets(df, odors, "opto_AIR", "opto_AIR")

        # If it incorrectly computed delta, y_raw would be all zeros
        assert np.any(y_raw != 0.0), "Control target must be raw PER, not delta"


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------


class TestBuildX:
    def test_shape_consistent(self):
        odors = FIXTURE_ODORS
        X, feat_names, meta = build_X_for_odors(
            odors, _mock_feature_builder, intersection=True
        )
        assert X.shape == (7, 3)
        assert len(feat_names) == 3

    def test_errors_on_zero_features(self):
        def empty_builder(odors):
            return np.zeros((len(odors), 0)), [], {}

        with pytest.raises(ValueError, match="0 features"):
            build_X_for_odors(FIXTURE_ODORS, empty_builder, intersection=True)


# ---------------------------------------------------------------------------
# LOOCV
# ---------------------------------------------------------------------------


class TestLOOCV:
    def test_loocv_has_7_folds(self):
        rng = np.random.RandomState(42)
        X = rng.rand(7, 3)
        y = rng.rand(7) - 0.5

        result = loocv_fit_predict(
            X, y,
            model="lasso",
            alpha_grid=[0.001, 0.01, 0.1],
            seed=0,
        )
        assert result["fold_preds"].shape == (7,)
        assert result["fold_true"].shape == (7,)
        assert result["fold_weights"].shape == (7, 3)
        assert result["metrics"]["n_folds"] == 7

    def test_loocv_deterministic(self):
        rng = np.random.RandomState(42)
        X = rng.rand(7, 3)
        y = rng.rand(7) - 0.5

        r1 = loocv_fit_predict(
            X, y, model="lasso", alpha_grid=[0.01, 0.1], seed=0
        )
        r2 = loocv_fit_predict(
            X, y, model="lasso", alpha_grid=[0.01, 0.1], seed=0
        )
        np.testing.assert_allclose(r1["fold_preds"], r2["fold_preds"])
        np.testing.assert_allclose(r1["fold_weights"], r2["fold_weights"])
        assert r1["metrics"]["loo_mse"] == r2["metrics"]["loo_mse"]

    def test_loocv_elasticnet(self):
        rng = np.random.RandomState(42)
        X = rng.rand(7, 3)
        y = rng.rand(7) - 0.5

        result = loocv_fit_predict(
            X, y,
            model="elasticnet",
            alpha_grid=[0.01, 0.1],
            l1_ratio=0.5,
            seed=0,
        )
        assert result["fold_weights"].shape == (7, 3)
        assert "loo_mse" in result["metrics"]


# ---------------------------------------------------------------------------
# Weight aggregation
# ---------------------------------------------------------------------------


class TestAggregateWeights:
    def test_mean_and_stability(self):
        # 3 folds, 2 features: all positive for feature 0, mixed for feature 1
        fw = np.array([
            [1.0, -0.5],
            [2.0, 0.3],
            [1.5, -0.2],
        ])
        mean_w, std_w, stability = aggregate_weights(fw)

        np.testing.assert_allclose(mean_w, [1.5, -0.4 / 3])
        assert stability[0] == 1.0  # All positive -> 100% agreement
        assert stability[1] < 1.0   # Mixed signs


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_creates_all_files(self, tmp_path):
        odors = FIXTURE_ODORS
        feature_names = ["f0", "f1", "f2"]
        conditions = ["opto_AIR", "opto_EB"]

        # Fake LOOCV results
        n = len(odors)
        fake_result = {
            "fold_preds": np.zeros(n),
            "fold_true": np.zeros(n),
            "fold_weights": np.ones((n, 3)),
            "fold_intercepts": np.zeros(n),
            "selected_hparams": {
                "fold_alphas": [0.01] * n,
                "fold_l1_ratios": [0.5] * n,
                "fold_inner_mse": [0.0] * n,
            },
            "metrics": {
                "loo_mse": 0.01,
                "loo_r": 0.9,
                "loo_r_pval": 0.01,
                "loo_r2": 0.8,
                "n_folds": n,
            },
        }

        condition_data = {
            cond: {
                "y_raw": np.zeros(n),
                "y_centered": np.zeros(n),
                "centering_mean": 0.0,
                "loocv_result": fake_result,
            }
            for cond in conditions
        }

        overview_path = export_all(
            str(tmp_path),
            conditions=conditions,
            control_row="opto_AIR",
            odors=odors,
            feature_names=feature_names,
            condition_data=condition_data,
        )

        assert Path(overview_path).exists()
        overview_df = pd.read_csv(overview_path)
        assert len(overview_df) == 2
        assert set(overview_df["condition"]) == {"opto_AIR", "opto_EB"}

        # Check per-condition files
        for cond in conditions:
            from door_toolkit.multicond_loocv import _condition_slug
            slug = _condition_slug(cond)
            assert (tmp_path / "predictions_{0}.csv".format(slug)).exists()
            assert (tmp_path / "weights_folds_{0}.csv".format(slug)).exists()
            assert (tmp_path / "weights_mean_{0}.csv".format(slug)).exists()
            assert (tmp_path / "summary_{0}.json".format(slug)).exists()


# ---------------------------------------------------------------------------
# Integration: full pipeline with mock feature builder
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_run_multicond_loocv_end_to_end(self, tmp_path):
        result = run_multicond_loocv(
            csv_path=str(FIXTURE_CSV),
            control_row="opto_AIR",
            conditions=["opto_AIR", "opto_EB", "opto_hex", "opto_benz_1", "opto_3-oct"],
            feature_builder=_mock_feature_builder,
            model="lasso",
            alpha_grid=[0.001, 0.01, 0.1],
            seed=0,
            outdir=str(tmp_path),
        )

        assert result["overview_csv"]
        assert Path(result["overview_csv"]).exists()
        assert len(result["odors"]) == 7
        assert len(result["conditions"]) == 5

        # Verify predictions have 7 folds per condition
        for cond in result["conditions"]:
            cd = result["condition_data"][cond]
            lr = cd["loocv_result"]
            assert lr["fold_preds"].shape == (7,)
            assert lr["fold_weights"].shape == (7, 3)

    def test_determinism_across_runs(self, tmp_path):
        kwargs = dict(
            csv_path=str(FIXTURE_CSV),
            control_row="opto_AIR",
            conditions=["opto_AIR", "opto_EB"],
            feature_builder=_mock_feature_builder,
            model="lasso",
            alpha_grid=[0.01, 0.1],
            seed=0,
        )
        r1 = run_multicond_loocv(**kwargs, outdir=str(tmp_path / "run1"))
        r2 = run_multicond_loocv(**kwargs, outdir=str(tmp_path / "run2"))

        for cond in ["opto_AIR", "opto_EB"]:
            p1 = r1["condition_data"][cond]["loocv_result"]["fold_preds"]
            p2 = r2["condition_data"][cond]["loocv_result"]["fold_preds"]
            np.testing.assert_allclose(p1, p2)

            w1 = r1["condition_data"][cond]["loocv_result"]["fold_weights"]
            w2 = r2["condition_data"][cond]["loocv_result"]["fold_weights"]
            np.testing.assert_allclose(w1, w2)

    def test_feature_shape_stable_across_conditions(self, tmp_path):
        """All conditions must use the same feature set (intersection)."""
        result = run_multicond_loocv(
            csv_path=str(FIXTURE_CSV),
            control_row="opto_AIR",
            conditions=["opto_AIR", "opto_EB", "opto_hex"],
            feature_builder=_mock_feature_builder,
            model="lasso",
            alpha_grid=[0.01, 0.1],
            seed=0,
            outdir=str(tmp_path),
        )

        shapes = set()
        for cond in result["conditions"]:
            fw = result["condition_data"][cond]["loocv_result"]["fold_weights"]
            shapes.add(fw.shape)

        # All conditions should have the same (n_folds, n_features) shape
        assert len(shapes) == 1
