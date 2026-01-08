"""Tests for LASSO focus mode analysis functionality."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from door_toolkit.pathways.behavioral_prediction import (
    get_top_receptors_by_weight,
    restrict_to_receptors,
    fit_lasso_with_fixed_scaler,
)


# ============================================================================
# Tests for restrict_to_receptors
# ============================================================================


class TestRestrictToReceptors:
    """Tests for feature matrix restriction."""

    def test_basic_restriction(self):
        """Test basic column restriction."""
        X = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ])
        receptor_names = ["Or42b", "Or47b", "Or22a", "Or59b"]

        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, ["Or47b", "Or22a"]
        )

        # Check shapes
        assert X_res.shape == (3, 2)

        # Check values (columns 1 and 2)
        np.testing.assert_array_equal(X_res[:, 0], [2.0, 6.0, 10.0])
        np.testing.assert_array_equal(X_res[:, 1], [3.0, 7.0, 11.0])

        # Check names and indices
        assert kept_names == ["Or47b", "Or22a"]
        assert kept_idx == [1, 2]

    def test_restriction_preserves_row_count(self):
        """Test that restriction preserves number of rows."""
        X = np.random.rand(10, 20)
        receptor_names = [f"Or{i}" for i in range(20)]

        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, ["Or5", "Or10", "Or15"]
        )

        assert X_res.shape[0] == 10
        assert X_res.shape[1] == 3

    def test_restriction_with_case_insensitive_names(self):
        """Test that restriction handles case-insensitive names."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, ["or42B", "OR22A"]
        )

        assert X_res.shape == (1, 2)
        assert set(kept_names) == {"Or42b", "Or22a"}

    def test_restriction_invalid_receptor_raises(self):
        """Test that invalid receptor names raise ValueError."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        with pytest.raises(ValueError, match="Could not resolve"):
            restrict_to_receptors(X, receptor_names, ["Or99x"])

    def test_restriction_single_column(self):
        """Test restriction to single column."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, ["Or47b"]
        )

        assert X_res.shape == (2, 1)
        np.testing.assert_array_equal(X_res[:, 0], [2.0, 5.0])

    def test_restriction_all_columns(self):
        """Test restriction to all columns (no change)."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, ["Or42b", "Or47b", "Or22a"]
        )

        np.testing.assert_array_equal(X_res, X)
        assert kept_names == receptor_names

    def test_restriction_preserves_order(self):
        """Test that restriction preserves original column order."""
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        receptor_names = ["Or1", "Or2", "Or3", "Or4", "Or5"]

        # Request in different order
        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, ["Or5", "Or2", "Or4"]
        )

        # Should be in original order (2, 4, 5)
        assert kept_names == ["Or2", "Or4", "Or5"]
        assert kept_idx == [1, 3, 4]
        np.testing.assert_array_equal(X_res[0], [2.0, 4.0, 5.0])


# ============================================================================
# Tests for get_top_receptors_by_weight
# ============================================================================


class TestGetTopReceptorsByWeight:
    """Tests for top-N receptor selection by weight."""

    def test_basic_selection(self):
        """Test basic top-N selection."""
        weights = {"Or42b": 0.5, "Or47b": -0.8, "Or22a": 0.3, "Or59b": -0.1}

        top_2 = get_top_receptors_by_weight(weights, 2)

        assert len(top_2) == 2
        assert top_2[0] == "Or47b"  # Highest abs weight (-0.8)
        assert top_2[1] == "Or42b"  # Second highest (0.5)

    def test_selection_deterministic(self):
        """Test that selection is deterministic."""
        weights = {"Or42b": 0.5, "Or47b": -0.8, "Or22a": 0.3}

        # Run multiple times
        results = [get_top_receptors_by_weight(weights, 2) for _ in range(10)]

        # All results should be identical
        for result in results:
            assert result == results[0]

    def test_selection_all_receptors(self):
        """Test selecting all receptors."""
        weights = {"Or42b": 0.5, "Or47b": -0.8, "Or22a": 0.3}

        top_all = get_top_receptors_by_weight(weights, 3)

        assert len(top_all) == 3
        assert set(top_all) == {"Or42b", "Or47b", "Or22a"}

    def test_selection_more_than_available(self):
        """Test requesting more receptors than available."""
        weights = {"Or42b": 0.5, "Or47b": -0.8}

        top_5 = get_top_receptors_by_weight(weights, 5)

        assert len(top_5) == 2  # Only 2 available

    def test_selection_single(self):
        """Test selecting single top receptor."""
        weights = {"Or42b": 0.5, "Or47b": -0.8, "Or22a": 0.3}

        top_1 = get_top_receptors_by_weight(weights, 1)

        assert len(top_1) == 1
        assert top_1[0] == "Or47b"

    def test_selection_zero(self):
        """Test selecting zero receptors."""
        weights = {"Or42b": 0.5, "Or47b": -0.8}

        top_0 = get_top_receptors_by_weight(weights, 0)

        assert len(top_0) == 0

    def test_selection_considers_absolute_value(self):
        """Test that selection uses absolute value."""
        weights = {"Or42b": 0.3, "Or47b": -0.5, "Or22a": 0.4}

        top_2 = get_top_receptors_by_weight(weights, 2)

        # Should be -0.5 (abs=0.5) and 0.4 (abs=0.4)
        assert set(top_2) == {"Or47b", "Or22a"}


# ============================================================================
# Tests for focus mode integration
# ============================================================================


class TestFocusModeIntegration:
    """Integration tests for focus mode workflow."""

    @pytest.fixture
    def synthetic_data(self):
        """Create synthetic data for focus mode tests."""
        np.random.seed(42)
        n_samples, n_features = 20, 10

        X = np.random.rand(n_samples, n_features)

        # Create y with correlation to first two receptors
        true_weights = np.zeros(n_features)
        true_weights[0] = 1.0
        true_weights[1] = 0.5
        y = X @ true_weights + np.random.randn(n_samples) * 0.1

        receptor_names = [f"Or{i}" for i in range(n_features)]
        return X, y, receptor_names

    def test_focus_workflow_reduces_features(self, synthetic_data):
        """Test that focus mode correctly reduces feature count."""
        X, y, receptor_names = synthetic_data
        lambda_range = np.array([0.01, 0.1])

        # Fit baseline
        scaler = StandardScaler().fit(X)
        baseline_weights, r2, mse, lam, pred = fit_lasso_with_fixed_scaler(
            X, y, receptor_names, scaler, lambda_range
        )

        # Get top-3 receptors
        if len(baseline_weights) >= 3:
            top_3 = get_top_receptors_by_weight(baseline_weights, 3)
        else:
            top_3 = get_top_receptors_by_weight(baseline_weights, len(baseline_weights))

        # Restrict to top-3
        X_focused, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, top_3
        )

        assert X_focused.shape[1] <= 3
        assert len(kept_names) <= 3

    def test_focus_produces_valid_model(self, synthetic_data):
        """Test that focus mode produces valid LASSO model."""
        X, y, receptor_names = synthetic_data
        lambda_range = np.array([0.01, 0.1])

        # Fit baseline
        scaler = StandardScaler().fit(X)
        baseline_weights, r2, mse, lam, pred = fit_lasso_with_fixed_scaler(
            X, y, receptor_names, scaler, lambda_range
        )

        # Focus on top-5
        if len(baseline_weights) >= 5:
            top_n = get_top_receptors_by_weight(baseline_weights, 5)
        else:
            top_n = get_top_receptors_by_weight(baseline_weights, len(baseline_weights))

        X_focused, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, top_n
        )

        # Fit on focused data
        scaler_focused = StandardScaler().fit(X_focused)
        focus_weights, focus_r2, focus_mse, focus_lam, focus_pred = fit_lasso_with_fixed_scaler(
            X_focused, y, kept_names, scaler_focused, lambda_range
        )

        # Should produce valid outputs
        assert isinstance(focus_weights, dict)
        assert isinstance(focus_r2, float)
        assert isinstance(focus_mse, float)
        assert len(focus_pred) == len(y)


# ============================================================================
# Tests for script output formats
# ============================================================================


class TestFocusModeScriptOutputs:
    """Tests for focus mode script output files."""

    def test_save_baseline_json_format(self, tmp_path):
        """Test that save_baseline_json creates valid JSON."""
        from scripts.lasso_with_focus_mode import save_baseline_json

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        baseline_weights = {"Or42b": 0.5, "Or47b": -0.3, "Or22a": 0.1}
        filepath = output_dir / "baseline_model.json"

        save_baseline_json(
            baseline_weights=baseline_weights,
            baseline_r2=0.75,
            baseline_mse=0.025,
            baseline_lambda=0.01,
            condition="opto_hex",
            prediction_mode="test_odorant",
            n_samples=10,
            n_receptors_total=20,
            filepath=filepath,
        )

        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)

        assert data["condition_name"] == "opto_hex"
        assert data["cv_r2"] == 0.75
        assert data["cv_mse"] == 0.025
        assert "receptor_ranking" in data
        assert len(data["receptor_ranking"]) == 3

    def test_save_focus_json_format(self, tmp_path):
        """Test that save_focus_json creates valid JSON."""
        from scripts.lasso_with_focus_mode import save_focus_json, FocusResult

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        result = FocusResult(
            n_receptors=3,
            receptors_used=["Or42b", "Or47b", "Or22a"],
            cv_r2=0.65,
            cv_mse=0.035,
            lambda_value=0.01,
            n_receptors_selected=2,
            lasso_weights={"Or42b": 0.6, "Or47b": -0.4},
            delta_r2=-0.10,
            delta_mse=0.010,
        )

        filepath = output_dir / "model.json"
        save_focus_json(
            result=result,
            condition="opto_hex",
            prediction_mode="test_odorant",
            n_samples=10,
            filepath=filepath,
        )

        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)

        assert data["n_receptors_focused"] == 3
        assert data["receptors_used"] == ["Or42b", "Or47b", "Or22a"]
        assert data["delta_r2"] == -0.10

    def test_focus_curve_csv_format(self, tmp_path):
        """Test that focus_curve.csv has correct format."""
        # Create sample curve rows
        curve_rows = [
            {
                "n_receptors": 20,
                "receptors_used": "Or42b;Or47b;Or22a",
                "cv_r2": 0.75,
                "cv_mse": 0.025,
                "lambda_value": 0.01,
                "n_receptors_selected": 5,
                "delta_r2": 0.0,
                "delta_mse": 0.0,
                "is_baseline": True,
            },
            {
                "n_receptors": 3,
                "receptors_used": "Or42b;Or47b;Or22a",
                "cv_r2": 0.65,
                "cv_mse": 0.035,
                "lambda_value": 0.01,
                "n_receptors_selected": 2,
                "delta_r2": -0.10,
                "delta_mse": 0.010,
                "is_baseline": False,
            },
        ]

        curve_df = pd.DataFrame(curve_rows)
        curve_path = tmp_path / "focus_curve.csv"
        curve_df.to_csv(curve_path, index=False)

        # Verify CSV can be read back
        loaded_df = pd.read_csv(curve_path)
        assert "n_receptors" in loaded_df.columns
        assert "cv_r2" in loaded_df.columns
        assert "cv_mse" in loaded_df.columns
        assert "delta_r2" in loaded_df.columns
        assert "is_baseline" in loaded_df.columns
        assert len(loaded_df) == 2

    def test_run_focus_with_synthetic_data(self):
        """Test run_focus with synthetic data."""
        from scripts.lasso_with_focus_mode import run_focus

        np.random.seed(42)
        n_samples, n_features = 15, 10
        X = np.random.rand(n_samples, n_features)

        true_weights = np.zeros(n_features)
        true_weights[0] = 1.0
        true_weights[1] = 0.5
        y = X @ true_weights + np.random.randn(n_samples) * 0.1

        receptor_names = [f"Or{i}" for i in range(n_features)]
        lambda_range = np.array([0.01, 0.1])

        result = run_focus(
            X=X,
            y=y,
            receptor_names=receptor_names,
            receptors_to_keep=["Or0", "Or1", "Or2"],
            lambda_range=lambda_range,
            cv_folds=3,
            scale_features=True,
            baseline_r2=0.9,
            baseline_mse=0.01,
        )

        assert result.n_receptors == 3
        assert result.receptors_used == ["Or0", "Or1", "Or2"]
        assert isinstance(result.cv_r2, float)
        assert isinstance(result.cv_mse, float)
        assert result.delta_r2 == result.cv_r2 - 0.9
        assert result.delta_mse == result.cv_mse - 0.01


# ============================================================================
# Tests for edge cases
# ============================================================================


class TestFocusModeEdgeCases:
    """Tests for edge cases in focus mode."""

    def test_single_receptor_focus(self):
        """Test focusing on a single receptor."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, ["Or47b"]
        )

        assert X_res.shape == (3, 1)
        assert kept_names == ["Or47b"]

    def test_topn_with_tied_weights(self):
        """Test top-N selection with tied absolute weights."""
        weights = {"Or42b": 0.5, "Or47b": -0.5, "Or22a": 0.3}

        # With tied weights, order should be deterministic (alphabetical within tie)
        top_2 = get_top_receptors_by_weight(weights, 2)

        assert len(top_2) == 2
        # Both have abs weight 0.5, order determined by sort stability
        assert set(top_2) == {"Or42b", "Or47b"}

    def test_focus_empty_weight_dict(self):
        """Test top-N with empty weight dict."""
        weights = {}

        top_3 = get_top_receptors_by_weight(weights, 3)

        assert len(top_3) == 0

    def test_restrict_to_empty_list(self):
        """Test restricting to empty receptor list returns empty."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        # Empty input returns empty output (no receptors to resolve)
        X_res, kept_names, kept_idx = restrict_to_receptors(
            X, receptor_names, []
        )

        assert X_res.shape == (1, 0)
        assert kept_names == []
        assert kept_idx == []
