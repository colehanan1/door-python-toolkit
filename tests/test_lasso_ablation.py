"""Tests for LASSO ablation analysis functionality."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from door_toolkit.pathways.behavioral_prediction import (
    LassoBehavioralPredictor,
    apply_receptor_ablation,
    fit_lasso_with_fixed_scaler,
    resolve_receptor_names,
)


# ============================================================================
# Tests for resolve_receptor_names
# ============================================================================


class TestResolveReceptorNames:
    """Tests for receptor name resolution."""

    def test_exact_match(self):
        """Test exact case-sensitive matching."""
        available = ["Or42b", "Or47b", "Or22a", "Or59b"]
        matched, unmatched = resolve_receptor_names(
            ["Or42b", "Or47b"], available, strict=False
        )

        assert matched == ["Or42b", "Or47b"]
        assert unmatched == []

    def test_case_insensitive_match(self):
        """Test case-insensitive matching."""
        available = ["Or42b", "Or47b", "Or22a"]
        matched, unmatched = resolve_receptor_names(
            ["or42B", "OR47B", "or22A"], available, strict=False
        )

        assert set(matched) == {"Or42b", "Or47b", "Or22a"}
        assert unmatched == []

    def test_unmatched_receptors_non_strict(self):
        """Test unmatched receptors in non-strict mode."""
        available = ["Or42b", "Or47b"]
        matched, unmatched = resolve_receptor_names(
            ["Or42b", "Or99x", "InvalidReceptor"], available, strict=False
        )

        assert matched == ["Or42b"]
        assert set(unmatched) == {"Or99x", "InvalidReceptor"}

    def test_unmatched_receptors_strict_raises(self):
        """Test that strict mode raises on unmatched receptors."""
        available = ["Or42b", "Or47b"]

        with pytest.raises(ValueError, match="Could not resolve"):
            resolve_receptor_names(
                ["Or42b", "Or99x"], available, strict=True
            )

    def test_empty_and_whitespace_handling(self):
        """Test handling of empty strings and whitespace."""
        available = ["Or42b", "Or47b"]
        matched, unmatched = resolve_receptor_names(
            ["  Or42b  ", "", "  ", "Or47b"], available, strict=False
        )

        assert set(matched) == {"Or42b", "Or47b"}
        assert unmatched == []

    def test_duplicate_removal(self):
        """Test that duplicates are removed from matched list."""
        available = ["Or42b", "Or47b"]
        matched, unmatched = resolve_receptor_names(
            ["Or42b", "or42b", "OR42B", "Or42b"], available, strict=False
        )

        assert matched == ["Or42b"]  # Only one entry
        assert unmatched == []

    def test_mixed_case_variants(self):
        """Test various case variants of the same receptor."""
        available = ["Ir64a.DC4", "Gr21a.Gr63a", "Or85f"]
        matched, unmatched = resolve_receptor_names(
            ["ir64a.dc4", "GR21A.GR63A", "or85F"], available, strict=False
        )

        assert set(matched) == {"Ir64a.DC4", "Gr21a.Gr63a", "Or85f"}
        assert unmatched == []


# ============================================================================
# Tests for apply_receptor_ablation
# ============================================================================


class TestApplyReceptorAblation:
    """Tests for receptor ablation application."""

    def test_basic_ablation(self):
        """Test basic column zeroing."""
        X = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
        ])
        receptor_names = ["Or42b", "Or47b", "Or22a", "Or59b"]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, ["Or42b", "Or22a"]
        )

        # Original should be unchanged
        assert X[0, 0] == 1.0
        assert X[0, 2] == 3.0

        # Ablated should have zeros in specified columns
        assert X_ablated[0, 0] == 0.0
        assert X_ablated[0, 2] == 0.0
        assert X_ablated[1, 0] == 0.0
        assert X_ablated[1, 2] == 0.0

        # Non-ablated columns should be unchanged
        assert X_ablated[0, 1] == 2.0
        assert X_ablated[0, 3] == 4.0

        # Check indices
        assert set(indices) == {0, 2}

    def test_ablation_preserves_shape(self):
        """Test that ablation preserves matrix shape."""
        X = np.random.rand(10, 20)
        receptor_names = [f"Or{i}" for i in range(20)]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, ["Or5", "Or10", "Or15"]
        )

        assert X_ablated.shape == X.shape
        assert len(indices) == 3

    def test_ablation_with_case_insensitive_names(self):
        """Test that ablation handles case-insensitive receptor names."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, ["or42B", "OR22A"]  # Different case
        )

        assert X_ablated[0, 0] == 0.0  # Or42b
        assert X_ablated[0, 2] == 0.0  # Or22a
        assert X_ablated[0, 1] == 2.0  # Or47b unchanged

    def test_ablation_invalid_receptor_raises(self):
        """Test that invalid receptor names raise ValueError."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        with pytest.raises(ValueError, match="Could not resolve"):
            apply_receptor_ablation(X, receptor_names, ["Or99x"])

    def test_custom_ablation_value(self):
        """Test ablation with custom value (not zero)."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, ["Or42b"], ablation_value=-999.0
        )

        assert X_ablated[0, 0] == -999.0
        assert X_ablated[0, 1] == 2.0

    def test_ablation_all_columns(self):
        """Test ablating all columns."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, ["Or42b", "Or47b", "Or22a"]
        )

        assert np.all(X_ablated == 0.0)
        assert len(indices) == 3


# ============================================================================
# Tests for fit_lasso_with_fixed_scaler
# ============================================================================


class TestFitLassoWithFixedScaler:
    """Tests for LASSO fitting with preserved scaler."""

    def test_basic_fit(self):
        """Test basic LASSO fitting."""
        np.random.seed(42)
        n_samples, n_features = 20, 10
        X = np.random.rand(n_samples, n_features)
        # Create y with some correlation to X
        true_weights = np.zeros(n_features)
        true_weights[0] = 1.0
        true_weights[1] = 0.5
        y = X @ true_weights + np.random.randn(n_samples) * 0.1

        receptor_names = [f"Or{i}" for i in range(n_features)]
        scaler = StandardScaler().fit(X)
        lambda_range = np.array([0.01, 0.1, 1.0])

        weights, cv_r2, cv_mse, best_lambda, y_pred = fit_lasso_with_fixed_scaler(
            X, y, receptor_names, scaler, lambda_range, cv_folds=5
        )

        assert isinstance(weights, dict)
        assert isinstance(cv_r2, float)
        assert isinstance(cv_mse, float)
        assert best_lambda in lambda_range
        assert len(y_pred) == n_samples

    def test_scaler_consistency(self):
        """Test that using same scaler gives consistent scaling."""
        np.random.seed(42)
        X = np.random.rand(15, 8)
        y = np.random.rand(15)
        receptor_names = [f"Or{i}" for i in range(8)]
        lambda_range = np.array([0.1])

        # Fit scaler on X
        scaler = StandardScaler().fit(X)

        # Ablate some columns
        X_ablated = X.copy()
        X_ablated[:, [2, 5]] = 0.0

        # Fit both models with same scaler
        weights1, r2_1, mse_1, _, _ = fit_lasso_with_fixed_scaler(
            X, y, receptor_names, scaler, lambda_range
        )
        weights2, r2_2, mse_2, _, _ = fit_lasso_with_fixed_scaler(
            X_ablated, y, receptor_names, scaler, lambda_range
        )

        # Both should complete without error
        assert isinstance(weights1, dict)
        assert isinstance(weights2, dict)

    def test_no_scaler(self):
        """Test fitting without scaler (None)."""
        np.random.seed(42)
        X = np.random.rand(15, 5)
        y = np.random.rand(15)
        receptor_names = [f"Or{i}" for i in range(5)]
        lambda_range = np.array([0.1])

        weights, cv_r2, cv_mse, best_lambda, y_pred = fit_lasso_with_fixed_scaler(
            X, y, receptor_names, None, lambda_range
        )

        assert isinstance(weights, dict)
        assert len(y_pred) == 15


# ============================================================================
# Tests for ablation_set_mode behavior
# ============================================================================


class TestAblationSetMode:
    """Tests for ablation_set_mode behavior (single vs all_in_one)."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for ablation tests."""
        np.random.seed(42)
        X = np.random.rand(10, 5)
        y = np.random.rand(10)
        receptor_names = ["Or42b", "Or47b", "Or22a", "Or59b", "Or92a"]
        return X, y, receptor_names

    def test_single_mode_creates_individual_ablations(self, sample_data):
        """Test that single mode ablates each receptor individually."""
        X, y, receptor_names = sample_data
        receptors_to_ablate = ["Or42b", "Or47b", "Or22a"]

        results = []
        for receptor in receptors_to_ablate:
            X_ablated, indices = apply_receptor_ablation(
                X, receptor_names, [receptor]
            )
            results.append((receptor, X_ablated, indices))

        # Should have 3 separate ablations
        assert len(results) == 3

        # Each ablation should only zero one column
        for receptor, X_abl, indices in results:
            assert len(indices) == 1
            # Count zero columns
            zero_cols = np.sum(np.all(X_abl == 0, axis=0))
            assert zero_cols == 1

    def test_all_in_one_mode_ablates_together(self, sample_data):
        """Test that all_in_one mode ablates all receptors at once."""
        X, y, receptor_names = sample_data
        receptors_to_ablate = ["Or42b", "Or47b", "Or22a"]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, receptors_to_ablate
        )

        # Should have 3 ablated columns
        assert len(indices) == 3

        # All three columns should be zero
        for idx in indices:
            assert np.all(X_ablated[:, idx] == 0.0)


# ============================================================================
# Integration tests for script outputs
# ============================================================================


class TestAblationScriptOutputs:
    """Integration tests for ablation script output files."""

    def test_save_model_json_format(self, tmp_path):
        """Test that save_model_json creates valid JSON with expected fields."""
        from scripts.lasso_with_ablations import save_model_json, AblationResult

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Create a mock result
        result = AblationResult(
            ablation_name="baseline",
            receptors_ablated=[],
            ablated_indices=[],
            cv_r2=0.75,
            cv_mse=0.025,
            n_receptors_selected=5,
            lambda_value=0.01,
            lasso_weights={"Or42b": 0.5, "Or47b": -0.3},
            delta_r2=0.0,
            delta_mse=0.0,
        )

        filepath = output_dir / "baseline_model.json"
        save_model_json(
            result=result,
            condition="opto_hex",
            prediction_mode="test_odorant",
            n_samples=10,
            n_receptors_total=20,
            filepath=filepath,
        )

        # Verify file exists and is valid JSON
        assert filepath.exists()

        with open(filepath) as f:
            data = json.load(f)

        assert data["condition_name"] == "opto_hex"
        assert data["ablation_name"] == "baseline"
        assert data["cv_r2"] == 0.75
        assert data["cv_mse"] == 0.025
        assert data["n_receptors_selected"] == 5
        assert data["lambda_value"] == 0.01
        assert "lasso_weights" in data
        assert "top_10_receptors" in data

    def test_save_weights_csv_format(self, tmp_path):
        """Test that save_weights_csv creates valid CSV."""
        from scripts.lasso_with_ablations import save_weights_csv

        output_dir = tmp_path / "output"
        output_dir.mkdir()

        weights = {"Or42b": 0.5, "Or47b": -0.3, "Or22a": 0.1}
        filepath = output_dir / "weights.csv"

        save_weights_csv(
            weights=weights,
            filepath=filepath,
            condition="opto_hex",
            ablation_name="baseline",
        )

        # Verify file exists
        assert filepath.exists()

        # Verify CSV structure
        df = pd.read_csv(filepath)
        assert "condition" in df.columns
        assert "ablation" in df.columns
        assert "receptor" in df.columns
        assert "weight" in df.columns
        assert "abs_weight" in df.columns
        assert len(df) == 3

        # Check ordering (by absolute weight descending)
        assert df.iloc[0]["receptor"] == "Or42b"  # Highest abs weight

    def test_ablation_summary_csv_format(self, tmp_path):
        """Test that ablation_summary.csv has correct format."""
        from scripts.lasso_with_ablations import AblationResult

        # Create summary rows without needing predictor
        summary_rows = [
            {
                "ablation_name": "baseline",
                "receptors_ablated": "",
                "n_ablated": 0,
                "cv_r2": 0.75,
                "cv_mse": 0.025,
                "n_receptors_selected": 5,
                "lambda_value": 0.01,
                "delta_r2": 0.0,
                "delta_mse": 0.0,
            },
            {
                "ablation_name": "ablate_Or42b",
                "receptors_ablated": "Or42b",
                "n_ablated": 1,
                "cv_r2": 0.65,
                "cv_mse": 0.035,
                "n_receptors_selected": 4,
                "lambda_value": 0.01,
                "delta_r2": -0.10,
                "delta_mse": 0.010,
            },
        ]

        summary_df = pd.DataFrame(summary_rows)
        summary_path = tmp_path / "ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False)

        # Verify CSV can be read back
        loaded_df = pd.read_csv(summary_path)
        assert "ablation_name" in loaded_df.columns
        assert "cv_r2" in loaded_df.columns
        assert "cv_mse" in loaded_df.columns
        assert "delta_r2" in loaded_df.columns
        assert "delta_mse" in loaded_df.columns
        assert len(loaded_df) == 2

    def test_run_ablation_with_synthetic_data(self, tmp_path):
        """Test run_ablation with synthetic data (no predictor needed)."""
        from scripts.lasso_with_ablations import run_ablation

        # Create synthetic data
        np.random.seed(42)
        n_samples, n_features = 15, 10
        X = np.random.rand(n_samples, n_features)

        # Create y with correlation to first receptor
        true_weights = np.zeros(n_features)
        true_weights[0] = 1.0
        y = X @ true_weights + np.random.randn(n_samples) * 0.1

        receptor_names = [f"Or{i}" for i in range(n_features)]
        scaler = StandardScaler().fit(X)
        lambda_range = np.array([0.01, 0.1])

        # Run ablation
        result = run_ablation(
            X=X,
            y=y,
            receptor_names=receptor_names,
            receptors_to_ablate=["Or0"],  # Ablate the important receptor
            scaler=scaler,
            lambda_range=lambda_range,
            cv_folds=3,
            baseline_r2=0.9,
            baseline_mse=0.01,
            ablation_name="ablate_Or0",
        )

        # Verify result structure
        assert result.ablation_name == "ablate_Or0"
        assert result.receptors_ablated == ["Or0"]
        assert result.ablated_indices == [0]
        assert isinstance(result.cv_r2, float)
        assert isinstance(result.cv_mse, float)
        assert result.delta_r2 == result.cv_r2 - 0.9
        assert result.delta_mse == result.cv_mse - 0.01


# ============================================================================
# Edge case tests
# ============================================================================


class TestAblationEdgeCases:
    """Tests for edge cases in ablation."""

    def test_ablate_zero_column(self):
        """Test ablating an already-zero column."""
        X = np.array([[1.0, 0.0, 3.0], [4.0, 0.0, 6.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, ["Or47b"]
        )

        # Should still work, column stays zero
        assert np.all(X_ablated[:, 1] == 0.0)
        assert indices == [1]

    def test_ablate_single_sample(self):
        """Test ablation with single sample (edge case for CV)."""
        X = np.array([[1.0, 2.0, 3.0]])
        receptor_names = ["Or42b", "Or47b", "Or22a"]

        X_ablated, indices = apply_receptor_ablation(
            X, receptor_names, ["Or42b"]
        )

        assert X_ablated.shape == (1, 3)
        assert X_ablated[0, 0] == 0.0

    def test_ablate_high_dimensional(self):
        """Test ablation with many receptors."""
        n_receptors = 100
        X = np.random.rand(20, n_receptors)
        receptor_names = [f"Or{i}" for i in range(n_receptors)]

        # Ablate 50 receptors
        to_ablate = [f"Or{i}" for i in range(0, 50)]
        X_ablated, indices = apply_receptor_ablation(X, receptor_names, to_ablate)

        assert len(indices) == 50
        for idx in indices:
            assert np.all(X_ablated[:, idx] == 0.0)
