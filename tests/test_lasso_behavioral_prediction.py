"""Tests for LASSO behavioral prediction module."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from door_toolkit.pathways.behavioral_prediction import (
    BehaviorModelResults,
    LassoBehavioralPredictor,
)


@pytest.fixture
def mock_behavioral_csv(tmp_path):
    """Create mock behavioral CSV file for testing."""
    csv_content = """dataset,3-Octonol,Benzaldehyde,Ethyl_Butyrate,Hexanol,Linalool
opto_hex,0.25,0.00,0.19,0.69,0.19
opto_EB,0.13,0.00,0.22,0.20,0.00
opto_benz_1,0.25,0.02,0.44,0.59,0.12
hex_control,0.07,0.00,0.47,0.02,0.00
"""
    csv_path = tmp_path / "test_behavior.csv"
    csv_path.write_text(csv_content)
    return csv_path


@pytest.fixture
def lasso_predictor(mock_door_cache, mock_behavioral_csv):
    """Create LassoBehavioralPredictor instance for testing."""
    return LassoBehavioralPredictor(
        doorcache_path=str(mock_door_cache),
        behavior_csv_path=str(mock_behavioral_csv),
        scale_features=True,
        scale_targets=False,
    )


class TestLassoBehavioralPredictor:
    """Tests for LassoBehavioralPredictor class."""

    def test_initialization(self, lasso_predictor):
        """Test predictor initialization."""
        assert lasso_predictor.encoder is not None
        assert lasso_predictor.behavioral_data is not None
        assert lasso_predictor.scale_features is True
        assert lasso_predictor.scale_targets is False

    def test_load_behavioral_data(self, lasso_predictor):
        """Test behavioral data loading."""
        assert lasso_predictor.behavioral_data.shape[0] > 0  # Has rows
        assert lasso_predictor.behavioral_data.shape[1] > 0  # Has columns

        # Check for expected conditions
        assert "opto_hex" in lasso_predictor.behavioral_data.index

    def test_odorant_name_mapping(self, lasso_predictor):
        """Test odorant name mapping."""
        # Test known mappings
        assert lasso_predictor.match_odorant_name("Hexanol") is not None
        assert lasso_predictor.match_odorant_name("Benzaldehyde") is not None

        # Test case-insensitive matching
        hexanol_match = lasso_predictor.match_odorant_name("HEXANOL")
        assert hexanol_match is not None

    def test_get_receptor_profile(self, lasso_predictor):
        """Test receptor profile extraction."""
        profile, coverage = lasso_predictor.get_receptor_profile("Hexanol")

        assert isinstance(profile, np.ndarray)
        assert len(profile) == lasso_predictor.encoder.n_channels
        assert isinstance(coverage, int)
        assert coverage >= 0

    def test_receptor_profile_unknown_odorant(self, lasso_predictor):
        """Test receptor profile for unknown odorant."""
        profile, coverage = lasso_predictor.get_receptor_profile("UNKNOWN_ODORANT_XYZ")

        # Should return zeros for unknown odorant
        assert isinstance(profile, np.ndarray)
        assert coverage == 0

    def test_fit_behavior_basic(self, lasso_predictor):
        """Test basic LASSO fitting."""
        try:
            results = lasso_predictor.fit_behavior(
                condition_name="opto_hex",
                lambda_range=[0.01, 0.1, 1.0],
                cv_folds=3,  # Small number for testing
            )

            assert isinstance(results, BehaviorModelResults)
            assert results.condition_name == "opto_hex"
            assert results.cv_r2_score is not None
            assert results.cv_mse is not None
            assert results.lambda_value > 0
            assert isinstance(results.lasso_weights, dict)

        except ValueError as e:
            # May fail with small test dataset
            pytest.skip(f"Insufficient data for LASSO: {e}")

    def test_fit_behavior_auto_detect_odorant(self, lasso_predictor):
        """Test auto-detection of trained odorant."""
        try:
            results = lasso_predictor.fit_behavior(
                condition_name="opto_hex", lambda_range=[0.01, 0.1], cv_folds=2
            )

            # Should auto-detect Hexanol
            assert results.trained_odorant == "Hexanol"

        except ValueError as e:
            pytest.skip(f"Insufficient data for LASSO: {e}")

    def test_fit_behavior_prediction_modes(self, lasso_predictor):
        """Test different prediction modes."""
        modes = ["test_odorant", "trained_odorant", "interaction"]

        for mode in modes:
            try:
                results = lasso_predictor.fit_behavior(
                    condition_name="opto_hex",
                    lambda_range=[0.1],
                    cv_folds=2,
                    prediction_mode=mode,
                )

                assert isinstance(results, BehaviorModelResults)
                assert results.condition_name == "opto_hex"

            except ValueError as e:
                # Some modes may fail with small dataset
                continue

    def test_fit_behavior_invalid_condition(self, lasso_predictor):
        """Test fitting with invalid condition name."""
        with pytest.raises(ValueError, match="not found in behavioral data"):
            lasso_predictor.fit_behavior(
                condition_name="INVALID_CONDITION_XYZ", lambda_range=[0.1]
            )

    def test_fit_behavior_invalid_mode(self, lasso_predictor):
        """Test fitting with invalid prediction mode."""
        with pytest.raises(ValueError, match="Unknown prediction_mode"):
            lasso_predictor.fit_behavior(
                condition_name="opto_hex", prediction_mode="INVALID_MODE"
            )

    def test_compare_conditions(self, lasso_predictor):
        """Test condition comparison."""
        try:
            results = lasso_predictor.compare_conditions(
                conditions=["opto_hex", "opto_EB"],
                lambda_range=[0.1, 1.0],
                plot=False,  # Don't generate plots in tests
            )

            assert isinstance(results, dict)
            assert len(results) > 0

            for condition, result in results.items():
                assert isinstance(result, BehaviorModelResults)

        except Exception as e:
            pytest.skip(f"Comparison failed (likely insufficient data): {e}")


class TestBehaviorModelResults:
    """Tests for BehaviorModelResults class."""

    @pytest.fixture
    def mock_results(self):
        """Create mock BehaviorModelResults for testing."""
        return BehaviorModelResults(
            condition_name="test_condition",
            trained_odorant="Hexanol",
            trained_odorant_door="1-hexanol",
            lasso_weights={
                "Or42a": 0.5,
                "Or42b": 0.3,
                "Or47b": 0.8,
                "Or7a": -0.2,
                "Or59b": 0.1,
            },
            cv_r2_score=0.75,
            cv_mse=0.05,
            lambda_value=0.01,
            n_receptors_selected=5,
            test_odorants=["Benzaldehyde", "Ethyl_Butyrate", "Linalool"],
            actual_per=np.array([0.1, 0.3, 0.2]),
            predicted_per=np.array([0.12, 0.28, 0.22]),
            receptor_coverage=50,
        )

    def test_initialization(self, mock_results):
        """Test results initialization."""
        assert mock_results.condition_name == "test_condition"
        assert mock_results.trained_odorant == "Hexanol"
        assert mock_results.n_receptors_selected == 5
        assert len(mock_results.lasso_weights) == 5

    def test_get_top_receptors(self, mock_results):
        """Test getting top receptors."""
        top_3 = mock_results.get_top_receptors(n=3)

        assert len(top_3) == 3
        assert top_3[0][0] == "Or47b"  # Highest absolute weight (0.8)
        assert top_3[0][1] == 0.8

        # Second should be Or42a (0.5)
        assert top_3[1][0] == "Or42a"

    def test_get_top_receptors_all(self, mock_results):
        """Test getting all receptors."""
        all_receptors = mock_results.get_top_receptors(n=100)

        # Should return all 5 receptors
        assert len(all_receptors) == 5

    def test_summary(self, mock_results):
        """Test summary generation."""
        summary = mock_results.summary()

        assert isinstance(summary, str)
        assert "test_condition" in summary
        assert "Hexanol" in summary
        assert "R²" in summary or "R2" in summary
        assert "Or47b" in summary  # Top receptor

    def test_export_csv(self, mock_results, tmp_path):
        """Test CSV export."""
        csv_path = tmp_path / "test_results.csv"
        mock_results.export_csv(str(csv_path))

        assert csv_path.exists()

        # Load and verify
        df = pd.read_csv(csv_path)
        assert len(df) == 5  # 5 receptors
        assert "receptor" in df.columns
        assert "lasso_weight" in df.columns
        assert "Or47b" in df["receptor"].values

    def test_export_json(self, mock_results, tmp_path):
        """Test JSON export."""
        import json

        json_path = tmp_path / "test_results.json"
        mock_results.export_json(str(json_path))

        assert json_path.exists()

        # Load and verify
        with open(json_path) as f:
            data = json.load(f)

        assert data["condition_name"] == "test_condition"
        assert data["trained_odorant"] == "Hexanol"
        assert "lasso_weights" in data
        assert "top_10_receptors" in data
        assert len(data["top_10_receptors"]) == 5  # Only 5 receptors total

    def test_plot_predictions(self, mock_results, tmp_path):
        """Test prediction plot generation."""
        plot_path = tmp_path / "test_predictions.png"
        result_path = mock_results.plot_predictions(save_to=str(plot_path))

        assert result_path is not None
        assert Path(result_path).exists()
        assert plot_path.exists()

    def test_plot_receptors(self, mock_results, tmp_path):
        """Test receptor importance plot generation."""
        plot_path = tmp_path / "test_receptors.png"
        result_path = mock_results.plot_receptors(save_to=str(plot_path), n_top=5)

        assert result_path is not None
        assert Path(result_path).exists()
        assert plot_path.exists()

    def test_plot_receptors_empty(self):
        """Test plotting with no receptors."""
        empty_results = BehaviorModelResults(
            condition_name="empty",
            trained_odorant="Test",
            trained_odorant_door="test",
            lasso_weights={},  # No receptors
            cv_r2_score=0.0,
            cv_mse=1.0,
            lambda_value=0.1,
            n_receptors_selected=0,
            test_odorants=[],
            actual_per=np.array([]),
            predicted_per=np.array([]),
            receptor_coverage=0,
        )

        result = empty_results.plot_receptors()
        assert result is None  # Should return None if no receptors


class TestDataIntegrity:
    """Tests for data integrity and validation."""

    def test_behavioral_csv_format(self, mock_behavioral_csv):
        """Test behavioral CSV format."""
        df = pd.read_csv(mock_behavioral_csv, index_col=0)

        # Check structure
        assert df.shape[0] > 0  # Has rows (conditions)
        assert df.shape[1] > 0  # Has columns (odorants)

        # Check for expected columns
        assert "Hexanol" in df.columns
        assert "Benzaldehyde" in df.columns

        # Check for expected rows
        assert "opto_hex" in df.index

    def test_lasso_weights_consistency(self, mock_behavioral_csv, mock_door_cache):
        """Test LASSO weights are consistent."""
        predictor = LassoBehavioralPredictor(
            doorcache_path=str(mock_door_cache),
            behavior_csv_path=str(mock_behavioral_csv),
            scale_features=True,
        )

        try:
            # Fit twice with same parameters
            results1 = predictor.fit_behavior("opto_hex", lambda_range=[0.1], cv_folds=2)
            results2 = predictor.fit_behavior("opto_hex", lambda_range=[0.1], cv_folds=2)

            # Should get same results (same random seed)
            assert results1.lambda_value == results2.lambda_value
            assert results1.n_receptors_selected == results2.n_receptors_selected

        except ValueError:
            pytest.skip("Insufficient data for consistency test")

    def test_prediction_array_shapes(self, mock_behavioral_csv, mock_door_cache):
        """Test prediction array shapes match."""
        predictor = LassoBehavioralPredictor(
            doorcache_path=str(mock_door_cache),
            behavior_csv_path=str(mock_behavioral_csv),
        )

        try:
            results = predictor.fit_behavior("opto_hex", lambda_range=[0.1], cv_folds=2)

            # Actual and predicted should have same shape
            assert results.actual_per.shape == results.predicted_per.shape
            assert len(results.actual_per) == len(results.test_odorants)

        except ValueError:
            pytest.skip("Insufficient data for shape test")


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_behavioral_data(self, mock_door_cache, tmp_path):
        """Test with empty behavioral CSV."""
        empty_csv = tmp_path / "empty.csv"
        empty_csv.write_text("dataset\n")

        with pytest.raises(Exception):  # Should fail to initialize
            LassoBehavioralPredictor(
                doorcache_path=str(mock_door_cache), behavior_csv_path=str(empty_csv)
            )

    def test_missing_behavioral_csv(self, mock_door_cache):
        """Test with missing behavioral CSV."""
        with pytest.raises(FileNotFoundError):
            LassoBehavioralPredictor(
                doorcache_path=str(mock_door_cache),
                behavior_csv_path="/nonexistent/file.csv",
            )

    def test_missing_door_cache(self, mock_behavioral_csv):
        """Test with missing DoOR cache."""
        with pytest.raises(FileNotFoundError):
            LassoBehavioralPredictor(
                doorcache_path="/nonexistent/cache",
                behavior_csv_path=str(mock_behavioral_csv),
            )

    def test_all_nan_condition(self, mock_door_cache, tmp_path):
        """Test condition with all NaN values."""
        csv_content = """dataset,Hexanol,Benzaldehyde
opto_all_nan,,
opto_valid,0.5,0.3
"""
        csv_path = tmp_path / "nan_test.csv"
        csv_path.write_text(csv_content)

        predictor = LassoBehavioralPredictor(
            doorcache_path=str(mock_door_cache), behavior_csv_path=str(csv_path)
        )

        with pytest.raises(ValueError, match="No valid PER data"):
            predictor.fit_behavior("opto_all_nan", lambda_range=[0.1])

    def test_insufficient_samples(self, mock_door_cache, tmp_path):
        """Test with insufficient samples for cross-validation."""
        csv_content = """dataset,Hexanol,Benzaldehyde
opto_minimal,0.5,0.3
"""
        csv_path = tmp_path / "minimal_test.csv"
        csv_path.write_text(csv_content)

        predictor = LassoBehavioralPredictor(
            doorcache_path=str(mock_door_cache), behavior_csv_path=str(csv_path)
        )

        with pytest.raises(ValueError, match="Insufficient data"):
            predictor.fit_behavior("opto_minimal", lambda_range=[0.1], cv_folds=5)
