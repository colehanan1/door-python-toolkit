"""Tests for neural network preprocessing modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import curve_fit

from door_toolkit.neural.concentration_models import (
    ConcentrationResponseModel,
    HillParameters,
)
from door_toolkit.neural.sparse_encoding import SparseEncoder, create_kc_like_encoding
from door_toolkit.neural.preprocessor import DoORNeuralPreprocessor


class TestHillParameters:
    """Tests for HillParameters dataclass."""

    def test_initialization(self):
        """Test Hill parameters initialization."""
        params = HillParameters(
            r_max=1.0, ec50=0.1, hill_coefficient=1.5, r_baseline=0.0
        )

        assert params.r_max == 1.0
        assert params.ec50 == 0.1
        assert params.hill_coefficient == 1.5

    def test_evaluate(self):
        """Test Hill equation evaluation."""
        params = HillParameters(r_max=1.0, ec50=0.1, hill_coefficient=1.0)

        concentrations = np.array([0.01, 0.1, 1.0])
        responses = params.evaluate(concentrations)

        assert len(responses) == 3
        assert all(responses >= 0)
        assert responses[1] > responses[0]  # Higher concentration -> higher response


class TestConcentrationResponseModel:
    """Tests for ConcentrationResponseModel class."""

    def test_hill_equation(self):
        """Test Hill equation computation."""
        model = ConcentrationResponseModel()

        response = model.hill_equation(
            concentration=0.1, r_max=1.0, ec50=0.1, n=1.0
        )

        assert response > 0
        assert response <= 1.0

    def test_fit_hill_equation(self):
        """Test Hill equation fitting."""
        model = ConcentrationResponseModel()

        # Generate synthetic data
        concentrations = np.array([0.001, 0.01, 0.1, 1.0])
        responses = np.array([0.1, 0.3, 0.7, 0.9])

        params = model.fit_hill_equation(concentrations, responses)

        assert isinstance(params, HillParameters)
        assert params.r_max > 0
        assert params.ec50 > 0
        assert params.hill_coefficient > 0

    def test_predict_concentration_response(self):
        """Test concentration response prediction."""
        model = ConcentrationResponseModel()
        params = HillParameters(r_max=1.0, ec50=0.1, hill_coefficient=1.5)

        concentrations = np.logspace(-4, 0, 10)
        responses = model.predict_concentration_response(params, concentrations)

        assert len(responses) == 10
        assert all(responses >= 0)

    def test_generate_concentration_series(self):
        """Test concentration series generation."""
        model = ConcentrationResponseModel()
        params = HillParameters(r_max=1.0, ec50=0.1, hill_coefficient=1.5)

        conc, resp = model.generate_concentration_series(params, n_points=20)

        assert len(conc) == 20
        assert len(resp) == 20
        assert all(conc > 0)

    def test_add_concentration_noise(self):
        """Test noise addition."""
        model = ConcentrationResponseModel()
        clean_responses = np.array([0.5, 0.7, 0.9])

        noisy = model.add_concentration_noise(
            clean_responses, noise_type="gaussian", noise_level=0.1
        )

        assert len(noisy) == len(clean_responses)
        assert not np.allclose(noisy, clean_responses)


class TestSparseEncoder:
    """Tests for SparseEncoder class."""

    def test_initialization(self):
        """Test encoder initialization."""
        encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)

        assert encoder.n_input == 78
        assert encoder.n_output == 2000
        assert encoder.sparsity == 0.05
        assert encoder.weights.shape == (2000, 78)

    def test_encode_single(self):
        """Test encoding single ORN response."""
        encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)

        orn_response = np.random.randn(78)
        kc_response = encoder.encode(orn_response, method="winner-take-all")

        assert kc_response.shape == (2000,)
        # Check sparsity (should be close to 5%)
        sparsity = (kc_response > 0).mean()
        assert 0.03 <= sparsity <= 0.07  # Allow some tolerance

    def test_encode_batch(self):
        """Test encoding batch of ORN responses."""
        encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)

        orn_batch = np.random.randn(10, 78)
        kc_batch = encoder.encode_batch(orn_batch)

        assert kc_batch.shape == (10, 2000)

        # Check sparsity across batch
        sparsity = (kc_batch > 0).mean()
        assert 0.03 <= sparsity <= 0.07

    def test_get_sparsity_stats(self):
        """Test sparsity statistics."""
        encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)

        orn_response = np.random.randn(78)
        kc_response = encoder.encode(orn_response)

        stats = encoder.get_sparsity_stats(kc_response)

        assert "sparsity" in stats
        assert "n_active_mean" in stats
        assert 0 <= stats["sparsity"] <= 1

    def test_add_noise(self):
        """Test noise addition to sparse encoding."""
        encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)

        orn_response = np.random.randn(78)
        kc_response = encoder.encode(orn_response)

        noisy_kc = encoder.add_noise(kc_response, noise_type="gaussian", noise_level=0.1)

        assert noisy_kc.shape == kc_response.shape
        assert not np.allclose(noisy_kc, kc_response)

    def test_generate_augmented_dataset(self):
        """Test augmented dataset generation."""
        encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)

        orn_data = np.random.randn(10, 78)
        aug_orn, aug_kc = encoder.generate_augmented_dataset(
            orn_data, n_augmentations=3
        )

        # Original + 3 augmentations per sample
        assert len(aug_orn) == 10 * 4
        assert len(aug_kc) == 10 * 4
        assert aug_orn.shape[1] == 78
        assert aug_kc.shape[1] == 2000

    def test_threshold_method(self):
        """Test threshold-based sparsification."""
        encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)

        orn_response = np.random.randn(78)
        kc_response = encoder.encode(orn_response, method="threshold")

        assert kc_response.shape == (2000,)
        sparsity = (kc_response > 0).mean()
        assert sparsity > 0


class TestDoORNeuralPreprocessor:
    """Tests for DoORNeuralPreprocessor class."""

    @pytest.fixture
    def mock_preprocessor(self, mock_door_cache):
        """Create mock preprocessor."""
        return DoORNeuralPreprocessor(str(mock_door_cache), n_kc_neurons=2000)

    def test_initialization(self, mock_preprocessor):
        """Test preprocessor initialization."""
        assert mock_preprocessor.encoder is not None
        assert mock_preprocessor.sparse_encoder is not None
        assert mock_preprocessor.concentration_model is not None

    def test_create_sparse_encoding(self, mock_preprocessor):
        """Test sparse encoding creation."""
        # Use only first 10 odorants for speed
        odorants = mock_preprocessor.encoder.odorant_names[:10]

        sparse_data = mock_preprocessor.create_sparse_encoding(
            sparsity_level=0.05, odorants=odorants
        )

        assert sparse_data.shape[0] == len(odorants)
        assert sparse_data.shape[1] == 2000

        # Check sparsity
        sparsity = (sparse_data > 0).mean()
        assert 0.02 <= sparsity <= 0.08

    def test_get_dataset_statistics(self, mock_preprocessor):
        """Test dataset statistics."""
        stats = mock_preprocessor.get_dataset_statistics()

        assert "n_odorants" in stats
        assert "n_receptors" in stats
        assert "mean_response" in stats
        assert stats["n_odorants"] > 0
        assert stats["n_receptors"] > 0

    def test_create_training_validation_split(self, mock_preprocessor):
        """Test train/val split creation."""
        train, val = mock_preprocessor.create_training_validation_split(
            train_fraction=0.8, random_seed=42
        )

        assert len(train) > 0
        assert len(val) > 0
        total = len(train) + len(val)
        assert len(train) / total == pytest.approx(0.8, abs=0.05)

        # Check no overlap
        assert len(set(train) & set(val)) == 0


def test_create_kc_like_encoding():
    """Test convenience function for KC encoding."""
    orn_responses = np.random.randn(10, 78)
    kc_responses = create_kc_like_encoding(orn_responses, n_kcs=2000, sparsity=0.05)

    assert kc_responses.shape == (10, 2000)
    sparsity = (kc_responses > 0).mean()
    assert 0.03 <= sparsity <= 0.08


def test_model_mixture_interactions():
    """Test odor mixture modeling."""
    model = ConcentrationResponseModel()

    params1 = HillParameters(r_max=1.0, ec50=0.1, hill_coefficient=1.5)
    params2 = HillParameters(r_max=0.8, ec50=0.2, hill_coefficient=1.2)

    concentrations = np.array([[0.1, 0.1], [0.2, 0.05]])

    responses = model.model_mixture_interactions(
        [params1, params2], concentrations, interaction_type="additive"
    )

    assert len(responses) == 2
    assert all(responses > 0)


def test_dose_response_dataset():
    """Test dose-response dataset creation."""
    model = ConcentrationResponseModel()

    params_dict = {
        "hexanol": HillParameters(1.0, 0.1, 1.5),
        "ethyl butyrate": HillParameters(0.8, 0.2, 1.2),
    }

    dataset = model.create_dose_response_dataset(
        params_dict, n_concentrations=10, add_noise=False
    )

    assert len(dataset) == 20  # 2 odorants * 10 concentrations
    assert "odorant" in dataset.columns
    assert "concentration" in dataset.columns
    assert "response" in dataset.columns
