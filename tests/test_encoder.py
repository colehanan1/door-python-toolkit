"""Unit tests for DoOR encoder."""

import pytest
import numpy as np
from pathlib import Path

from door_toolkit import DoOREncoder
from tests.fixtures import mock_door_cache, mock_encoder, mock_encoder_torch


class TestDoOREncoder:
    """Test DoOREncoder functionality."""
    
    def test_encoder_init(self, mock_encoder):
        """Test encoder initialization."""
        assert mock_encoder.n_channels == 10
        assert len(mock_encoder.receptor_names) == 10
        assert len(mock_encoder.odorant_names) == 20
    
    def test_encode_single(self, mock_encoder):
        """Test encoding single odorant."""
        # Use first available odorant
        odor = mock_encoder.odorant_names[0]
        pn = mock_encoder.encode(odor)

        assert pn.shape == (10,)
        assert pn.dtype == np.float32
        assert np.all((pn >= 0) | np.isnan(pn))  # Allow NaN for missing data
    
    def test_encode_batch(self, mock_encoder):
        """Test batch encoding."""
        odors = mock_encoder.odorant_names[:3]
        pn_batch = mock_encoder.batch_encode(odors)

        assert pn_batch.shape == (3, 10)
        assert pn_batch.dtype == np.float32
    
    def test_encode_not_found(self, mock_encoder):
        """Test encoding non-existent odorant."""
        with pytest.raises(KeyError):
            mock_encoder.encode("nonexistent_odor_xyz")

    def test_list_odorants(self, mock_encoder):
        """Test listing odorants."""
        all_odors = mock_encoder.list_available_odorants()
        assert len(all_odors) == 20

        # Test pattern filtering
        acetates = mock_encoder.list_available_odorants(pattern="acetate")
        assert all("acetate" in o.lower() for o in acetates)
        assert len(acetates) >= 2  # ethyl acetate, methyl acetate, isoamyl acetate

    def test_get_coverage(self, mock_encoder):
        """Test receptor coverage stats."""
        odor = mock_encoder.odorant_names[0]
        stats = mock_encoder.get_receptor_coverage(odor)

        assert "n_tested" in stats
        assert "n_active" in stats
        assert "max_response" in stats
        assert stats["n_tested"] >= 0
        assert stats["n_active"] >= 0

    def test_get_metadata(self, mock_encoder):
        """Test metadata retrieval."""
        odor = mock_encoder.odorant_names[0]
        meta = mock_encoder.get_odor_metadata(odor)

        assert isinstance(meta, dict)
        assert "Name" in meta

    def test_fill_missing(self, mock_encoder):
        """Test fill_missing parameter."""
        odor = mock_encoder.odorant_names[0]

        pn_zero = mock_encoder.encode(odor, fill_missing=0.0)
        pn_half = mock_encoder.encode(odor, fill_missing=0.5)

        # Should differ where original has NaN
        assert not np.allclose(pn_zero, pn_half, equal_nan=True)


def test_torch_integration(mock_door_cache):
    """Test PyTorch tensor output."""
    try:
        import torch
        encoder = DoOREncoder(str(mock_door_cache), use_torch=True)

        odor = encoder.odorant_names[0]
        pn = encoder.encode(odor)

        assert isinstance(pn, torch.Tensor)
        assert pn.dtype == torch.float32
        assert pn.shape == (10,)

    except ImportError:
        pytest.skip("PyTorch not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
