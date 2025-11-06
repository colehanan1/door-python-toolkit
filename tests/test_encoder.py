"""Unit tests for DoOR encoder."""

import pytest
import numpy as np
from pathlib import Path

from door_toolkit import DoOREncoder


# Skip tests if cache doesn't exist
pytestmark = pytest.mark.skipif(
    not Path("door_cache").exists(),
    reason="DoOR cache not found. Run DoORExtractor first."
)


class TestDoOREncoder:
    """Test DoOREncoder functionality."""
    
    @pytest.fixture
    def encoder(self):
        """Create encoder instance."""
        return DoOREncoder("door_cache", use_torch=False)
    
    def test_encoder_init(self, encoder):
        """Test encoder initialization."""
        assert encoder.n_channels == 78
        assert len(encoder.receptor_names) == 78
        assert len(encoder.odorant_names) > 0
    
    def test_encode_single(self, encoder):
        """Test encoding single odorant."""
        # Use first available odorant
        odor = encoder.odorant_names[0]
        pn = encoder.encode(odor)
        
        assert pn.shape == (78,)
        assert pn.dtype == np.float32
        assert np.all((pn >= 0) & (pn <= 1))
    
    def test_encode_batch(self, encoder):
        """Test batch encoding."""
        odors = encoder.odorant_names[:3]
        pn_batch = encoder.batch_encode(odors)
        
        assert pn_batch.shape == (3, 78)
        assert pn_batch.dtype == np.float32
    
    def test_encode_not_found(self, encoder):
        """Test encoding non-existent odorant."""
        with pytest.raises(KeyError):
            encoder.encode("nonexistent_odor_xyz")
    
    def test_list_odorants(self, encoder):
        """Test listing odorants."""
        all_odors = encoder.list_available_odorants()
        assert len(all_odors) > 0
        
        # Test pattern filtering
        acetates = encoder.list_available_odorants(pattern="acetate")
        assert all("acetate" in o.lower() for o in acetates)
    
    def test_get_coverage(self, encoder):
        """Test receptor coverage stats."""
        odor = encoder.odorant_names[0]
        stats = encoder.get_receptor_coverage(odor)
        
        assert "n_tested" in stats
        assert "n_active" in stats
        assert "max_response" in stats
        assert stats["n_tested"] >= 0
        assert stats["n_active"] >= 0
    
    def test_get_metadata(self, encoder):
        """Test metadata retrieval."""
        odor = encoder.odorant_names[0]
        meta = encoder.get_odor_metadata(odor)
        
        assert isinstance(meta, dict)
        assert "Name" in meta
    
    def test_fill_missing(self, encoder):
        """Test fill_missing parameter."""
        odor = encoder.odorant_names[0]
        
        pn_zero = encoder.encode(odor, fill_missing=0.0)
        pn_half = encoder.encode(odor, fill_missing=0.5)
        
        # Should differ where original has NaN
        assert not np.allclose(pn_zero, pn_half)


@pytest.mark.skipif(
    not Path("door_cache").exists(),
    reason="DoOR cache not found"
)
def test_torch_integration():
    """Test PyTorch tensor output."""
    try:
        import torch
        encoder = DoOREncoder("door_cache", use_torch=True)
        
        odor = encoder.odorant_names[0]
        pn = encoder.encode(odor)
        
        assert isinstance(pn, torch.Tensor)
        assert pn.dtype == torch.float32
        assert pn.shape == (78,)
        
    except ImportError:
        pytest.skip("PyTorch not installed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
