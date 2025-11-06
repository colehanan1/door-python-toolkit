"""Unit tests for DoOR utilities."""

import pytest
from pathlib import Path
import pandas as pd

from door_toolkit.utils import (
    load_response_matrix,
    load_odor_metadata,
    list_odorants,
    get_receptor_info,
    validate_cache,
)
from tests.fixtures import mock_door_cache


class TestUtils:
    """Test utility functions."""

    def test_load_response_matrix_parquet(self, mock_door_cache):
        """Test loading response matrix in parquet format."""
        df = load_response_matrix(str(mock_door_cache), format="parquet")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (20, 10)

    def test_load_response_matrix_csv(self, mock_door_cache):
        """Test loading response matrix in CSV format."""
        df = load_response_matrix(str(mock_door_cache), format="csv")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (20, 10)

    def test_load_response_matrix_numpy(self, mock_door_cache):
        """Test loading response matrix from NumPy."""
        df = load_response_matrix(str(mock_door_cache), format="numpy")
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (20, 10)

    def test_load_odor_metadata(self, mock_door_cache):
        """Test loading odor metadata."""
        meta = load_odor_metadata(str(mock_door_cache))
        assert isinstance(meta, pd.DataFrame)
        assert len(meta) == 20
        assert "Name" in meta.columns
        assert "CAS" in meta.columns

    def test_list_odorants(self, mock_door_cache):
        """Test listing odorants."""
        odors = list_odorants(str(mock_door_cache))
        assert len(odors) == 20

        # Test pattern filtering
        acetates = list_odorants(str(mock_door_cache), pattern="acetate")
        assert all("acetate" in o.lower() for o in acetates)

    def test_get_receptor_info(self, mock_door_cache):
        """Test receptor coverage info."""
        info = get_receptor_info(str(mock_door_cache))
        assert isinstance(info, pd.DataFrame)
        assert len(info) == 10
        assert "receptor" in info.columns
        assert "n_odorants_tested" in info.columns
        assert "coverage_pct" in info.columns

    def test_validate_cache(self, mock_door_cache):
        """Test cache validation."""
        is_valid = validate_cache(str(mock_door_cache))
        assert is_valid is True

    def test_validate_cache_invalid(self):
        """Test validation with invalid cache."""
        is_valid = validate_cache("nonexistent_cache")
        assert is_valid is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
