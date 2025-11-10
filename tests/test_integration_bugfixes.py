"""
Unit tests for DoOR integration bug fixes.

Tests for:
1. Bug Fix: LTK calculation with object dtype arrays
2. Bug Fix: Odorant name resolution with InChIKey mapping
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from door_toolkit.integration.door_utils import calculate_lifetime_kurtosis
from door_toolkit.integration.odorant_mapping import OdorantMapper


class TestLTKCalculationBugFix:
    """Tests for Bug Fix 1: LTK calculation with mixed data types."""

    def test_ltk_with_numeric_data(self):
        """Test LTK calculation with pure numeric data."""
        # Create numeric response vector
        responses = np.array([0.1, 0.2, 0.9, 0.1, 0.05, 0.15, 0.8, 0.3])

        ltk = calculate_lifetime_kurtosis(responses)

        # Should return a valid numeric value
        assert isinstance(ltk, (float, np.floating))
        assert not np.isnan(ltk)

    def test_ltk_with_mixed_data_strings(self):
        """Test LTK calculation with mixed numeric and string data (bug scenario)."""
        # Create mixed data like DoOR matrix (strings like 'SFR')
        responses = np.array(["SFR", 0.1, 0.2, 0.9, 0.1, 0.05], dtype=object)

        # Should handle gracefully without TypeError
        ltk = calculate_lifetime_kurtosis(responses)

        # Should return a valid numeric value (strings are coerced to NaN and removed)
        assert isinstance(ltk, (float, np.floating))

    def test_ltk_with_pandas_series(self):
        """Test LTK calculation with pandas Series (common DoOR usage)."""
        responses = pd.Series([0.1, 0.2, 0.9, 0.1, 0.05, "SFR", None, 0.8])

        ltk = calculate_lifetime_kurtosis(responses)

        # Should handle pandas Series correctly
        assert isinstance(ltk, (float, np.floating))
        assert not np.isnan(ltk)

    def test_ltk_with_insufficient_data(self):
        """Test LTK with insufficient data points."""
        # Only 3 data points (need at least 4 for kurtosis)
        responses = np.array([0.1, 0.2, 0.3])

        ltk = calculate_lifetime_kurtosis(responses)

        # Should return NaN for insufficient data
        assert np.isnan(ltk)

    def test_ltk_with_all_nan(self):
        """Test LTK with all NaN values."""
        responses = np.array([np.nan, np.nan, np.nan, np.nan])

        ltk = calculate_lifetime_kurtosis(responses)

        # Should return NaN
        assert np.isnan(ltk)

    def test_ltk_with_identical_values(self):
        """Test LTK with all identical values (zero variance)."""
        responses = np.array([0.5, 0.5, 0.5, 0.5, 0.5])

        ltk = calculate_lifetime_kurtosis(responses)

        # Should return NaN (division by zero in denominator)
        assert np.isnan(ltk)

    def test_ltk_specialist_vs_generalist(self):
        """Test that LTK distinguishes specialists from generalists."""
        # Specialist: one strong response, rest weak
        specialist = np.array([0.95, 0.05, 0.02, 0.01, 0.03, 0.04, 0.02, 0.01])

        # Generalist: many moderate responses
        generalist = np.array([0.5, 0.45, 0.6, 0.55, 0.5, 0.48, 0.52, 0.51])

        ltk_specialist = calculate_lifetime_kurtosis(specialist)
        ltk_generalist = calculate_lifetime_kurtosis(generalist)

        # Specialist should have higher LTK
        assert ltk_specialist > ltk_generalist


class TestOdorantMappingBugFix:
    """Tests for Bug Fix 2: Odorant name to InChIKey mapping."""

    def test_mapper_initialization(self):
        """Test OdorantMapper initialization."""
        mapper = OdorantMapper()

        assert mapper is not None
        assert len(mapper.name_to_inchikey) > 0
        assert len(mapper.inchikey_to_name) > 0

    def test_get_inchikey_from_common_name(self):
        """Test resolving common name to InChIKey."""
        mapper = OdorantMapper()

        # Test exact match
        inchikey = mapper.get_inchikey("acetic acid")
        assert inchikey == "QTBSBXVTEAMEQO-UHFFFAOYSA-N"

        # Test case insensitivity
        inchikey = mapper.get_inchikey("ACETIC ACID")
        assert inchikey == "QTBSBXVTEAMEQO-UHFFFAOYSA-N"

        # Test CO2
        inchikey = mapper.get_inchikey("CO2")
        assert inchikey == "CURLTUGMZLYLDI-UHFFFAOYSA-N"

    def test_get_inchikey_with_synonym(self):
        """Test resolving synonym to InChIKey."""
        mapper = OdorantMapper()

        # Test synonym mapping
        inchikey = mapper.get_inchikey("carbon dioxide")
        co2_inchikey = mapper.get_inchikey("CO2")

        assert inchikey == co2_inchikey

    def test_get_common_name_from_inchikey(self):
        """Test reverse lookup: InChIKey to common name."""
        mapper = OdorantMapper()

        name = mapper.get_common_name("QTBSBXVTEAMEQO-UHFFFAOYSA-N")
        assert name.lower() == "acetic acid"

    def test_is_inchikey_format(self):
        """Test InChIKey format detection."""
        mapper = OdorantMapper()

        # Valid InChIKey
        assert mapper.is_inchikey("QTBSBXVTEAMEQO-UHFFFAOYSA-N") is True

        # Common name
        assert mapper.is_inchikey("acetic acid") is False

        # Invalid format
        assert mapper.is_inchikey("TOOSHORT") is False

    def test_search_by_name(self):
        """Test fuzzy search functionality."""
        mapper = OdorantMapper()

        # Search for 'acet' should find acetic acid, acetate compounds, etc.
        results = mapper.search_by_name("acet")

        assert len(results) > 0
        # Should find acetic acid
        names = [r[0] for r in results]
        assert any("acetic" in name for name in names)

    def test_list_all_odorants(self):
        """Test listing all available odorants."""
        mapper = OdorantMapper()

        odorants = mapper.list_all_odorants()

        assert len(odorants) >= 50  # Should have at least 50 odorants
        assert "Acetic Acid" in odorants or "acetic acid" in [o.lower() for o in odorants]
        assert "CO2" in odorants or "co2" in [o.lower() for o in odorants]

    def test_get_inchikey_not_found(self):
        """Test behavior when odorant not found."""
        mapper = OdorantMapper()

        # Non-existent odorant
        inchikey = mapper.get_inchikey("nonexistent_chemical_xyz123")

        assert inchikey is None


class TestIntegrationWithDoORMatrix:
    """Integration tests with mock DoOR matrix."""

    def test_ltk_on_doorflike_matrix(self):
        """Test LTK calculation on DoOR-like matrix with mixed data."""
        # Create mock DoOR matrix with mixed types
        odorants = [f"odorant_{i}" for i in range(20)]
        receptors = ["Or47b", "Or82a", "Or69a"]

        # Create data with some 'SFR' values
        data = []
        for receptor in receptors:
            row = [np.random.random() if np.random.random() > 0.1 else "SFR" for _ in odorants]
            data.append(row)

        door_matrix = pd.DataFrame(data, index=receptors, columns=odorants)

        # Calculate LTK for each receptor
        ltk_values = {}
        for receptor in receptors:
            responses = door_matrix.loc[receptor]
            ltk = calculate_lifetime_kurtosis(responses)
            ltk_values[receptor] = ltk

        # Should complete without errors
        assert len(ltk_values) == len(receptors)

        # At least some should be valid (not all NaN)
        valid_count = sum(1 for v in ltk_values.values() if not np.isnan(v))
        assert valid_count > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
