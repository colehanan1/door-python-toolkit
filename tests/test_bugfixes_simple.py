"""
Simple test script for DoOR integration bug fixes (without pytest).

Tests for:
1. Bug Fix: LTK calculation with object dtype arrays
2. Bug Fix: Odorant name resolution with InChIKey mapping
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from door_toolkit.integration.door_utils import calculate_lifetime_kurtosis
from door_toolkit.integration.odorant_mapping import OdorantMapper


def test_ltk_with_numeric_data():
    """Test LTK calculation with pure numeric data."""
    print("Test 1: LTK with numeric data...", end=" ")
    responses = np.array([0.1, 0.2, 0.9, 0.1, 0.05, 0.15, 0.8, 0.3])
    ltk = calculate_lifetime_kurtosis(responses)

    assert isinstance(ltk, (float, np.floating))
    assert not np.isnan(ltk)
    print(f"✓ PASS (LTK={ltk:.2f})")


def test_ltk_with_mixed_data():
    """Test LTK calculation with mixed numeric and string data (bug scenario)."""
    print("Test 2: LTK with mixed data (strings like 'SFR')...", end=" ")
    responses = np.array(["SFR", 0.1, 0.2, 0.9, 0.1, 0.05], dtype=object)

    try:
        ltk = calculate_lifetime_kurtosis(responses)
        assert isinstance(ltk, (float, np.floating))
        print(f"✓ PASS (LTK={ltk:.2f})")
    except TypeError as e:
        print(f"✗ FAIL: {e}")
        raise


def test_ltk_with_pandas_series():
    """Test LTK calculation with pandas Series."""
    print("Test 3: LTK with pandas Series (mixed types)...", end=" ")
    responses = pd.Series([0.1, 0.2, 0.9, 0.1, 0.05, "SFR", None, 0.8])
    ltk = calculate_lifetime_kurtosis(responses)

    assert isinstance(ltk, (float, np.floating))
    assert not np.isnan(ltk)
    print(f"✓ PASS (LTK={ltk:.2f})")


def test_odorant_mapper():
    """Test OdorantMapper initialization and basic functions."""
    print("Test 4: OdorantMapper initialization...", end=" ")
    mapper = OdorantMapper()

    assert mapper is not None
    assert len(mapper.name_to_inchikey) > 0
    print(f"✓ PASS ({len(mapper.name_to_inchikey)} odorants loaded)")


def test_get_inchikey():
    """Test resolving common name to InChIKey."""
    print("Test 5: Get InChIKey from common name...", end=" ")
    mapper = OdorantMapper()

    # Test exact match
    inchikey = mapper.get_inchikey("acetic acid")
    assert inchikey == "QTBSBXVTEAMEQO-UHFFFAOYSA-N"

    # Test case insensitivity
    inchikey2 = mapper.get_inchikey("ACETIC ACID")
    assert inchikey2 == "QTBSBXVTEAMEQO-UHFFFAOYSA-N"

    # Test CO2
    co2_key = mapper.get_inchikey("CO2")
    assert co2_key == "CURLTUGMZLYLDI-UHFFFAOYSA-N"

    print("✓ PASS")


def test_get_common_name():
    """Test reverse lookup: InChIKey to common name."""
    print("Test 6: Get common name from InChIKey...", end=" ")
    mapper = OdorantMapper()

    name = mapper.get_common_name("QTBSBXVTEAMEQO-UHFFFAOYSA-N")
    assert name.lower() == "acetic acid"

    print(f"✓ PASS ('{name}')")


def test_is_inchikey():
    """Test InChIKey format detection."""
    print("Test 7: InChIKey format detection...", end=" ")
    mapper = OdorantMapper()

    assert mapper.is_inchikey("QTBSBXVTEAMEQO-UHFFFAOYSA-N") is True
    assert mapper.is_inchikey("acetic acid") is False
    assert mapper.is_inchikey("TOOSHORT") is False

    print("✓ PASS")


def test_list_all_odorants():
    """Test listing all available odorants."""
    print("Test 8: List all odorants...", end=" ")
    mapper = OdorantMapper()

    odorants = mapper.list_all_odorants()
    assert len(odorants) >= 50

    print(f"✓ PASS ({len(odorants)} odorants)")


def main():
    """Run all tests."""
    print()
    print("=" * 70)
    print("DoOR Integration Bug Fix Tests")
    print("=" * 70)
    print()

    tests = [
        test_ltk_with_numeric_data,
        test_ltk_with_mixed_data,
        test_ltk_with_pandas_series,
        test_odorant_mapper,
        test_get_inchikey,
        test_get_common_name,
        test_is_inchikey,
        test_list_all_odorants,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  Error: {e}")
            failed += 1

    print()
    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
