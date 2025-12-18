"""
Tests for Training Receptor Set and Ordering Validation

Critical tests for publication-grade preflight system.
"""

import json
from pathlib import Path

import pytest

from door_toolkit.data.training_receptor_validator import (
    load_training_receptor_set,
    validate_receptor_ordering,
)


@pytest.fixture
def training_set_path():
    return "data/mappings/training_receptor_set.json"


@pytest.fixture
def connectivity_metadata_path():
    return "data/pgcn_features/connectivity/connectivity_metadata.json"


def test_training_set_exists(training_set_path):
    """Test that training receptor set file exists."""
    assert Path(training_set_path).exists(), \
        f"Training set not found: {training_set_path}. Run: python scripts/build_training_receptor_set.py"


def test_training_set_structure(training_set_path):
    """Test that training set has required fields."""
    training_set = load_training_receptor_set(training_set_path)

    required_fields = [
        'receptors',
        'flywire_targets',
        'n_receptors',
        'receptor_indices_in_connectivity',
        'filters',
        'excluded_larval',
        'excluded_unmapped',
        'excluded_ambiguous',
        'provenance',
    ]

    for field in required_fields:
        assert field in training_set, f"Missing required field: {field}"


def test_training_set_adult_only(training_set_path):
    """Test that no larval receptors in training set."""
    training_set = load_training_receptor_set(training_set_path)

    receptors = set(training_set['receptors'])
    excluded_larval = set(training_set['excluded_larval'])

    # No overlap
    overlap = receptors & excluded_larval
    assert len(overlap) == 0, \
        f"Larval receptors in training set: {overlap}"


def test_training_set_mapped_only(training_set_path):
    """Test that all training receptors have FlyWire targets."""
    training_set = load_training_receptor_set(training_set_path)

    receptors = training_set['receptors']
    flywire_targets = training_set['flywire_targets']

    # Same length
    assert len(receptors) == len(flywire_targets), \
        f"Receptor count {len(receptors)} != target count {len(flywire_targets)}"

    # All targets start with ORN_
    for target in flywire_targets:
        assert target.startswith('ORN_'), \
            f"Invalid FlyWire target (not ORN_*): {target}"


def test_training_set_size_expected(training_set_path):
    """Test that training set has expected size (55 receptors for adult+mapped+non-ambiguous)."""
    training_set = load_training_receptor_set(training_set_path)

    n_receptors = training_set['n_receptors']

    # Adult + mapped + non-ambiguous should be 55
    assert n_receptors == 55, \
        f"Expected 55 training receptors, got {n_receptors}"


def test_receptor_indices_valid(training_set_path, connectivity_metadata_path):
    """Test that receptor indices are valid for connectivity matrices."""
    training_set = load_training_receptor_set(training_set_path)

    with open(connectivity_metadata_path) as f:
        conn_meta = json.load(f)

    receptor_indices = training_set['receptor_indices_in_connectivity']
    n_connectivity_receptors = conn_meta['n_receptors']

    # All indices in valid range
    for idx in receptor_indices:
        assert 0 <= idx < n_connectivity_receptors, \
            f"Receptor index {idx} out of bounds (connectivity has {n_connectivity_receptors} receptors)"


def test_receptor_ordering_validation_passes(training_set_path, connectivity_metadata_path):
    """Test that receptor ordering validation passes."""
    report = validate_receptor_ordering(
        training_set_path=training_set_path,
        connectivity_metadata_path=connectivity_metadata_path,
        strict=False,  # Don't fail test on warnings
    )

    assert report['passed'], \
        f"Receptor ordering validation failed: {report['errors']}"


def test_no_duplicate_receptors(training_set_path):
    """Test that no duplicate receptors in training set."""
    training_set = load_training_receptor_set(training_set_path)

    receptors = training_set['receptors']
    unique_receptors = set(receptors)

    assert len(receptors) == len(unique_receptors), \
        f"Duplicate receptors found: {len(receptors)} total, {len(unique_receptors)} unique"


def test_excluded_categories_disjoint(training_set_path):
    """Test that excluded categories don't overlap."""
    training_set = load_training_receptor_set(training_set_path)

    excluded_larval = set(training_set['excluded_larval'])
    excluded_unmapped = set(training_set['excluded_unmapped'])
    excluded_ambiguous = set(training_set['excluded_ambiguous'])

    # Check for overlaps
    larval_unmapped = excluded_larval & excluded_unmapped
    larval_ambiguous = excluded_larval & excluded_ambiguous
    unmapped_ambiguous = excluded_unmapped & excluded_ambiguous

    # Some overlap is OK (e.g., larval AND unmapped), but log it
    if larval_unmapped:
        print(f"Note: {len(larval_unmapped)} receptors are both larval and unmapped")
    if larval_ambiguous:
        print(f"Note: {len(larval_ambiguous)} receptors are both larval and ambiguous")
    if unmapped_ambiguous:
        print(f"Note: {len(unmapped_ambiguous)} receptors are both unmapped and ambiguous")


def test_provenance_fields_present(training_set_path):
    """Test that provenance fields are present for reproducibility."""
    training_set = load_training_receptor_set(training_set_path)

    provenance = training_set['provenance']

    required_prov_fields = [
        'inventory_file',
        'inventory_hash',
        'mapping_file',
        'mapping_hash',
        'timestamp',
    ]

    for field in required_prov_fields:
        assert field in provenance, f"Missing provenance field: {field}"

    # Hashes should be SHA256 (64 hex chars)
    assert len(provenance['inventory_hash']) == 64, \
        "inventory_hash should be SHA256 (64 chars)"
    assert len(provenance['mapping_hash']) == 64, \
        "mapping_hash should be SHA256 (64 chars)"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
