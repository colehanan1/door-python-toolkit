"""
Training Receptor Set Validator

Validates consistency between training receptor set, connectivity matrices,
feature schema, and model configuration.

Decision → Evidence → Implementation
-----------------------------------
Decision: Fail fast if receptor ordering is inconsistent across artifacts.
Evidence: Silent ordering mismatches are a major source of bugs in neuroscience
         models (Botvinick et al., Trends Cogn Sci 2020). Better to crash early
         than train on misaligned data.
Implementation: Explicit validation at training startup. No silent fixes.
"""

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def load_training_receptor_set(path: str) -> Dict:
    """Load training receptor set artifact."""
    with open(path) as f:
        return json.load(f)


def validate_receptor_ordering(
    training_set_path: str,
    connectivity_metadata_path: str,
    feature_schema_path: Optional[str] = None,
    strict: bool = True,
) -> Dict[str, any]:
    """
    Validate receptor ordering consistency across artifacts.

    Args:
        training_set_path: Path to training_receptor_set.json
        connectivity_metadata_path: Path to connectivity_metadata.json
        feature_schema_path: Path to feature_metadata.json (optional)
        strict: If True, fail on any mismatch. If False, warn only.

    Returns:
        validation_report: Dict with validation results

    Raises:
        ValueError: If strict=True and validation fails
    """
    report = {
        'passed': True,
        'errors': [],
        'warnings': [],
        'n_training_receptors': 0,
        'n_connectivity_receptors': 0,
    }

    # Load training set
    try:
        training_set = load_training_receptor_set(training_set_path)
        report['n_training_receptors'] = training_set['n_receptors']
        training_receptors = training_set['receptors']
        receptor_indices = training_set.get('receptor_indices_in_connectivity', None)
    except Exception as e:
        report['passed'] = False
        report['errors'].append(f"Failed to load training set: {e}")
        if strict:
            raise ValueError(f"Training set load failed: {e}")
        return report

    # Load connectivity metadata
    try:
        with open(connectivity_metadata_path) as f:
            conn_meta = json.load(f)
        report['n_connectivity_receptors'] = conn_meta['n_receptors']
        connectivity_receptors = conn_meta.get('receptor_names', [])
    except Exception as e:
        report['passed'] = False
        report['errors'].append(f"Failed to load connectivity metadata: {e}")
        if strict:
            raise ValueError(f"Connectivity metadata load failed: {e}")
        return report

    # Validate receptor indices
    if receptor_indices is None:
        report['passed'] = False
        report['errors'].append(
            "training_receptor_set.json missing 'receptor_indices_in_connectivity'. "
            "Regenerate with: python scripts/build_training_receptor_set.py"
        )
        if strict:
            raise ValueError("Missing receptor index mapping")
        return report

    # Validate: training receptors are subset of connectivity receptors
    for i, receptor in enumerate(training_receptors):
        expected_idx = receptor_indices[i]
        if expected_idx >= len(connectivity_receptors):
            report['passed'] = False
            report['errors'].append(
                f"Receptor index {expected_idx} for '{receptor}' out of bounds "
                f"(connectivity has {len(connectivity_receptors)} receptors)"
            )
        else:
            connectivity_receptor = connectivity_receptors[expected_idx]
            if connectivity_receptor != receptor:
                report['passed'] = False
                report['errors'].append(
                    f"Receptor ordering mismatch at index {expected_idx}: "
                    f"training expects '{receptor}' but connectivity has '{connectivity_receptor}'"
                )

    # Check for larval contamination
    larval_excluded = training_set.get('excluded_larval', [])
    if len(larval_excluded) > 0:
        report['larval_excluded_count'] = len(larval_excluded)
        report['warnings'].append(
            f"{len(larval_excluded)} larval receptors excluded: {larval_excluded[:5]}"
            f"{'...' if len(larval_excluded) > 5 else ''}"
        )

        # Check that none made it into training set
        larval_in_training = set(training_receptors) & set(larval_excluded)
        if larval_in_training:
            report['passed'] = False
            report['errors'].append(
                f"CRITICAL: Larval receptors in training set: {larval_in_training}"
            )

    # Check for unmapped contamination
    unmapped_excluded = training_set.get('excluded_unmapped', [])
    if len(unmapped_excluded) > 0:
        unmapped_in_training = set(training_receptors) & set(unmapped_excluded)
        if unmapped_in_training:
            report['passed'] = False
            report['errors'].append(
                f"CRITICAL: Unmapped receptors in training set: {unmapped_in_training}"
            )

    # Validate feature schema if provided
    if feature_schema_path:
        try:
            with open(feature_schema_path) as f:
                schema = json.load(f)
            test_profile_indices = schema['feature_groups'].get('test_door_profile', [])
            if len(test_profile_indices) != report['n_connectivity_receptors']:
                report['warnings'].append(
                    f"Feature schema test profile has {len(test_profile_indices)} dimensions "
                    f"but connectivity has {report['n_connectivity_receptors']} receptors. "
                    f"Model will extract training subset using receptor_indices."
                )
        except Exception as e:
            report['warnings'].append(f"Could not validate feature schema: {e}")

    # Final validation
    if report['errors'] and strict:
        error_msg = "\n".join(report['errors'])
        raise ValueError(f"Receptor ordering validation FAILED:\n{error_msg}")

    return report


def extract_training_receptor_subset(
    full_connectivity_matrix: torch.Tensor,
    training_set_path: str,
    dim: int = 0,
) -> torch.Tensor:
    """
    Extract training receptor subset from full connectivity matrix.

    Args:
        full_connectivity_matrix: Full connectivity tensor (e.g., [78, 841])
        training_set_path: Path to training_receptor_set.json
        dim: Dimension along which to extract (0 for ORN dim)

    Returns:
        subset_matrix: Connectivity for training receptors only (e.g., [55, 841])
    """
    training_set = load_training_receptor_set(training_set_path)
    receptor_indices = training_set['receptor_indices_in_connectivity']

    if receptor_indices is None:
        raise ValueError(
            "training_receptor_set.json missing receptor_indices_in_connectivity"
        )

    # Extract subset
    if dim == 0:
        subset_matrix = full_connectivity_matrix[receptor_indices, :]
    elif dim == 1:
        subset_matrix = full_connectivity_matrix[:, receptor_indices]
    else:
        raise ValueError(f"Unsupported dim: {dim}")

    return subset_matrix


def print_validation_report(report: Dict) -> None:
    """Print validation report in human-readable format."""
    print("=" * 70)
    print("RECEPTOR ORDERING VALIDATION REPORT")
    print("=" * 70)
    print(f"Status: {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    print(f"Training receptors: {report['n_training_receptors']}")
    print(f"Connectivity receptors: {report['n_connectivity_receptors']}")
    print()

    if report['errors']:
        print("ERRORS:")
        for error in report['errors']:
            print(f"  ❌ {error}")
        print()

    if report['warnings']:
        print("WARNINGS:")
        for warning in report['warnings']:
            print(f"  ⚠️  {warning}")
        print()

    if 'larval_excluded_count' in report:
        print(f"Larval receptors excluded: {report['larval_excluded_count']}")

    print("=" * 70)
