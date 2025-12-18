"""
Receptor-to-Glomerulus Mapping Accounting
==========================================

This module provides explicit accounting for receptor→glomerulus mappings to prevent
confusion between:
- **Receptor counts**: Total number of receptor genes (e.g., Or7a, Ir31a, ...)
- **Unique glomerulus counts**: Number of distinct glomeruli/channels they map to

Scientific context: In FlyWire, multiple receptors can map to the same glomerulus
(many-to-one collapse). Example: OR82A and OR94A might both map to VA6, so
"+2 receptors, +1 unique glomerulus".

This module ensures every analysis explicitly reports BOTH counts and collision details.

Key Functions:
- compute_mapping_stats(): Comprehensive statistics about a receptor→glomerulus mapping
- build_glomerulus_to_receptors(): Reverse mapping to detect collisions
- summarize_collisions(): Find glomeruli with multiple receptors
- is_larval_receptor(): Adult/larval filtering logic
- write_mapping_stats_json(): Persist accounting to JSON artifact

Author: Claude Code (Senior Scientific Software Engineer)
Date: 2025-12-17
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from door_toolkit.integration.receptor_identifier import normalize_receptor_identifier

logger = logging.getLogger(__name__)


# Larval-only receptors (excluded in adult-only mode)
# Source: DoOR 2.0 mapping metadata (DoOR.mappings in DoOR.data v2.0.0) + literature.
# Operational definition here: receptors flagged as larva=True and adult=False in DoOR.mappings.
# These receptors are expressed in Drosophila larvae but not in the adult antennal lobe.
LARVAL_ONLY_RECEPTORS = {
    # DoOR responding units flagged adult=False, larva=True (DoOR.data v2.0.0)
    'OR1A',
    'OR22C',
    'OR24A',
    'OR30A',
    'OR45A',
    'OR45B',
    'OR59A',
    'OR63A',
    'OR74A',
    'OR83A',
    'OR85C',
    'OR94A',
    'OR94B',
}

# Note: This is a conservative list. Update data/larval_receptors.txt for customization


def is_larval_receptor(receptor: str, custom_larval_list: Optional[Set[str]] = None) -> bool:
    """
    Determine if a receptor is larval-only (should be excluded in adult-only mode).

    Args:
        receptor: Receptor name (e.g., 'OR33B', 'Ir31a')
        custom_larval_list: Optional custom set of larval receptors (overrides default)

    Returns:
        True if receptor is larval-only, False if adult or unknown

    Example:
        >>> is_larval_receptor('OR33B')
        True
        >>> is_larval_receptor('OR7A')
        False
    """
    receptor_upper = normalize_receptor_identifier(receptor)

    if custom_larval_list is not None:
        return receptor_upper in {normalize_receptor_identifier(r) for r in custom_larval_list}

    return receptor_upper in LARVAL_ONLY_RECEPTORS


def build_glomerulus_to_receptors(receptor_to_glomerulus: Dict[str, str]) -> Dict[str, List[str]]:
    """
    Build reverse mapping: glomerulus → list of receptors.

    This reveals many-to-one collapses where multiple receptors map to the same glomerulus.

    Args:
        receptor_to_glomerulus: Dict mapping receptor → glomerulus (e.g., {'OR7A': 'DL5'})

    Returns:
        Dict mapping glomerulus → sorted list of receptors

    Example:
        >>> mapping = {'OR82A': 'VA6', 'OR94A': 'VA6', 'OR7A': 'DL5'}
        >>> build_glomerulus_to_receptors(mapping)
        {'VA6': ['OR82A', 'OR94A'], 'DL5': ['OR7A']}
    """
    glom_to_receptors: Dict[str, List[str]] = {}

    for receptor, glomerulus in sorted(receptor_to_glomerulus.items()):
        if glomerulus not in glom_to_receptors:
            glom_to_receptors[glomerulus] = []
        glom_to_receptors[glomerulus].append(receptor)

    # Sort receptor lists for deterministic output
    for glom in glom_to_receptors:
        glom_to_receptors[glom] = sorted(glom_to_receptors[glom])

    return glom_to_receptors


def summarize_collisions(
    glom_to_receptors: Dict[str, List[str]],
    min_size: int = 2
) -> Dict[str, List[str]]:
    """
    Find glomeruli with multiple receptors (many-to-one collisions).

    Args:
        glom_to_receptors: Dict from build_glomerulus_to_receptors()
        min_size: Minimum number of receptors to count as collision (default 2)

    Returns:
        Dict of glomerulus → list of receptors, filtered to collisions only

    Example:
        >>> glom_map = {'VA6': ['OR82A', 'OR94A'], 'DL5': ['OR7A']}
        >>> summarize_collisions(glom_map)
        {'VA6': ['OR82A', 'OR94A']}
    """
    return {
        glom: receptors
        for glom, receptors in sorted(glom_to_receptors.items())
        if len(receptors) >= min_size
    }


def compute_mapping_stats(
    receptor_to_glomerulus: Dict[str, str],
    *,
    input_receptors: Optional[List[str]] = None,
    excluded_receptors: Optional[List[str]] = None,
    unmapped_receptors: Optional[List[str]] = None,
    note: Optional[str] = None,
    adult_only: bool = True,
    custom_larval_list: Optional[Set[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics about a receptor→glomerulus mapping.

    **CRITICAL**: Returns BOTH receptor counts AND unique glomerulus counts to prevent
    confusion. The many-to-one mapping means these can differ!

    Args:
        receptor_to_glomerulus: Dict mapping receptor → glomerulus
        input_receptors: Optional list of all candidate receptors (before mapping/filtering)
        excluded_receptors: Optional list of receptors excluded from mapping
        unmapped_receptors: Optional list of receptors that failed to map
        note: Optional descriptive note about this mapping
        adult_only: If True, exclude larval-only receptors
        custom_larval_list: Optional custom set of larval receptors

    Returns:
        Dict containing:
        - n_receptors_total: Total number of receptors in mapping
        - n_receptors_mapped: Number successfully mapped to glomeruli
        - n_receptors_unmapped: Number that failed to map
        - n_receptors_excluded_larval: Number excluded as larval-only (if adult_only=True)
        - n_unique_glomeruli_total_in_mapping: Unique glomeruli across all receptors
        - n_unique_glomeruli_from_mapped_receptors: Unique glomeruli from mapped receptors
        - glomerulus_to_receptors: Dict of glomerulus → list of receptors
        - collisions: Dict of glomeruli with ≥2 receptors (many-to-one)
        - collision_count: Number of glomeruli with collisions
        - collision_summary: Human-readable summary (e.g., "VA6: OR82A, OR94A")
        - note: User-provided note

    Example:
        >>> mapping = {'OR82A': 'VA6', 'OR94A': 'VA6', 'OR7A': 'DL5'}
        >>> stats = compute_mapping_stats(mapping, note="Test mapping")
        >>> print(stats['n_receptors_total'])  # 3
        >>> print(stats['n_unique_glomeruli_from_mapped_receptors'])  # 2 (VA6, DL5)
        >>> print(stats['collision_summary'])  # ['VA6: OR82A, OR94A']
    """
    # Apply adult-only filtering if requested
    filtered_mapping = receptor_to_glomerulus.copy()
    excluded_larval = []

    if adult_only:
        excluded_larval = [
            r for r in filtered_mapping.keys()
            if is_larval_receptor(r, custom_larval_list)
        ]
        for receptor in excluded_larval:
            del filtered_mapping[receptor]

        if excluded_larval:
            logger.info(
                f"Adult-only mode: Excluded {len(excluded_larval)} larval receptors: "
                f"{sorted(excluded_larval)}"
            )

    # Build reverse mapping (glomerulus → receptors)
    glom_to_receptors = build_glomerulus_to_receptors(filtered_mapping)

    # Detect collisions (many-to-one)
    collisions = summarize_collisions(glom_to_receptors)

    # Compute collision summary strings
    collision_summary = [
        f"{glom}: {', '.join(receptors)}"
        for glom, receptors in sorted(collisions.items())
    ]

    # Compute counts
    n_receptors_mapped = len(filtered_mapping)
    n_unique_glomeruli = len(glom_to_receptors)

    # Handle unmapped/excluded counts
    unmapped_list = unmapped_receptors or []
    excluded_list = excluded_receptors or []

    n_receptors_unmapped = len(unmapped_list)
    n_receptors_excluded = len(excluded_list) + len(excluded_larval)

    # Infer total from input_receptors if provided
    if input_receptors is not None:
        n_receptors_total = len(input_receptors)
    else:
        n_receptors_total = n_receptors_mapped + n_receptors_unmapped

    # Build comprehensive stats dict
    stats = {
        # Receptor counts
        'n_receptors_total': n_receptors_total,
        'n_receptors_mapped': n_receptors_mapped,
        'n_receptors_unmapped': n_receptors_unmapped,
        'n_receptors_excluded_larval': len(excluded_larval),
        'n_receptors_excluded_other': len(excluded_list),
        'n_receptors_excluded_total': n_receptors_excluded,

        # Glomerulus counts (DISTINCT from receptor counts!)
        'n_unique_glomeruli_total_in_mapping': n_unique_glomeruli,
        'n_unique_glomeruli_from_mapped_receptors': n_unique_glomeruli,  # Same for now

        # Many-to-one collision details
        'glomerulus_to_receptors': glom_to_receptors,
        'collisions': collisions,
        'collision_count': len(collisions),
        'collision_summary': collision_summary,

        # Lists for audit
        'receptors_mapped': sorted(filtered_mapping.keys()),
        'receptors_unmapped': sorted(unmapped_list),
        'receptors_excluded_larval': sorted(excluded_larval),
        'receptors_excluded_other': sorted(excluded_list),

        # Metadata
        'note': note,
        'adult_only_mode': adult_only,
    }

    return stats


def write_mapping_stats_json(output_path: Path, stats: Dict[str, Any]) -> None:
    """
    Write mapping statistics to JSON file for reproducibility.

    Args:
        output_path: Path to JSON file
        stats: Dict from compute_mapping_stats()

    Example:
        >>> stats = compute_mapping_stats(mapping)
        >>> write_mapping_stats_json(Path('mapping_stats.json'), stats)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2, sort_keys=True)

    logger.info(f"Wrote mapping statistics to {output_path}")


def log_mapping_stats(stats: Dict[str, Any], level: int = logging.INFO) -> None:
    """
    Log mapping statistics in human-readable format.

    **CRITICAL**: Clearly distinguishes receptor counts from unique glomerulus counts.

    Args:
        stats: Dict from compute_mapping_stats()
        level: Logging level (default INFO)
    """
    logger.log(level, "=" * 70)
    logger.log(level, "RECEPTOR → GLOMERULUS MAPPING STATISTICS")
    logger.log(level, "=" * 70)

    if stats.get('note'):
        logger.log(level, f"Note: {stats['note']}")
        logger.log(level, "")

    # RECEPTOR COUNTS (clearly labeled)
    logger.log(level, "RECEPTOR COUNTS:")
    logger.log(level, f"  Total receptors (candidates): {stats['n_receptors_total']}")
    logger.log(level, f"  Receptors mapped to glomeruli: {stats['n_receptors_mapped']}")
    logger.log(level, f"  Receptors unmapped: {stats['n_receptors_unmapped']}")

    if stats['adult_only_mode']:
        logger.log(level, f"  Receptors excluded (larval-only): {stats['n_receptors_excluded_larval']}")
    if stats['n_receptors_excluded_other'] > 0:
        logger.log(level, f"  Receptors excluded (other): {stats['n_receptors_excluded_other']}")

    logger.log(level, "")

    # UNIQUE GLOMERULUS COUNTS (clearly labeled, DISTINCT from receptors!)
    logger.log(level, "UNIQUE GLOMERULUS COUNTS (may differ from receptor counts!):")
    logger.log(level, f"  Unique glomeruli from mapped receptors: {stats['n_unique_glomeruli_from_mapped_receptors']}")
    logger.log(level, "")

    # MANY-TO-ONE COLLISIONS
    if stats['collision_count'] > 0:
        logger.log(level, f"MANY-TO-ONE COLLAPSES ({stats['collision_count']} glomeruli with ≥2 receptors):")
        for summary_line in stats['collision_summary']:
            logger.log(level, f"  {summary_line}")
        logger.log(level, "")
    else:
        logger.log(level, "No many-to-one collapses detected (1:1 mapping)")
        logger.log(level, "")

    # SUMMARY
    logger.log(level, "SUMMARY:")
    logger.log(level,
        f"  {stats['n_receptors_mapped']} receptors map to "
        f"{stats['n_unique_glomeruli_from_mapped_receptors']} unique glomeruli"
    )
    if stats['collision_count'] > 0:
        logger.log(level,
            f"  {stats['collision_count']} glomeruli receive input from multiple receptors"
        )

    logger.log(level, "=" * 70)


def format_mapping_summary(stats: Dict[str, Any]) -> str:
    """
    Format mapping statistics as a compact summary string.

    Args:
        stats: Dict from compute_mapping_stats()

    Returns:
        Compact summary string (e.g., "44 receptors → 44 unique glomeruli (no collisions)")

    Example:
        >>> stats = compute_mapping_stats(mapping)
        >>> print(format_mapping_summary(stats))
        "3 receptors → 2 unique glomeruli (1 collision)"
    """
    n_rec = stats['n_receptors_mapped']
    n_glom = stats['n_unique_glomeruli_from_mapped_receptors']
    n_coll = stats['collision_count']

    collision_str = f"{n_coll} collision{'s' if n_coll != 1 else ''}" if n_coll > 0 else "no collisions"

    return f"{n_rec} receptors → {n_glom} unique glomeruli ({collision_str})"
