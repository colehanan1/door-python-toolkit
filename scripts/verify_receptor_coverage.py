#!/usr/bin/env python3
"""
Verify receptor mapping coverage after updates.
"""

from pathlib import Path
import sys

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.integration.mapping_accounting import (
    compute_mapping_stats,
    format_mapping_summary
)
from door_toolkit.integration.receptor_identifier import normalize_receptor_identifier


def verify_coverage() -> bool:
    door_df = pd.read_parquet("door_cache/response_matrix_norm.parquet")

    if door_df.shape[0] > door_df.shape[1]:
        door_df = door_df.T

    door_receptors = sorted(door_df.index.tolist())

    mapping_file = Path("data/mappings/door_to_flywire_mapping.csv")
    mapping_df = pd.read_csv(mapping_file)
    # Count only unambiguous, valid ORN_* mappings as "mapped".
    usable = mapping_df.copy()
    usable["door_name"] = usable["door_name"].astype(str)
    usable["flywire_glomerulus"] = usable["flywire_glomerulus"].astype(str)
    if "is_ambiguous" in usable.columns:
        ambiguous = usable["is_ambiguous"].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y"})
        usable = usable[~ambiguous]
    usable = usable[usable["flywire_glomerulus"].str.strip().ne("")]
    usable = usable[usable["flywire_glomerulus"].str.strip().str.startswith("ORN_")]
    mapped_receptors = sorted(usable["door_name"].dropna().unique().tolist())

    # Compute mapping statistics to prevent receptor/glomerulus count confusion
    receptor_to_glom = dict(zip(usable["door_name"], usable["flywire_glomerulus"]))

    # Pre-normalization (exact string match)
    unmapped_receptors_pre = sorted(set(door_receptors) - set(mapped_receptors))

    # Post-normalization (canonical matching key)
    door_key_to_name = {}
    for r in door_receptors:
        door_key_to_name.setdefault(normalize_receptor_identifier(r), r)

    mapped_keys = {
        normalize_receptor_identifier(r)
        for r in mapped_receptors
        if normalize_receptor_identifier(r)
    }
    door_keys = set(door_key_to_name.keys())
    unmapped_keys_post = sorted(door_keys - mapped_keys)
    unmapped_receptors = [door_key_to_name[k] for k in unmapped_keys_post if k in door_key_to_name]

    mapping_stats = compute_mapping_stats(
        receptor_to_glom,
        input_receptors=door_receptors,
        unmapped_receptors=unmapped_receptors,
        note="DoOR → FlyWire coverage verification",
        adult_only=False
    )

    coverage_pre = len(mapped_receptors) / len(door_receptors) * 100
    mapped_post = len(door_receptors) - len(unmapped_receptors)
    coverage_post = mapped_post / len(door_receptors) * 100
    missing = unmapped_receptors

    print("=" * 70)
    print("RECEPTOR → GLOMERULUS MAPPING COVERAGE VERIFICATION")
    print("=" * 70)
    print(f"DoOR receptors (total): {len(door_receptors)}")
    print(f"Mapped receptors (mapping CSV rows): {mapping_stats['n_receptors_mapped']}")
    print(f"Unique glomeruli: {mapping_stats['n_unique_glomeruli_from_mapped_receptors']}")
    print(f"Collisions (many-to-one): {mapping_stats['collision_count']} glomeruli")
    print(f"Coverage (pre-normalization):  {coverage_pre:.1f}%")
    print(f"Coverage (post-normalization): {coverage_post:.1f}%")
    if unmapped_receptors_pre != unmapped_receptors:
        rescued = sorted(set(unmapped_receptors_pre) - set(unmapped_receptors))
        if rescued:
            print(f"Rescued by normalization ({len(rescued)}): {rescued}")
    print()

    if mapping_stats['collision_count'] > 0:
        print(f"Many-to-one collapses (showing first 5):")
        for collision_line in mapping_stats['collision_summary'][:5]:
            print(f"  {collision_line}")
        if mapping_stats['collision_count'] > 5:
            print(f"  ... and {mapping_stats['collision_count'] - 5} more")
        print()

    if coverage_post >= 80:
        print("OK: Coverage >= 80% target.")
    elif coverage_post >= 70:
        print("OK: Coverage >= 70%.")
    else:
        print("WARNING: Coverage below 70% - consider adding more mappings.")

    print()
    print(f"Missing receptors ({len(missing)}):")
    for rec in missing:
        if rec in door_df.index:
            benzaldehyde_response = (
                door_df.loc[rec, "benzaldehyde"]
                if "benzaldehyde" in door_df.columns
                else 0
            )
            priority = "HIGH" if benzaldehyde_response > 0.1 else "LOW"
            print(f"  - {rec:15s} (priority: {priority})")
        else:
            print(f"  - {rec:15s}")

    print("=" * 70)

    return coverage_post >= 80


if __name__ == "__main__":
    success = verify_coverage()
    raise SystemExit(0 if success else 1)
