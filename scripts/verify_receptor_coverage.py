#!/usr/bin/env python3
"""
Verify receptor mapping coverage after updates.
"""

from pathlib import Path

import pandas as pd


def verify_coverage() -> bool:
    door_df = pd.read_parquet("door_cache/response_matrix_norm.parquet")

    if door_df.shape[0] > door_df.shape[1]:
        door_df = door_df.T

    door_receptors = sorted(door_df.index.tolist())

    mapping_file = Path("data/mappings/door_to_flywire_mapping_complete.csv")
    mapping_df = pd.read_csv(mapping_file)
    mapped_receptors = sorted(mapping_df["door_name"].unique().tolist())

    coverage = len(mapped_receptors) / len(door_receptors) * 100
    missing = sorted(set(door_receptors) - set(mapped_receptors))

    print("=" * 70)
    print("RECEPTOR MAPPING COVERAGE VERIFICATION")
    print("=" * 70)
    print(f"DoOR receptors: {len(door_receptors)}")
    print(f"Mapped receptors: {len(mapped_receptors)}")
    print(f"Coverage: {coverage:.1f}%")
    print()

    if coverage >= 80:
        print("OK: Coverage >= 80% target.")
    elif coverage >= 70:
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

    return coverage >= 80


if __name__ == "__main__":
    success = verify_coverage()
    raise SystemExit(0 if success else 1)
