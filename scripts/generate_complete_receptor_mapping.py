#!/usr/bin/env python3
"""
Generate the authoritative DoOR → FlyWire ORN_<glomerulus> mapping artifact.

Output:
- data/mappings/door_to_flywire_mapping.csv
- data/mappings/door_to_flywire_mapping_stats.json

This script is intended to be deterministic and publication-auditable:
- Uses DoOR.mappings (DoOR.data v2.0.0) as the primary source where available.
- Applies curated sensillum reference and manual override tables with explicit provenance.
- Runs strict validations that raise on invalid targets, known mismatches, or conflicts.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.integration.door_to_flywire_mapping import (  # noqa: E402
    build_authoritative_door_to_flywire_mapping,
    default_paths,
    load_door_mappings_full,
    load_door_receptors_from_cache,
)
from door_toolkit.integration.mapping_accounting import (  # noqa: E402
    compute_mapping_stats,
    format_mapping_summary,
    write_mapping_stats_json,
)


def _load_optional_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    paths = default_paths()

    matrix_path = paths["door_cache_matrix"]
    if not matrix_path.exists():
        raise FileNotFoundError(f"DoOR cache matrix not found: {matrix_path}")

    door_units = load_door_receptors_from_cache(matrix_path)

    door_mappings_df = load_door_mappings_full()
    manual_overrides_df = _load_optional_csv(paths["manual_overrides"])
    sensillum_reference_df = _load_optional_csv(paths["sensillum_reference"])

    mapping_df = build_authoritative_door_to_flywire_mapping(
        door_units,
        door_mappings_df=door_mappings_df,
        manual_overrides_df=manual_overrides_df if not manual_overrides_df.empty else None,
        sensillum_reference_df=sensillum_reference_df if not sensillum_reference_df.empty else None,
    )

    output_csv = paths["mapping_output"]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(output_csv, index=False)

    # ------------------------------------------------------------------
    # Mapping statistics (unambiguous mappings only)
    # ------------------------------------------------------------------
    unambiguous = mapping_df[
        mapping_df["flywire_glomerulus"].astype(str).str.strip().ne("")
        & ~mapping_df["is_ambiguous"].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y"})
    ].copy()
    receptor_to_glom = dict(zip(unambiguous["door_name"].astype(str), unambiguous["flywire_glomerulus"].astype(str)))

    unmapped = sorted(set(door_units) - set(receptor_to_glom.keys()))

    mapping_stats = compute_mapping_stats(
        receptor_to_glom,
        input_receptors=list(door_units),
        unmapped_receptors=unmapped,
        note="Authoritative DoOR → FlyWire ORN_<glomerulus> mapping (unambiguous rows only)",
        adult_only=False,
    )

    stats_output = output_csv.parent / "door_to_flywire_mapping_stats.json"
    write_mapping_stats_json(stats_output, mapping_stats)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    ambiguous_units = sorted(
        {
            d
            for d, grp in mapping_df.groupby("door_name")
            if len(grp) > 1
            and grp["flywire_glomerulus"].astype(str).str.strip().ne("").any()
            and grp["is_ambiguous"].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y"}).all()
        }
    )

    print("=" * 70)
    print("AUTHORITATIVE DoOR → FlyWire MAPPING GENERATED")
    print("=" * 70)
    print(f"Output CSV: {output_csv}")
    print(f"Stats JSON: {stats_output}")
    print()
    print(f"DoOR responding units: {len(door_units)}")
    print(f"Unambiguous mapped units: {len(receptor_to_glom)}")
    print(f"Ambiguous units (excluded from unambiguous stats): {len(ambiguous_units)}")
    print(f"Unmapped units: {len(unmapped)}")
    print()
    print(f"  {format_mapping_summary(mapping_stats)}")
    if mapping_stats["collision_count"] > 0:
        print("  Many-to-one collapses (first 10):")
        for line in mapping_stats["collision_summary"][:10]:
            print(f"    - {line}")
    print()

    if ambiguous_units:
        print("Ambiguous DoOR units (multi-glomerulus):")
        for unit in ambiguous_units:
            targets = sorted(
                set(
                    mapping_df.loc[mapping_df["door_name"] == unit, "flywire_glomerulus"]
                    .astype(str)
                    .str.strip()
                    .tolist()
                )
            )
            targets = [t for t in targets if t]
            print(f"  - {unit}: {', '.join(targets)}")
        print()

    if unmapped:
        print("Unmapped DoOR units (no ORN_<glomerulus> target):")
        for unit in unmapped:
            print(f"  - {unit}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    main()

