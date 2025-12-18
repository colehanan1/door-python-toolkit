#!/usr/bin/env python3
"""
Generate unified receptor inventory CSV (single source of truth).

Output: data/mappings/receptor_inventory.csv

Inventory merges:
- DoOR receptor list (Münch & Galizia 2016; DoOR 2.0)
- Mapping + provenance (normalization-safe; DoOR.mappings preferred when available)
- Larval vs adult life-stage metadata (larval excluded from adult-only analyses)
- FlyWire connectivity coverage metrics (from connectivity_statistics.csv)
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Tuple

import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.integration.receptor_identifier import normalize_receptor_identifier
from door_toolkit.integration.receptor_inventory import (
    build_receptor_inventory_dataframe,
    get_larval_receptors_present,
    validate_inventory_schema,
)


DOOR_CACHE_MATRIX = Path("door_cache/response_matrix_norm.parquet")
MAPPING_FILE = Path("data/mappings/door_to_flywire_mapping.csv")
CONNECTIVITY_STATS_FILE = Path("data/pgcn_features/connectivity/connectivity_statistics.csv")
OUTPUT_FILE = Path("data/mappings/receptor_inventory.csv")

def _load_door_receptors(matrix_path: Path) -> Tuple[pd.DataFrame, list[str]]:
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"DoOR cache not found: {matrix_path}. Run DoORExtractor to create the cache."
        )

    door_df = pd.read_parquet(matrix_path)
    # Ensure receptors are in index (rows)
    if door_df.shape[0] > door_df.shape[1]:
        door_df = door_df.T

    receptors = sorted(door_df.index.tolist())
    return door_df, receptors


def main() -> None:
    print("=" * 70)
    print("GENERATING UNIFIED RECEPTOR INVENTORY")
    print("=" * 70)
    print()

    print(f"Loading DoOR receptors from {DOOR_CACHE_MATRIX}...")
    _, door_receptors = _load_door_receptors(DOOR_CACHE_MATRIX)
    print(f"✓ Found {len(door_receptors)} receptors in DoOR cache")

    mapping_df = None
    if MAPPING_FILE.exists():
        mapping_df = pd.read_csv(MAPPING_FILE)
        print(f"✓ Loaded mapping CSV: {MAPPING_FILE} ({len(mapping_df)} rows)")
    else:
        print(f"⚠️  Mapping CSV not found: {MAPPING_FILE} (continuing without it)")

    connectivity_df = None
    if CONNECTIVITY_STATS_FILE.exists():
        connectivity_df = pd.read_csv(CONNECTIVITY_STATS_FILE)
        print(f"✓ Loaded connectivity stats: {CONNECTIVITY_STATS_FILE} ({len(connectivity_df)} rows)")
    else:
        print(f"⚠️  Connectivity stats not found: {CONNECTIVITY_STATS_FILE} (metrics will be zero-filled)")

    # ---------------------------------------------------------------------
    # Validation: mapping coverage pre/post normalization (mapping CSV only)
    # ---------------------------------------------------------------------
    mapped_pre = 0
    mapped_post = 0
    rescued_by_normalization: list[str] = []

    if mapping_df is not None and not mapping_df.empty:
        usable_rows = mapping_df.copy()
        usable_rows["door_name"] = usable_rows["door_name"].astype(str)
        usable_rows["flywire_glomerulus"] = usable_rows["flywire_glomerulus"].astype(str)

        # Count only unambiguous, valid ORN_* targets as "mapped".
        if "is_ambiguous" in usable_rows.columns:
            ambiguous = usable_rows["is_ambiguous"].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y"})
            usable_rows = usable_rows[~ambiguous]
        usable_rows = usable_rows[usable_rows["flywire_glomerulus"].str.strip().ne("")]
        usable_rows = usable_rows[usable_rows["flywire_glomerulus"].str.strip().str.startswith("ORN_")]

        pre_keys = set(usable_rows["door_name"].astype(str))
        post_keys = {normalize_receptor_identifier(x) for x in usable_rows["door_name"].astype(str)}

        for r in door_receptors:
            if r in pre_keys:
                mapped_pre += 1
            if normalize_receptor_identifier(r) in post_keys:
                mapped_post += 1
                if r not in pre_keys:
                    rescued_by_normalization.append(r)

    # ---------------------------------------------------------------------
    # Build inventory
    # ---------------------------------------------------------------------
    inventory_df = build_receptor_inventory_dataframe(
        door_receptors,
        mapping_df=mapping_df,
        connectivity_df=connectivity_df,
        include_mapping_source_column=True,
    )
    validate_inventory_schema(inventory_df)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(OUTPUT_FILE, index=False)

    # ---------------------------------------------------------------------
    # Summary / validation prints
    # ---------------------------------------------------------------------
    total = len(inventory_df)
    mapped_final = int((inventory_df["is_mapped"] == "Yes").sum())
    unmapped_final = total - mapped_final

    larval_list = get_larval_receptors_present(door_receptors)
    larval_total = len(larval_list)
    larval_mapped = int(((inventory_df["life_stage"] == "Larval") & (inventory_df["is_mapped"] == "Yes")).sum())
    larval_unmapped = larval_total - larval_mapped

    adult_total = int((inventory_df["life_stage"] == "Adult").sum())
    adult_mapped = int(((inventory_df["life_stage"] == "Adult") & (inventory_df["is_mapped"] == "Yes")).sum())

    adult_ambiguous_list = inventory_df[
        (inventory_df["life_stage"] == "Adult") & (inventory_df["status"].astype(str).str.startswith("Ambiguous"))
    ]["receptor_name"].tolist()

    adult_unmapped_list = inventory_df[
        (inventory_df["life_stage"] == "Adult") & (inventory_df["status"].astype(str).str.startswith("Unmapped"))
    ]["receptor_name"].tolist()

    adult_ambiguous = len(adult_ambiguous_list)
    adult_unmapped = len(adult_unmapped_list)

    print()
    print("=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"Total DoOR receptors:                 {len(door_receptors)}")
    if mapping_df is not None:
        print(f"Mapped (pre-normalization, CSV):      {mapped_pre}")
        print(f"Mapped (post-normalization, CSV):     {mapped_post}")
        if rescued_by_normalization:
            print(f"Rescued by normalization:             {len(rescued_by_normalization)}")
            print(f"  {rescued_by_normalization}")
    print(f"Mapped (final inventory):             {mapped_final}")
    print()
    print(f"Larval receptors (count):             {larval_total}")
    print(f"Larval receptor names (len):          {len(larval_list)}")
    print(f"Larval receptors (excluded list):     {', '.join(larval_list)}")
    print()
    print(f"Adult unmapped receptors (post-norm): {adult_unmapped} / {adult_total}")
    if adult_ambiguous_list:
        print(f"Adult ambiguous receptors (excluded): {adult_ambiguous}")
        for i, name in enumerate(adult_ambiguous_list, 1):
            print(f"  A{i:02d}. {name}")
    if adult_unmapped_list:
        for i, name in enumerate(adult_unmapped_list, 1):
            print(f"  {i:2d}. {name}")
    print()

    print("=" * 70)
    print("RECEPTOR INVENTORY SUMMARY")
    print("=" * 70)
    print(f"Total receptors in DoOR:              {total}")
    print(f"Mapped to FlyWire glomerulus:         {mapped_final} ({100*mapped_final/total:.1f}%)")
    print(f"Unmapped (needs mapping):             {unmapped_final} ({100*unmapped_final/total:.1f}%)")
    print()
    print("Larval receptors (excluded from adult analysis):")
    print(f"  Total larval receptors:             {larval_total}")
    print(f"  Larval mapped to FlyWire:           {larval_mapped}")
    print(f"  Larval unmapped:                    {larval_unmapped}")
    print()
    print("Adult receptors (adult brain analyses):")
    print(f"  Total adult receptors:              {adult_total}")
    print(f"  Adult mapped to FlyWire:            {adult_mapped} ({100*adult_mapped/adult_total:.1f}%)")
    print(f"  Adult ambiguous (excluded):         {adult_ambiguous} ({100*adult_ambiguous/adult_total:.1f}%)")
    print(f"  Adult unmapped (needs work):        {adult_unmapped} ({100*adult_unmapped/adult_total:.1f}%)")
    print()
    print(f"✅ Saved inventory to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
