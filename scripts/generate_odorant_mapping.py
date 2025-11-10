#!/usr/bin/env python3
"""
Generate complete odorant name → InChIKey mapping from the DoOR cache.

The DoOR response matrix already contains every odorant as a column
identified by its InChIKey. This script extracts the full list and
builds a CSV mapping file that can later be enriched with human-readable
names pulled from PubChem or other metadata sources.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DOOR_CACHE_FILE = Path("door_cache/response_matrix_norm.parquet")
OUTPUT_FILE = Path("data/mappings/odorant_name_to_inchikey_full.csv")


def main() -> None:
    """Entry point."""
    if not DOOR_CACHE_FILE.exists():
        print(f"❌ ERROR: DoOR cache not found: {DOOR_CACHE_FILE}")
        sys.exit(1)

    print("Loading DoOR response matrix...")
    door_df = pd.read_parquet(DOOR_CACHE_FILE)

    # Ensure receptors are rows. If odorants are rows, transpose.
    if door_df.shape[0] > door_df.shape[1]:
        door_df = door_df.T

    inchikeys = [col for col in door_df.columns if col != "rownames"]
    print(f"Found {len(inchikeys)} odorants in DoOR database")

    mapping_df = pd.DataFrame(
        [
            {
                "common_name": inchikey,  # placeholder until enriched
                "inchikey": inchikey,
                "alternative_names": "",
            }
            for inchikey in inchikeys
        ]
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_FILE, index=False)

    print(f"✅ Saved {len(mapping_df)} odorant mappings to {OUTPUT_FILE}")
    print("\n⚠️  NOTE: This file uses InChIKeys as placeholder names.")
    print("   Populate the 'common_name' column using DoOR documentation")
    print("   or run the PubChem enrichment script to fetch canonical names.")


if __name__ == "__main__":
    main()
