#!/usr/bin/env python3
"""
Extract all unique glomerulus identifiers from FlyWire processed label CSVs.

Scans data/ for files named processed_labels*.csv (or .csv.gz),
collects any columns containing "glom" or "receptor", and writes the
unique values to data/mappings/flywire_glomeruli_complete.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Set

import pandas as pd

DATA_DIR = Path("data")
OUTPUT_FILE = Path("data/mappings/flywire_glomeruli_complete.txt")


def iter_processed_files() -> Iterable[Path]:
    """Yield processed label CSV (or gz) files within data/."""
    yield from DATA_DIR.glob("**/processed_labels*.csv*")


def main() -> None:
    files = list(iter_processed_files())

    if not files:
        print("❌ No processed_labels CSV files found in data/")
        print("   Looking for files like processed_labels.csv or processed_labels_<receptor>.csv")
        sys.exit(1)

    print(f"Found {len(files)} processed label files:")
    for file in files:
        print(f"  • {file}")

    all_glomeruli: Set[str] = set()

    for file in files:
        try:
            df = pd.read_csv(file)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  ⚠️  Could not read {file.name}: {exc}")
            continue

        glom_columns = [
            col for col in df.columns if "glom" in col.lower() or "receptor" in col.lower()
        ]

        if not glom_columns:
            print(f"  ⚠️  No glomerulus/receptor columns found in {file.name}")
            continue

        for col in glom_columns:
            values = (
                df[col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace({"": None})
                .dropna()
                .unique()
            )
            all_glomeruli.update(values)
        print(f"  ✅ Extracted {len(values)} entries from {file.name}")

    glomeruli_list = sorted(all_glomeruli)
    print(f"\n📊 Total unique glomeruli found: {len(glomeruli_list)}\n")

    for glom in glomeruli_list:
        print(f"  • {glom}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        for glom in glomeruli_list:
            handle.write(f"{glom}\n")

    print(f"\n✅ Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
