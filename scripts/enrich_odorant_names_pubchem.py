#!/usr/bin/env python3
"""
Fetch human-readable names for InChIKeys using the PubChem REST API.

Reads the InChIKey-only mapping produced by generate_odorant_mapping.py,
queries PubChem for each compound, and writes out a complete mapping file
with best-effort common names.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

SOURCE_FILE = Path("data/mappings/odorant_name_to_inchikey_full.csv")
OUTPUT_FILE = Path("data/mappings/odorant_name_to_inchikey_complete.csv")
API_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{}/property/Title/JSON"


def inchikey_to_name(inchikey: str, session: requests.Session) -> str:
    """
    Query PubChem for the compound name associated with an InChIKey.

    Args:
        inchikey: InChIKey identifier.
        session: Shared requests session for connection pooling.

    Returns:
        Common name if found, otherwise the original InChIKey.
    """
    try:
        response = session.get(API_URL.format(inchikey), timeout=5)
    except requests.RequestException as exc:
        print(f"  Warning: Could not fetch name for {inchikey}: {exc}")
        return inchikey

    if response.status_code != 200:
        return inchikey

    try:
        data = response.json()
        return data["PropertyTable"]["Properties"][0]["Title"]
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  Warning: Unexpected response for {inchikey}: {exc}")
        return inchikey


def main() -> None:
    """Entry point."""
    if not SOURCE_FILE.exists():
        print(f"❌ ERROR: Source mapping not found: {SOURCE_FILE}")
        sys.exit(1)

    mapping_df = pd.read_csv(SOURCE_FILE)
    total = len(mapping_df)

    print(f"Fetching names for {total} odorants from PubChem...")
    print("This may take several minutes. Press Ctrl+C to abort.")

    session = requests.Session()

    for idx, row in mapping_df.iterrows():
        inchikey = row["inchikey"]
        name = inchikey_to_name(inchikey, session)
        mapping_df.at[idx, "common_name"] = name

        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"  Processed {idx + 1}/{total} odorants...")

        # PubChem allows ~5 requests/sec; stay well below the limit.
        time.sleep(0.25)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ Saved complete mapping with names to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
