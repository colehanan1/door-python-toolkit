#!/usr/bin/env python3
"""
Generate a comprehensive DoOR receptor → FlyWire glomerulus mapping.

Combines canonical receptor-to-glomerulus relationships from the literature
with FlyWire naming conventions so that all key receptors (especially those
with known strong connectivity) are available for Analysis 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

import pandas as pd

STANDARD_MAPPING: List[Tuple[str, str, str]] = [
    ("Or7a", "DL5", "ORN"),
    ("Or10a", "VC3l", "ORN"),
    ("Or13a", "DC4", "ORN"),
    ("Or19a", "D", "ORN"),
    ("Or22a", "DM2", "ORN"),
    ("Or23a", "DM6", "ORN"),
    ("Or33a", "VC1", "ORN"),
    ("Or33b", "DC2", "ORN"),
    ("Or35a", "VC2", "ORN"),
    ("Or42a", "DM3", "ORN"),
    ("Or42b", "DM4", "ORN"),
    ("Or43a", "DM5", "ORN"),
    ("Or43b", "VM2", "ORN"),
    ("Or45a", "VC4", "ORN"),
    ("Or45b", "VC3m", "ORN"),
    ("Or47a", "VA1v", "ORN"),
    ("Or47b", "VA1d", "ORN"),
    ("Or49a", "VA3", "ORN"),
    ("Or49b", "DC1", "ORN"),
    ("Or56a", "DA1", "ORN"),
    ("Or59a", "DM1", "ORN"),
    ("Or59b", "DL4", "ORN"),
    ("Or59c", "VA4", "ORN"),
    ("Or65a", "DL1", "ORN"),
    ("Or67a", "VA6", "ORN"),
    ("Or67b", "DA2", "ORN"),
    ("Or67c", "DA3", "ORN"),
    ("Or67d", "DA1", "ORN"),
    ("Or69a", "D", "ORN"),
    ("Or71a", "DL2d", "ORN"),
    ("Or74a", "DA4m", "ORN"),
    ("Or82a", "VA3", "ORN"),
    ("Or83c", "DC3", "ORN"),
    ("Or85a", "DL3", "ORN"),
    ("Or85b", "VM5v", "ORN"),
    ("Or85c", "VM5d", "ORN"),
    ("Or85d", "VM7d", "ORN"),
    ("Or85e", "VM7v", "ORN"),
    ("Or85f", "VL2a", "ORN"),
    ("Or88a", "VA1lm", "ORN"),
    ("Or92a", "VA7m", "ORN"),
    ("Or98a", "VL2p", "ORN"),
    ("Or98b", "VM3", "ORN"),
    ("Ir75a", "DP1l", "ORN"),
    ("Ir84a", "DP1m", "ORN"),
    ("Gr21a.Gr63a", "V", "ORN"),
    ("ac1", "ac1", "ORN"),
    ("ac2", "ac2", "ORN"),
    ("ac3A", "ac3A", "ORN"),
    ("ac3B", "ac3B", "ORN"),
    ("ab2B", "ab2B", "ORN"),
    ("ab4B", "ab4B", "ORN"),
]

EXTRA_MAPPINGS: List[Dict[str, str]] = [
    {
        "door_name": "Or9a",
        "flywire_glomerulus": "VM4",
        "cell_type": "ORN",
        "source": "Couto et al. 2005; Fishilevich & Vosshall 2005",
        "confidence": "high",
        "notes": "Responds to hexanol and benzaldehyde",
    },
    {
        "door_name": "Or30a",
        "flywire_glomerulus": "DL2v",
        "cell_type": "ORN",
        "source": "Couto et al. 2005; Hallem & Carlson 2006",
        "confidence": "high",
        "notes": "Benzaldehyde-responsive",
    },
    {
        "door_name": "Or33c",
        "flywire_glomerulus": "VC2",
        "cell_type": "ORN",
        "source": "Hallem & Carlson 2006; Silbering et al. 2008",
        "confidence": "high",
        "notes": "Aromatic compounds",
    },
    {
        "door_name": "Or2a",
        "flywire_glomerulus": "DC4",
        "cell_type": "ORN",
        "source": "Couto et al. 2005; Dobritsa et al. 2003",
        "confidence": "high",
        "notes": "Antennal sensillum ab3A",
    },
    {
        "door_name": "Or46a",
        "flywire_glomerulus": "VA7l",
        "cell_type": "ORN",
        "source": "Couto et al. 2005",
        "confidence": "high",
        "notes": "Responds to esters",
    },
    {
        "door_name": "Or56a",
        "flywire_glomerulus": "DA1",
        "cell_type": "ORN",
        "source": "Couto et al. 2005; Hallem & Carlson 2006",
        "confidence": "high",
        "notes": "Geosmin receptor",
    },
    {
        "door_name": "Or94a",
        "flywire_glomerulus": "VA6",
        "cell_type": "ORN",
        "source": "Couto et al. 2005",
        "confidence": "medium",
        "notes": "Phenylacetaldehyde-sensitive",
    },
    {
        "door_name": "Or94b",
        "flywire_glomerulus": "VA7l",
        "cell_type": "ORN",
        "source": "Couto et al. 2005",
        "confidence": "medium",
        "notes": "Co-expressed with Or46a (variant)",
    },
    {
        "door_name": "Ir76a",
        "flywire_glomerulus": "DC1",
        "cell_type": "ORN",
        "source": "Silbering et al. 2011; Benton et al. 2009",
        "confidence": "high",
        "notes": "Ionotropic receptor for acids",
    },
    {
        "door_name": "Ir64a",
        "flywire_glomerulus": "DC4",
        "cell_type": "ORN",
        "source": "Silbering et al. 2011",
        "confidence": "high",
        "notes": "Ionotropic receptor (acids)",
    },
    {
        "door_name": "Ir31a",
        "flywire_glomerulus": "VC4",
        "cell_type": "ORN",
        "source": "Silbering et al. 2011",
        "confidence": "high",
        "notes": "Ionotropic receptor",
    },
    {
        "door_name": "Gr10a",
        "flywire_glomerulus": "V",
        "cell_type": "GRN",
        "source": "Kwon et al. 2007; Jones et al. 2007",
        "confidence": "medium",
        "notes": "Gustatory receptor (CO2-like)",
    },
    {
        "door_name": "Gr10b",
        "flywire_glomerulus": "V",
        "cell_type": "GRN",
        "source": "Kwon et al. 2007",
        "confidence": "medium",
        "notes": "Gustatory receptor",
    },
]

OUTPUT_FILE = Path("data/mappings/door_to_flywire_mapping_complete.csv")
DOOR_CACHE = Path("door_cache/response_matrix_norm.parquet")


def normalize_glomerulus(glomerulus: str) -> str:
    """Format FlyWire glomerulus names as ORN_<NAME> where applicable."""
    clean = glomerulus.strip()
    if not clean:
        raise ValueError("Empty glomerulus name encountered")

    lower = clean.lower()
    if lower.startswith(("ac", "ab")):
        return clean

    if not clean.startswith("ORN_"):
        clean = f"ORN_{clean}"
    return clean


def main() -> None:
    records = []
    for door_name, glomerulus, cell_type in STANDARD_MAPPING:
        records.append(
            {
                "door_name": door_name,
                "flywire_glomerulus": normalize_glomerulus(glomerulus),
                "cell_type": cell_type,
                "source": "literature_standard",
                "confidence": "medium",
                "notes": "",
            }
        )

    mapping_df = pd.DataFrame(records)

    for record in EXTRA_MAPPINGS:
        extra = record.copy()
        extra["flywire_glomerulus"] = normalize_glomerulus(extra["flywire_glomerulus"])
        door_name = extra["door_name"]
        mask = mapping_df["door_name"] == door_name
        if mask.any():
            for key, value in extra.items():
                mapping_df.loc[mask, key] = value
        else:
            mapping_df = pd.concat([mapping_df, pd.DataFrame([extra])], ignore_index=True)

    mapping_df.sort_values("door_name", inplace=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    mapping_df.to_csv(OUTPUT_FILE, index=False)

    print(f"📊 Generated mapping for {len(mapping_df)} receptors")
    print("\nSample mappings:")
    print(mapping_df.head(10).to_string(index=False))

    print("\n🔥 Ultra-strong connectivity pairs:")
    critical = mapping_df[mapping_df["door_name"].isin(["Or85f", "Or92a", "Or98a"])]
    print(critical.to_string(index=False))

    if DOOR_CACHE.exists():
        door_df = pd.read_parquet(DOOR_CACHE)
        if door_df.shape[0] > door_df.shape[1]:
            door_df = door_df.T
        door_receptors = set(door_df.index)
        mapped = set(mapping_df["door_name"])
        missing = sorted(door_receptors - mapped)
        if missing:
            print(f"\n⚠️  Still unmapped DoOR receptors ({len(missing)} shown first 20):")
            for rec in missing[:20]:
                print(f"  • {rec}")
        else:
            print("\n✅ All DoOR receptors are mapped!")
    else:
        print(f"\n⚠️  DoOR cache not found at {DOOR_CACHE}; skipping coverage check.")

    print(f"\n✅ Saved complete mapping to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
