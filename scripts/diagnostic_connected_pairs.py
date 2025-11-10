#!/usr/bin/env python3
"""
Diagnostic script to identify strongly connected receptor pairs.

This reveals which receptors have the rare, strong cross-talk connections
and helps identify odorants that should show network edges in Analysis 2.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from door_toolkit.integration.integrator import DoORFlyWireIntegrator
from door_toolkit.integration.odorant_mapper import OdorantMapper

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Find and analyze strongly connected receptor pairs."""
    print("\n" + "=" * 70)
    print("CONNECTED RECEPTOR PAIR DIAGNOSTIC")
    print("=" * 70 + "\n")

    logger.info("Initializing DoOR-FlyWire integrator...")
    integrator = DoORFlyWireIntegrator()

    logger.info("Building connectivity matrix...")
    connectivity = integrator.get_connectivity_matrix_door_indexed(
        threshold=1,
        pathway_type="all"
    )

    receptors: List[str] = connectivity.index.tolist()
    logger.info("Analyzing %d receptors...", len(receptors))

    results = []
    for i, rec1 in enumerate(receptors):
        for rec2 in receptors[i + 1:]:
            conn_strength = connectivity.loc[rec1, rec2]
            if conn_strength > 0:
                results.append({
                    "receptor1": rec1,
                    "receptor2": rec2,
                    "connectivity": conn_strength
                })

    df = pd.DataFrame(results).sort_values("connectivity", ascending=False)

    total_pairs = len(receptors) * (len(receptors) - 1) // 2
    connected_pairs = len(df)
    rate = (connected_pairs / total_pairs * 100) if total_pairs else 0.0

    print(f"\n📊 CONNECTIVITY STATISTICS")
    print("=" * 70)
    print(f"Total receptor pairs analyzed: {total_pairs}")
    print(f"Connected pairs (>0 synapses): {connected_pairs}")
    print(f"Connectivity rate: {rate:.1f}%")

    if connected_pairs > 0:
        print(f"\nConnectivity distribution:")
        print(f"  Min: {df['connectivity'].min():.0f} synapses")
        print(f"  Median: {df['connectivity'].median():.0f} synapses")
        print(f"  Mean: {df['connectivity'].mean():.1f} synapses")
        print(f"  Max: {df['connectivity'].max():.0f} synapses")
    else:
        print("\nNo connected pairs found at the current threshold.")

    print(f"\n🔥 TOP 10 STRONGEST CONNECTIONS")
    print("=" * 70)
    if connected_pairs == 0:
        print("No connections to display.")
    else:
        for idx, row in df.head(10).iterrows():
            print(
                f"{idx + 1:2d}. {row['receptor1']:15s} ↔ {row['receptor2']:15s}  "
                f"{row['connectivity']:6.0f} synapses"
            )

    # Categorize strengths
    categories = {
        "Ultra-strong (≥1000 synapses)": df[df["connectivity"] >= 1000],
        "Strong (100-999 synapses)": df[(df["connectivity"] >= 100) & (df["connectivity"] < 1000)],
        "Moderate (10-99 synapses)": df[(df["connectivity"] >= 10) & (df["connectivity"] < 100)],
        "Weak (1-9 synapses)": df[df["connectivity"] < 10],
    }

    print(f"\n📈 CONNECTIVITY CATEGORIES")
    print("=" * 70)
    for label, subset in categories.items():
        print(f"{label:35s}: {len(subset)} pairs")

    if not categories["Ultra-strong (≥1000 synapses)"].empty:
        mapper = OdorantMapper()
        print(f"\n⚡ ULTRA-STRONG CONNECTIONS (≥1000 synapses)")
        print("=" * 70)
        door_matrix = integrator.door_matrix
        for _, row in categories["Ultra-strong (≥1000 synapses)"].iterrows():
            rec1, rec2 = row["receptor1"], row["receptor2"]
            print(f"\n{rec1} ↔ {rec2}")
            print(f"  Connectivity: {row['connectivity']:.0f} synapses")

            if rec1 in door_matrix.index and rec2 in door_matrix.index:
                resp1 = door_matrix.loc[rec1]
                resp2 = door_matrix.loc[rec2]

                threshold = 0.3
                active1 = set(resp1[resp1 > threshold].dropna().index)
                active2 = set(resp2[resp2 > threshold].dropna().index)
                common = list(active1 & active2)

                print(f"  Odorants activating {rec1}: {len(active1)}")
                print(f"  Odorants activating {rec2}: {len(active2)}")
                print(f"  Odorants activating BOTH: {len(common)}")

                if common:
                    example_names = []
                    for inchikey in common[:5]:
                        name = mapper.get_common_name(inchikey)
                        if name:
                            example_names.append(name)
                    if example_names:
                        print(f"  Example odorants: {', '.join(example_names)}")

    output_dir = Path("output/integration/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "connected_receptor_pairs.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✅ Saved {len(df)} connected pairs to {output_file}")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
