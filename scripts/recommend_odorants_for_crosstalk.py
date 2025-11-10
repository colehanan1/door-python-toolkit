#!/usr/bin/env python3
"""
Recommend odorants likely to show cross-talk in Analysis 2.

Identifies odorants that activate strongly-connected receptor pairs,
making them good candidates for observing network edges.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from door_toolkit.integration.integrator import DoORFlyWireIntegrator
from door_toolkit.integration.odorant_mapper import OdorantMapper

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def main() -> None:
    print("\n" + "=" * 70)
    print("ODORANT RECOMMENDATION FOR CROSS-TALK ANALYSIS")
    print("=" * 70 + "\n")

    integrator = DoORFlyWireIntegrator()
    mapper = OdorantMapper()

    pairs_file = Path("output/integration/diagnostics/connected_receptor_pairs.csv")
    if not pairs_file.exists():
        print("❌ ERROR: Connected pairs file not found!")
        print(f"   Expected: {pairs_file}")
        print("   Please run: python scripts/diagnostic_connected_pairs.py first")
        return

    pairs_df = pd.read_csv(pairs_file)
    strong_pairs = pairs_df[pairs_df["connectivity"] >= 10]

    print(f"Analyzing {len(strong_pairs)} strongly connected pairs (≥10 synapses)...\n")

    door_matrix = integrator.door_matrix
    threshold = 0.3
    recommendations = []

    for _, pair in strong_pairs.iterrows():
        rec1, rec2 = pair["receptor1"], pair["receptor2"]

        if rec1 not in door_matrix.index or rec2 not in door_matrix.index:
            continue

        resp1 = door_matrix.loc[rec1]
        resp2 = door_matrix.loc[rec2]

        active1 = set(resp1[resp1 > threshold].dropna().index)
        active2 = set(resp2[resp2 > threshold].dropna().index)
        common_odors = active1 & active2

        for inchikey in common_odors:
            name = mapper.get_common_name(inchikey)
            if not name:
                continue
            recommendations.append({
                "odorant": name,
                "inchikey": inchikey,
                "receptor1": rec1,
                "receptor2": rec2,
                "connectivity": pair["connectivity"],
                "response1": resp1[inchikey],
                "response2": resp2[inchikey],
            })

    if not recommendations:
        print("No odorants found that activate multiple strongly connected pairs.")
        return

    rec_df = pd.DataFrame(recommendations)
    odor_counts = rec_df.groupby("odorant").agg({
        "connectivity": ["count", "sum", "max"],
        "response1": "mean",
        "response2": "mean"
    }).reset_index()

    odor_counts.columns = [
        "odorant",
        "n_connected_pairs",
        "total_connectivity",
        "max_connectivity",
        "avg_response1",
        "avg_response2"
    ]
    odor_counts = odor_counts.sort_values("n_connected_pairs", ascending=False)

    print("🎯 TOP 20 ODORANTS LIKELY TO SHOW CROSS-TALK")
    print("=" * 70)
    print(f"{'Rank':<5} {'Odorant':<25} {'Connected':<10} {'Max Conn':<10}")
    print(f"{'':5} {'':25} {'Pairs':10} {'Synapses':10}")
    print("-" * 70)

    for idx, row in odor_counts.head(20).iterrows():
        print(
            f"{idx + 1:<5} {row['odorant']:<25} "
            f"{row['n_connected_pairs']:<10.0f} {row['max_connectivity']:<10.0f}"
        )

    output_dir = Path("output/integration/diagnostics")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "odorant_recommendations.csv"
    detail_file = output_dir / "odorant_recommendations_detailed.csv"

    odor_counts.to_csv(summary_file, index=False)
    rec_df.to_csv(detail_file, index=False)

    print(f"\n✅ Saved recommendations to:")
    print(f"   Summary: {summary_file}")
    print(f"   Detailed: {detail_file}")

    print("\n💡 USAGE:")
    for idx, row in odor_counts.head(5).iterrows():
        print(f'   python scripts/analysis_2_odor_subnetwork.py --odorant "{row["odorant"]}"')

    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
