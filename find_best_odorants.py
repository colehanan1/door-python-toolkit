#!/usr/bin/env python3
"""
Find Best Odorants for a Receptor
==================================

This script helps you discover which odorants activate a specific receptor most strongly.
"""

import sys
from door_toolkit.pathways import PathwayAnalyzer
import pandas as pd

def find_best_odorants(receptor_name, top_n=10, odorant_pattern=None):
    """Find top odorants that activate a specific receptor."""

    print("=" * 70)
    print(f"Finding Best Odorants for {receptor_name}")
    print("=" * 70)

    analyzer = PathwayAnalyzer("door_cache")

    # Get all odorants or filter by pattern
    odorants = analyzer.encoder.odorant_names

    if odorant_pattern:
        odorants = [o for o in odorants if odorant_pattern.lower() in o.lower()]
        print(f"\nFiltered to {len(odorants)} odorants containing '{odorant_pattern}'")

    print(f"\nTesting {len(odorants)} odorants with {receptor_name}...")

    # Test each odorant
    results = []
    for odorant in odorants:
        try:
            pathway = analyzer.trace_custom_pathway(
                receptors=[receptor_name],
                odorants=[odorant],
                behavior="detection"
            )

            if pathway.strength > 0:  # Only include responsive odorants
                results.append({
                    "odorant": odorant,
                    "strength": pathway.strength
                })

        except Exception:
            continue

    if not results:
        print(f"\n⚠ No responsive odorants found for {receptor_name}")
        return None

    # Sort by strength
    df = pd.DataFrame(results)
    df = df.sort_values("strength", ascending=False)

    print(f"\n{'=' * 70}")
    print(f"Top {top_n} Odorants for {receptor_name}")
    print("=" * 70)

    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        print(f"{i:2d}. {row['odorant']:40s} - Strength: {row['strength']:.3f}")

    # Summary statistics
    print(f"\n{'=' * 70}")
    print("Summary Statistics")
    print("=" * 70)
    print(f"Total responsive odorants: {len(df)}")
    print(f"Mean strength: {df['strength'].mean():.3f}")
    print(f"Median strength: {df['strength'].median():.3f}")
    print(f"Max strength: {df['strength'].max():.3f}")
    print(f"Min strength (>0): {df['strength'].min():.3f}")

    # Strength distribution
    strong = len(df[df["strength"] >= 0.5])
    moderate = len(df[(df["strength"] >= 0.2) & (df["strength"] < 0.5)])
    weak = len(df[df["strength"] < 0.2])

    print(f"\nStrength distribution:")
    print(f"  Strong (≥0.5):      {strong:3d} odorants ({100*strong/len(df):5.1f}%)")
    print(f"  Moderate (0.2-0.5): {moderate:3d} odorants ({100*moderate/len(df):5.1f}%)")
    print(f"  Weak (<0.2):        {weak:3d} odorants ({100*weak/len(df):5.1f}%)")

    return df


if __name__ == "__main__":
    # Command line usage
    if len(sys.argv) < 2:
        print("Usage: python find_best_odorants.py RECEPTOR [PATTERN] [TOP_N]")
        print("\nExamples:")
        print("  python find_best_odorants.py Or42b")
        print("  python find_best_odorants.py Or47b alcohol 10")
        print("  python find_best_odorants.py Or7a acetate 15")
        sys.exit(1)

    receptor = sys.argv[1]
    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    df = find_best_odorants(receptor, top_n=top_n, odorant_pattern=pattern)

    if df is not None:
        # Save to CSV
        output_file = f"{receptor}_best_odorants.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✓ Full results saved to {output_file}")
