#!/usr/bin/env python3
"""
Test Custom Pathways - Batch Analysis
======================================

This script demonstrates how to test multiple receptor-odorant combinations
and compare pathway strengths.
"""

from door_toolkit.pathways import PathwayAnalyzer
import pandas as pd

print("=" * 70)
print("Testing Multiple Custom Pathways")
print("=" * 70)

# Initialize analyzer
analyzer = PathwayAnalyzer("door_cache")

# Test suite of receptor-odorant pairs from literature and hypothesis
test_cases = [
    # Known attractive pathways
    ("Or42b", ["ethyl butyrate"], "fruit attraction"),
    ("Or42b", ["ethyl acetate"], "fruit attraction"),
    ("Or47b", ["1-hexanol"], "feeding attraction"),
    ("Or7a", ["geranyl acetate"], "floral attraction"),

    # Other receptor-odorant combinations
    ("Or10a", ["benzyl acetate"], "detection"),
    ("Or22", ["ethyl butyrate"], "detection"),
    ("Or23a", ["geranyl acetate"], "detection"),
]

print("\nTesting pathways...")
results = []

for receptor, odorants, behavior in test_cases:
    try:
        pathway = analyzer.trace_custom_pathway(
            receptors=[receptor],
            odorants=odorants,
            behavior=behavior
        )

        results.append({
            "receptor": receptor,
            "odorant": odorants[0],
            "behavior": behavior,
            "strength": pathway.strength,
            "contribution": pathway.receptor_contributions.get(receptor, 0.0)
        })

        print(f"✓ {receptor} → {odorants[0]}: {pathway.strength:.3f}")

    except Exception as e:
        print(f"✗ {receptor} → {odorants[0]}: ERROR - {e}")
        results.append({
            "receptor": receptor,
            "odorant": odorants[0],
            "behavior": behavior,
            "strength": 0.0,
            "contribution": 0.0
        })

# Create summary DataFrame
df = pd.DataFrame(results)
df = df.sort_values("strength", ascending=False)

print("\n" + "=" * 70)
print("Pathway Strength Summary")
print("=" * 70)
print(df.to_string(index=False))

# Categorize by strength
print("\n" + "=" * 70)
print("Strength Categories")
print("=" * 70)

strong = df[df["strength"] >= 0.5]
moderate = df[(df["strength"] >= 0.2) & (df["strength"] < 0.5)]
weak = df[(df["strength"] > 0.0) & (df["strength"] < 0.2)]
no_response = df[df["strength"] == 0.0]

print(f"\nStrong activation (≥0.5): {len(strong)} pathways")
if len(strong) > 0:
    for _, row in strong.iterrows():
        print(f"  - {row['receptor']} → {row['odorant']}: {row['strength']:.3f}")

print(f"\nModerate activation (0.2-0.5): {len(moderate)} pathways")
if len(moderate) > 0:
    for _, row in moderate.iterrows():
        print(f"  - {row['receptor']} → {row['odorant']}: {row['strength']:.3f}")

print(f"\nWeak activation (0.0-0.2): {len(weak)} pathways")
if len(weak) > 0:
    for _, row in weak.iterrows():
        print(f"  - {row['receptor']} → {row['odorant']}: {row['strength']:.3f}")

print(f"\nNo response (0.0): {len(no_response)} pathways")

# Test multi-receptor pathway
print("\n" + "=" * 70)
print("Testing Multi-Receptor Pathway")
print("=" * 70)

multi_receptor_pathway = analyzer.trace_custom_pathway(
    receptors=["Or42b", "Or47b", "Or7a"],
    odorants=["ethyl acetate"],
    behavior="combined attraction"
)

print(f"\nPathway: Or42b + Or47b + Or7a → ethyl acetate")
print(f"Overall strength: {multi_receptor_pathway.strength:.3f}")
print(f"\nReceptor contributions:")
for receptor, contrib in sorted(
    multi_receptor_pathway.receptor_contributions.items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"  {receptor}: {contrib:.3f}")

# Test multi-odorant pathway
print("\n" + "=" * 70)
print("Testing Multi-Odorant Pathway")
print("=" * 70)

multi_odorant_pathway = analyzer.trace_custom_pathway(
    receptors=["Or42b"],
    odorants=["ethyl butyrate", "ethyl acetate", "benzyl acetate"],
    behavior="fruit ester blend"
)

print(f"\nPathway: Or42b → fruit ester blend")
print(f"Overall strength: {multi_odorant_pathway.strength:.3f}")
print(f"\nOdorant responses:")
for key, response in multi_odorant_pathway.metadata["odorant_responses"].items():
    odorant = key.split(":")[-1]
    print(f"  {odorant}: {response:.3f}")

print("\n" + "=" * 70)
print("Analysis Complete!")
print("=" * 70)
print(f"\nKey findings:")
print(f"  - Tested {len(test_cases)} receptor-odorant pairs")
print(f"  - Strongest pathway: {df.iloc[0]['receptor']} → {df.iloc[0]['odorant']} ({df.iloc[0]['strength']:.3f})")
print(f"  - {len(strong)} pathways show strong activation (≥0.5)")
print(f"  - {len(moderate) + len(weak)} pathways show detectable activation")
