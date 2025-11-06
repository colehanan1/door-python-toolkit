"""
Quick Data Analysis Script
===========================

Analyzes the characteristics of the connectivity data to understand
what pathway types exist and at what synapse strengths.
"""

from pathlib import Path
from door_toolkit.connectomics import CrossTalkNetwork
import pandas as pd

DATA_FILE = "interglomerular_crosstalk_pathways.csv"

def main():
    print("=" * 70)
    print("DATA CHARACTERISTICS ANALYSIS")
    print("=" * 70)
    print()

    # Load full network (no threshold)
    print("Loading full network (no threshold)...")
    network = CrossTalkNetwork.from_csv(DATA_FILE)
    print()

    print(network.summary())
    print("\n")

    # Analyze pathway types
    print("=" * 70)
    print("PATHWAY TYPE BREAKDOWN")
    print("=" * 70)
    print()

    df = network.data.pathways

    # Count by pathway type
    pathway_types = {}
    for _, row in df.iterrows():
        l1_cat = row['level1_category']
        l2_cat = row['level2_category']

        if l1_cat == 'Local_Neuron' and l2_cat == 'ORN':
            ptype = 'ORN→LN→ORN (lateral inhibition)'
        elif l1_cat == 'Local_Neuron' and l2_cat == 'Projection_Neuron':
            ptype = 'ORN→LN→PN (feedforward inhibition)'
        elif l1_cat == 'Projection_Neuron':
            ptype = 'ORN→PN→feedback'
        else:
            ptype = 'Other'

        pathway_types[ptype] = pathway_types.get(ptype, 0) + 1

    for ptype, count in sorted(pathway_types.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * count / len(df)
        print(f"{ptype}: {count:,} pathways ({pct:.1f}%)")
    print()

    # Synapse strength distribution
    print("=" * 70)
    print("SYNAPSE STRENGTH DISTRIBUTION (step 2)")
    print("=" * 70)
    print()

    syn_counts = df['synapse_count_step2']
    thresholds = [1, 5, 10, 20, 50, 100, 200]

    for thresh in thresholds:
        count = (syn_counts >= thresh).sum()
        pct = 100 * count / len(df)
        print(f"≥ {thresh:3d} synapses: {count:,} pathways ({pct:.1f}%)")
    print()

    # Analyze specific glomerulus pairs
    print("=" * 70)
    print("SPECIFIC GLOMERULUS PAIR ANALYSIS")
    print("=" * 70)
    print()

    pairs_to_check = [
        ("ORN_DL5", "ORN_VA1v"),
        ("ORN_DL5", "ORN_DA1"),
        ("ORN_VA1v", "ORN_DA1"),
        ("ORN_VM7v", "ORN_D"),  # From asymmetry analysis
    ]

    for source, target in pairs_to_check:
        print(f"\n{source} → {target}:")
        pathways = network.get_pathways_between_orns(source, target, by_glomerulus=True)

        if not pathways:
            print("  No pathways found")
            continue

        print(f"  Total pathways: {len(pathways)}")

        # Count by type
        type_counts = {'LN': 0, 'PN': 0}
        for p in pathways:
            if p['level1_category'] == 'Local_Neuron':
                type_counts['LN'] += 1
            else:
                type_counts['PN'] += 1

        print(f"  Via LNs: {type_counts['LN']}")
        print(f"  Via PNs: {type_counts['PN']}")

        # Synapse strength stats
        syn_counts_pair = [p['synapse_count_step2'] for p in pathways]
        print(f"  Synapse counts: min={min(syn_counts_pair)}, "
              f"max={max(syn_counts_pair)}, "
              f"mean={sum(syn_counts_pair)/len(syn_counts_pair):.1f}")

        # How many survive threshold 50?
        strong_pathways = [p for p in pathways if p['synapse_count_step2'] >= 50]
        print(f"  Strong pathways (≥50 synapses): {len(strong_pathways)}")

    print()
    print("=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    print()

    # Count ORN→LN→ORN pathways at different thresholds
    orn_ln_orn = df[(df['level1_category'] == 'Local_Neuron') &
                     (df['level2_category'] == 'ORN')]

    print("ORN→LN→ORN (lateral inhibition) pathways at different thresholds:")
    for thresh in [1, 5, 10, 20, 50]:
        count = (orn_ln_orn['synapse_count_step2'] >= thresh).sum()
        pct = 100 * count / len(df)
        print(f"  Threshold ≥{thresh:2d}: {count:,} pathways ({pct:.1f}% of total)")
    print()

    print("KEY FINDINGS:")
    print("  • Most pathways are ORN→PN→feedback, not ORN→LN→ORN")
    print("  • Lateral inhibition (ORN→LN→ORN) pathways are less common")
    print("  • High thresholds (50+) will exclude most lateral inhibition")
    print("  • For lateral inhibition studies, use threshold 5-10")
    print("  • For overall network structure, threshold 50 is good")
    print()


if __name__ == "__main__":
    main()
