"""
Example 2: ORN Pair Comparison
================================

This example demonstrates how to compare cross-talk between two ORNs or glomeruli.

Use cases:
- Quantifying mutual inhibition between glomeruli
- Testing hypotheses about odor mixture interactions
- Identifying asymmetric cross-talk patterns
"""

from pathlib import Path
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import compare_orn_pair

# Configuration
DATA_FILE = "interglomerular_crosstalk_pathways.csv"
OUTPUT_DIR = Path("output/connectomics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Example 2: ORN Pair Comparison")
    print("=" * 70)
    print()

    # Load network
    print("Loading network...")
    network = CrossTalkNetwork.from_csv(DATA_FILE)
    # Note: Lateral inhibition pathways are weak; use threshold 5-10
    network.set_min_synapse_threshold(5)
    print()

    # Compare two glomeruli
    # DL5 (cis-vaccenyl acetate) vs VA1v (vinegar/acetic acid)
    glom1 = "ORN_DL5"
    glom2 = "ORN_VA1v"

    print(f"Comparing {glom1} vs {glom2}...")
    print()

    comparison = compare_orn_pair(
        network,
        glom1,
        glom2,
        by_glomerulus=True
    )

    # Print summary
    print(comparison.summary())
    print()

    # Detailed analysis
    print("Detailed Analysis:")
    print("-" * 70)
    print(f"Bidirectional cross-talk: {comparison.has_bidirectional_crosstalk}")
    print(f"Asymmetry ratio: {comparison.get_asymmetry_ratio():.3f}")
    print()

    if comparison.get_asymmetry_ratio() > 0.2:
        print(f"→ {glom1} has STRONGER influence on {glom2}")
    elif comparison.get_asymmetry_ratio() < -0.2:
        print(f"→ {glom2} has STRONGER influence on {glom1}")
    else:
        print(f"→ Cross-talk is relatively SYMMETRIC")
    print()

    # Shared intermediate neurons
    print("Shared intermediate neurons:")
    print(f"  Local Neurons: {len(comparison.shared_intermediates['LNs'])}")
    print(f"  Projection Neurons: {len(comparison.shared_intermediates['PNs'])}")
    print()

    if comparison.shared_intermediates['LNs']:
        print("  Top shared LNs:")
        for i, ln_id in enumerate(comparison.shared_intermediates['LNs'][:5]):
            ln_info = network.get_neuron_info(ln_id)
            print(f"    {i+1}. {ln_info['type']}")
    print()

    # Export pathways
    import pandas as pd

    # Forward pathways
    df_forward = pd.DataFrame(comparison.pathways_1_to_2)
    output_forward = OUTPUT_DIR / f"{glom1}_to_{glom2}_pathways.csv"
    df_forward.to_csv(output_forward, index=False)
    print(f"Exported {glom1}→{glom2} pathways to {output_forward}")

    # Reverse pathways
    df_reverse = pd.DataFrame(comparison.pathways_2_to_1)
    output_reverse = OUTPUT_DIR / f"{glom2}_to_{glom1}_pathways.csv"
    df_reverse.to_csv(output_reverse, index=False)
    print(f"Exported {glom2}→{glom1} pathways to {output_reverse}")
    print()

    # Compare multiple pairs
    print("\nComparing multiple glomerulus pairs:")
    print("=" * 70)

    pairs_to_compare = [
        ("ORN_DL5", "ORN_VA1v"),
        ("ORN_DL5", "ORN_DA1"),
        ("ORN_VA1v", "ORN_DA1"),
    ]

    results = []
    for glom_a, glom_b in pairs_to_compare:
        comp = compare_orn_pair(network, glom_a, glom_b, by_glomerulus=True)
        results.append({
            'pair': f"{glom_a} vs {glom_b}",
            'pathways_forward': len(comp.pathways_1_to_2),
            'pathways_reverse': len(comp.pathways_2_to_1),
            'strength_forward': comp.cross_talk_strength['1_to_2'],
            'strength_reverse': comp.cross_talk_strength['2_to_1'],
            'asymmetry': comp.get_asymmetry_ratio(),
            'shared_LNs': len(comp.shared_intermediates['LNs']),
        })

    # Print comparison table
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))
    print()

    # Save comparison table
    output_table = OUTPUT_DIR / "pair_comparisons.csv"
    df_results.to_csv(output_table, index=False)
    print(f"Saved comparison table to {output_table}")
    print()

    print("Analysis complete!")


if __name__ == "__main__":
    main()
