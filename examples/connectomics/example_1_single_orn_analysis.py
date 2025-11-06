"""
Example 1: Single ORN/Glomerulus Analysis
==========================================

This example demonstrates how to analyze all pathways originating from
a single ORN or glomerulus.

Use cases:
- Understanding lateral inhibition patterns from a specific glomerulus
- Identifying which other glomeruli are affected by one ORN type
- Quantifying cross-talk strength from a source
"""

from pathlib import Path
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import analyze_single_orn
from door_toolkit.connectomics.visualization import plot_orn_pathways

# Configuration
DATA_FILE = "interglomerular_crosstalk_pathways.csv"  # Adjust path as needed
OUTPUT_DIR = Path("output/connectomics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Example 1: Single ORN/Glomerulus Analysis")
    print("=" * 70)
    print()

    # Load network
    print("Loading network from CSV...")
    network = CrossTalkNetwork.from_csv(DATA_FILE)
    print(f"Loaded network: {network.summary()}")
    print()

    # Set synapse threshold to focus on strong connections
    # Note: Threshold 10 keeps lateral inhibition pathways (which are typically weak)
    # Use 50+ for only the strongest PN feedback connections
    print("Setting minimum synapse threshold to 10...")
    network.set_min_synapse_threshold(10)
    print()

    # Analyze a specific glomerulus (e.g., DL5 - responds to cis-vaccenyl acetate)
    glomerulus = "ORN_DL5"
    print(f"Analyzing pathways from {glomerulus}...")
    print()

    results = analyze_single_orn(
        network,
        glomerulus,
        by_glomerulus=True
    )

    # Print summary
    print(results.summary())
    print()

    # Get the strongest pathways
    print("Top 10 strongest pathways:")
    print("-" * 70)
    strongest = results.get_strongest_pathways(n=10)
    for idx, row in strongest.iterrows():
        print(f"{row['level1_type']} → {row['level2_type']} "
              f"(synapses: {row['synapse_count_step2']})")
    print()

    # Analyze target glomeruli
    print("Target glomeruli distribution:")
    print("-" * 70)
    targets = results.get_targets_by_glomerulus()
    for target_glom, count in sorted(targets.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{target_glom}: {count} pathways")
    print()

    # Export pathways to CSV
    output_csv = OUTPUT_DIR / f"{glomerulus}_pathways.csv"
    df = results.to_dataframe()
    df.to_csv(output_csv, index=False)
    print(f"Exported pathways to {output_csv}")
    print()

    # Visualize pathways
    print("Generating visualization...")
    output_plot = OUTPUT_DIR / f"{glomerulus}_pathways.png"
    plot_orn_pathways(
        network,
        glomerulus,
        output_path=output_plot,
        by_glomerulus=True,
        dpi=300
    )
    print(f"Saved visualization to {output_plot}")
    print()

    # Analyze multiple glomeruli for comparison
    print("Comparing multiple glomeruli:")
    print("=" * 70)
    glomeruli_to_compare = ["ORN_DL5", "ORN_VA1v", "ORN_DA1"]

    for glom in glomeruli_to_compare:
        result = analyze_single_orn(network, glom, by_glomerulus=True)
        print(f"\n{glom}:")
        print(f"  Total pathways: {result.num_pathways}")
        print(f"  Intermediate LNs: {len(result.intermediate_neurons['LNs'])}")
        print(f"  Target ORNs: {len(result.target_neurons['ORNs'])}")

    print()
    print("Analysis complete!")


if __name__ == "__main__":
    main()
