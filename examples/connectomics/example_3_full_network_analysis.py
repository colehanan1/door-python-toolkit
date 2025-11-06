"""
Example 3: Full Network Analysis
=================================

This example demonstrates comprehensive network analysis including:
- Hub neuron detection
- Community detection
- Network statistics
- Glomerulus-level heatmaps

Use cases:
- Understanding global network organization
- Identifying key hub neurons
- Detecting functional modules
- Publication-ready network visualizations
"""

from pathlib import Path
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.connectomics.statistics import NetworkStatistics
from door_toolkit.connectomics.visualization import plot_network, plot_heatmap

# Configuration
DATA_FILE = "interglomerular_crosstalk_pathways.csv"
OUTPUT_DIR = Path("output/connectomics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Example 3: Full Network Analysis")
    print("=" * 70)
    print()

    # Load network
    print("Loading network...")
    network = CrossTalkNetwork.from_csv(DATA_FILE)
    network.set_min_synapse_threshold(50)  # Focus on strong connections
    print()

    # Print network summary
    print(network.summary())
    print("\n")

    # Create statistics analyzer
    print("Initializing statistical analysis...")
    stats = NetworkStatistics(network)
    print()

    # --- Hub Neuron Detection ---
    print("=" * 70)
    print("HUB NEURON DETECTION")
    print("=" * 70)
    print()

    print("Top hub LOCAL NEURONS (by degree):")
    print("-" * 70)
    hub_lns = stats.detect_hub_neurons(
        method='degree',
        threshold_percentile=90,
        neuron_category='Local_Neuron'
    )

    for i, (neuron_id, degree) in enumerate(hub_lns[:10]):
        neuron_info = network.get_neuron_info(neuron_id)
        print(f"  {i+1}. {neuron_info['type']} (degree: {degree})")
    print()

    print("Top hub LOCAL NEURONS (by betweenness centrality):")
    print("-" * 70)
    hub_lns_between = stats.detect_hub_neurons(
        method='betweenness',
        threshold_percentile=95,
        neuron_category='Local_Neuron'
    )

    for i, (neuron_id, centrality) in enumerate(hub_lns_between[:10]):
        neuron_info = network.get_neuron_info(neuron_id)
        print(f"  {i+1}. {neuron_info['type']} (betweenness: {centrality:.4f})")
    print()

    # --- Community Detection ---
    print("=" * 70)
    print("COMMUNITY DETECTION")
    print("=" * 70)
    print()

    print("Detecting communities at glomerulus level...")
    communities = stats.detect_communities(
        algorithm='louvain',
        level='glomerulus'
    )

    num_communities = max(communities.values()) + 1
    print(f"Found {num_communities} communities")
    print()

    # Group glomeruli by community
    from collections import defaultdict
    community_members = defaultdict(list)
    for glom, comm_id in communities.items():
        community_members[comm_id].append(glom)

    print("Community composition:")
    for comm_id in sorted(community_members.keys()):
        members = community_members[comm_id]
        print(f"\n  Community {comm_id} ({len(members)} glomeruli):")
        for glom in sorted(members)[:10]:  # Show first 10
            print(f"    - {glom}")
        if len(members) > 10:
            print(f"    ... and {len(members) - 10} more")
    print()

    # --- Path Length Analysis ---
    print("=" * 70)
    print("PATH LENGTH ANALYSIS")
    print("=" * 70)
    print()

    path_stats = stats.analyze_path_lengths()
    print(f"Mean path length: {path_stats['mean_path_length']:.2f}")
    print(f"Median path length: {path_stats['median_path_length']:.1f}")
    print(f"Max path length: {path_stats['max_path_length']}")
    print(f"Analyzed {path_stats['num_paths']} paths")
    print()

    print("Path length distribution:")
    for length, count in sorted(path_stats['path_length_distribution'].items())[:5]:
        print(f"  Length {length}: {count} paths")
    print()

    # --- Asymmetry Analysis ---
    print("=" * 70)
    print("ASYMMETRY ANALYSIS")
    print("=" * 70)
    print()

    print("Calculating asymmetry matrix...")
    asym_matrix = stats.calculate_asymmetry_matrix()

    print(f"Analyzed {len(asym_matrix)} glomerulus pairs")
    print(f"Mean asymmetry ratio: {asym_matrix['asymmetry_ratio'].mean():.3f}")
    print(f"Std asymmetry ratio: {asym_matrix['asymmetry_ratio'].std():.3f}")
    print()

    print("Most asymmetric connections (forward >> reverse):")
    most_asym_forward = asym_matrix.nlargest(5, 'asymmetry_ratio')
    for idx, row in most_asym_forward.iterrows():
        print(f"  {row['source_glomerulus']} → {row['target_glomerulus']}: "
              f"{row['asymmetry_ratio']:.3f} "
              f"(forward: {row['strength_forward']}, reverse: {row['strength_reverse']})")
    print()

    print("Most asymmetric connections (reverse >> forward):")
    most_asym_reverse = asym_matrix.nsmallest(5, 'asymmetry_ratio')
    for idx, row in most_asym_reverse.iterrows():
        print(f"  {row['source_glomerulus']} → {row['target_glomerulus']}: "
              f"{row['asymmetry_ratio']:.3f} "
              f"(forward: {row['strength_forward']}, reverse: {row['strength_reverse']})")
    print()

    # Export asymmetry matrix
    output_asym = OUTPUT_DIR / "asymmetry_matrix.csv"
    asym_matrix.to_csv(output_asym, index=False)
    print(f"Exported asymmetry matrix to {output_asym}")
    print()

    # --- Generate Full Report ---
    print("=" * 70)
    print("COMPREHENSIVE REPORT")
    print("=" * 70)
    print()

    report = stats.generate_full_report()
    print(report)
    print()

    # Save report to file
    output_report = OUTPUT_DIR / "network_analysis_report.txt"
    with open(output_report, 'w') as f:
        f.write(report)
    print(f"Saved full report to {output_report}")
    print()

    # --- Visualizations ---
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    print()

    # Full network plot
    print("Plotting full network (glomerulus level)...")
    plot_network(
        network,
        output_path=OUTPUT_DIR / "full_network_glomerulus.png",
        show_glomeruli=True,
        show_individual_neurons=False,
        layout='spring',
        dpi=300,
        min_synapse_display=20  # Show only strong connections
    )
    print("  ✓ Saved to full_network_glomerulus.png")

    # Heatmap
    print("Plotting glomerulus connectivity heatmap...")
    plot_heatmap(
        network,
        output_path=OUTPUT_DIR / "glomerulus_heatmap.png",
        dpi=300,
        log_scale=True
    )
    print("  ✓ Saved to glomerulus_heatmap.png")
    print()

    # Export network to other formats
    print("Exporting network to other formats...")
    network.export_to_graphml(OUTPUT_DIR / "network.graphml")
    print("  ✓ Exported to GraphML (for Cytoscape)")

    network.export_to_gexf(OUTPUT_DIR / "network.gexf")
    print("  ✓ Exported to GEXF (for Gephi)")
    print()

    print("Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
