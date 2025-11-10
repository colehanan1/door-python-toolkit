#!/usr/bin/env python3
"""
Analysis 2: Odor-Specific Subnetwork Extraction
===============================================

Extract and visualize the active cross-talk network for specific odorants.

Shows how different odorants activate different functional circuits with
distinct topological properties.

Usage:
    python scripts/analysis_2_odor_subnetwork.py --odorant "acetic acid" --threshold 0.3
    python scripts/analysis_2_odor_subnetwork.py --odorant "CO2" --threshold 0.5 --pathway-threshold 10
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import logging

# Default to a non-interactive backend unless the user explicitly overrides it.
if not os.environ.get("MPLBACKEND"):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from door_toolkit.integration import DoORFlyWireIntegrator
from door_toolkit.integration.door_utils import get_odorant_activated_receptors
from door_toolkit.integration.odorant_mapper import OdorantMapper

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analysis 2: Extract odor-specific cross-talk subnetwork",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example Odorants (from DoOR):
  Specialists (narrow activation):
    - CO2 (Gr21a.Gr63a only)
    - geranyl acetate (Or82a specialist)
    - geosmin (Or56a specialist)
    - methyl laurate (Or47b specialist)

  Generalists (broad activation):
    - 2-heptanone
    - hexanol
    - isopentyl acetate
    - acetic acid
        """
    )

    parser.add_argument(
        "--odorant",
        type=str,
        required=False,
        help="Odorant name (e.g., 'acetic acid', 'CO2', 'geranyl acetate') or InChIKey"
    )

    parser.add_argument(
        "--list-odorants",
        action="store_true",
        help="List all available odorant names and exit"
    )

    parser.add_argument(
        "--threshold",
        "--activation-threshold",
        dest="activation_threshold",
        type=float,
        default=0.3,
        help="Activation threshold (0-1). Response must exceed this to be considered activated (default: 0.3)"
    )

    parser.add_argument(
        "--pathway-threshold",
        type=int,
        default=1,
        help="Minimum synapse count for cross-talk pathways (default: 1). Lower values find more connections."
    )

    parser.add_argument(
        "--door-cache",
        type=str,
        default="door_cache",
        help="Path to DoOR cache directory (default: door_cache)"
    )

    parser.add_argument(
        "--connectomics-data",
        type=str,
        default="data/interglomerular_crosstalk_pathways.csv",
        help="Path to connectomics CSV (default: data/interglomerular_crosstalk_pathways.csv)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/integration/analysis_2",
        help="Output directory (default: output/integration/analysis_2)"
    )

    parser.add_argument(
        "--layout",
        type=str,
        choices=["spring", "circular", "kamada_kawai", "spectral"],
        default="spring",
        help="Network layout algorithm (default: spring)"
    )

    return parser.parse_args()


def extract_odor_subnetwork(integrator, odorant_name, activation_threshold, pathway_threshold):
    """
    Extract cross-talk subnetwork for a specific odorant.

    Returns:
        tuple: (activated_receptors, subnetwork_graph, activation_strengths)
    """
    logger.info(f"Extracting subnetwork for odorant: {odorant_name}")

    activated_series = get_odorant_activated_receptors(
        odorant_name,
        integrator.door_matrix,
        activation_threshold
    )

    if activated_series.empty:
        raise ValueError(
            f"Odorant '{odorant_name}' did not activate any receptors above threshold {activation_threshold}."
        )

    activated = activated_series.index.tolist()
    activation_strengths = activated_series.to_dict()

    logger.info(
        "Odorant activates %d receptors: %s%s",
        len(activated),
        activated[:5],
        "..." if len(activated) > 5 else "",
    )

    # Map to FlyWire glomeruli
    activated_glomeruli = []
    unmapped_receptors = []

    for r in activated:
        if r in integrator.door_to_flywire:
            glom = integrator.door_to_flywire[r]
            activated_glomeruli.append(glom)
            logger.debug(f"  {r} → {glom}")
        else:
            unmapped_receptors.append(r)
            logger.debug(f"  {r} → NOT MAPPED")

    logger.info(f"Mapped {len(activated_glomeruli)}/{len(activated)} receptors to FlyWire glomeruli")

    if unmapped_receptors:
        logger.warning(f"  {len(unmapped_receptors)} receptors not mapped: {unmapped_receptors[:3]}{'...' if len(unmapped_receptors) > 3 else ''}")

    if len(activated_glomeruli) == 0:
        raise ValueError(
            f"No activated receptors could be mapped to FlyWire glomeruli.\n"
            f"  Activated receptors: {list(activated)[:5]}\n"
            f"  Available mappings: {len(integrator.door_to_flywire)}"
        )

    activated_receptors_mapped = [rec for rec in activated if rec in integrator.door_to_flywire]

    logger.info("Building cross-talk subnetwork using DoOR-indexed connectivity...")
    try:
        connectivity_matrix = integrator.get_connectivity_matrix_door_indexed(
            threshold=pathway_threshold,
            pathway_type="all"
        )
    except ValueError as exc:
        logger.error("Failed to build connectivity matrix: %s", exc)
        raise

    receptor_list = sorted(activated_receptors_mapped)
    glom_map = integrator.door_to_flywire
    edges_found = []

    for i, rec1 in enumerate(receptor_list):
        for rec2 in receptor_list[i + 1:]:
            if rec1 in connectivity_matrix.index and rec2 in connectivity_matrix.index:
                try:
                    conn_val = float(connectivity_matrix.at[rec1, rec2])
                except (KeyError, ValueError):
                    continue

                if conn_val > 0:
                    pathway_type = "inhibitory"
                    try:
                        glom1 = glom_map.get(rec1)
                        glom2 = glom_map.get(rec2)
                        if glom1 and glom2:
                            pathways = integrator.network.get_pathways_between_orns(
                                glom1,
                                glom2,
                                by_glomerulus=True
                            )
                            if pathways:
                                has_inhib = any(p.get("level1_category") == "Local_Neuron" for p in pathways)
                                has_exc = any(p.get("level1_category") == "Projection_Neuron" for p in pathways)
                                if has_exc and not has_inhib:
                                    pathway_type = "excitatory"
                                elif has_exc and has_inhib:
                                    pathway_type = "mixed"
                                else:
                                    pathway_type = "inhibitory"
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.debug("Could not determine pathway type for %s↔%s: %s", rec1, rec2, exc)

                    edges_found.append({
                        "source": rec1,
                        "target": rec2,
                        "weight": conn_val,
                        "pathway_type": pathway_type
                    })

    logger.info("✅ Found %d edges among %d receptors", len(edges_found), len(receptor_list))

    import networkx as nx  # Local import to keep dependency scoped

    subgraph = nx.Graph()
    for rec in receptor_list:
        subgraph.add_node(
            rec,
            receptor=rec,
            activation=activation_strengths.get(rec, 0),
            node_type="activated_receptor"
        )

    for edge in edges_found:
        subgraph.add_edge(
            edge["source"],
            edge["target"],
            weight=edge["weight"],
            pathway_type=edge.get("pathway_type", "unknown")
        )

    if len(edges_found) == 0:
        logger.warning("\n⚠️  NO EDGES FOUND")
        logger.warning("   Activated receptors (mapped): %d", len(receptor_list))
        logger.warning("   Try lowering --pathway-threshold (current: %s)", pathway_threshold)
    else:
        logger.info("\n✅ DETECTED EDGES:")
        for edge in sorted(edges_found, key=lambda x: x["weight"], reverse=True)[:5]:
            logger.info(
                "   %s ↔ %s  %6.0f syn",
                edge["source"],
                edge["target"],
                edge["weight"]
            )

    return activated, subgraph, activation_strengths


def analyze_subnetwork(subgraph, odorant_name):
    """
    Analyze topological properties of subnetwork.
    """
    print()
    print("=" * 70)
    print(f"Subnetwork Analysis: {odorant_name}")
    print("=" * 70)

    # Basic stats
    print(f"\nNetwork Size:")
    print(f"  Nodes (activated glomeruli): {subgraph.number_of_nodes()}")
    print(f"  Edges (cross-talk pathways): {subgraph.number_of_edges()}")

    if subgraph.number_of_edges() == 0:
        print("\n  No cross-talk pathways found between activated glomeruli!")
        print("  Try lowering --pathway-threshold")
        return

    # Density
    possible_edges = subgraph.number_of_nodes() * (subgraph.number_of_nodes() - 1)
    density = subgraph.number_of_edges() / possible_edges if possible_edges > 0 else 0
    print(f"  Density: {density:.4f} ({100*density:.2f}% of possible connections)")

    # Degree distribution
    if subgraph.is_directed():
        in_degrees = dict(subgraph.in_degree())
        out_degrees = dict(subgraph.out_degree())
        total_degree = {node: in_degrees[node] + out_degrees[node] for node in subgraph.nodes()}

        print(f"\nConnectivity:")
        print(f"  Mean in-degree: {np.mean(list(in_degrees.values())):.2f}")
        print(f"  Mean out-degree: {np.mean(list(out_degrees.values())):.2f}")
        print(f"  Max in-degree: {max(in_degrees.values())}")
        print(f"  Max out-degree: {max(out_degrees.values())}")
    else:
        total_degree = dict(subgraph.degree())
        print(f"\nConnectivity:")
        print(f"  Mean degree: {np.mean(list(total_degree.values())):.2f}")
        print(f"  Max degree: {max(total_degree.values(), default=0)}")

    print(f"\nTop 5 most connected nodes (by degree):")
    top_nodes = sorted(total_degree.items(), key=lambda x: x[1], reverse=True)[:5]
    for node, degree in top_nodes:
        print(f"  {node}: {degree} connections")

    # Edge weights
    weights = [data['weight'] for _, _, data in subgraph.edges(data=True)]
    print(f"\nSynapse Weights:")
    print(f"  Mean: {np.mean(weights):.2f} synapses")
    print(f"  Median: {np.median(weights):.2f} synapses")
    print(f"  Range: {min(weights)} - {max(weights)} synapses")

    # Pathway types
    pathway_types = [data.get('pathway_type', 'unknown') for _, _, data in subgraph.edges(data=True)]
    from collections import Counter
    type_counts = Counter(pathway_types)
    print(f"\nPathway Types:")
    for ptype, count in type_counts.items():
        print(f"  {ptype}: {count} ({100*count/len(pathway_types):.1f}%)")

    print()


def visualize_subnetwork(subgraph, odorant_name, output_dir, layout_algorithm):
    """
    Create network visualization.
    """
    logger.info("Creating network visualization...")

    if subgraph.number_of_nodes() == 0:
        logger.warning("Empty subnetwork, skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(14, 12))

    # Layout
    if layout_algorithm == "spring":
        pos = nx.spring_layout(subgraph, k=2, iterations=50)
    elif layout_algorithm == "circular":
        pos = nx.circular_layout(subgraph)
    elif layout_algorithm == "kamada_kawai":
        pos = nx.kamada_kawai_layout(subgraph)
    elif layout_algorithm == "spectral":
        pos = nx.spectral_layout(subgraph)

    # Node sizes proportional to activation strength
    node_activations = [subgraph.nodes[node].get('activation', 0.3) for node in subgraph.nodes()]
    degrees = dict(subgraph.degree())
    node_sizes = [300 + degrees.get(node, 0) * 200 for node in subgraph.nodes()]
    node_colors = node_activations

    # Draw nodes
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        alpha=0.8,
        edgecolors='black',
        linewidths=2,
        ax=ax
    )

    edge_weights_map = {(u, v): subgraph[u][v].get('weight', 1) for u, v in subgraph.edges()}
    edge_colors = []
    edge_widths = []
    for (u, v), weight in edge_weights_map.items():
        edge_widths.append(0.5 + 3.0 * np.log10(weight + 1))
        ptype = subgraph[u][v].get('pathway_type', 'unknown')
        if ptype == 'inhibitory':
            edge_colors.append('blue')
        elif ptype == 'excitatory':
            edge_colors.append('red')
        elif ptype == 'mixed':
            edge_colors.append('purple')
        else:
            edge_colors.append('gray')

    nx.draw_networkx_edges(
        subgraph,
        pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.7,
        ax=ax
    )

    if 0 < len(edge_weights_map) <= 20:
        top_edges = sorted(edge_weights_map.items(), key=lambda item: item[1], reverse=True)[:5]
        edge_labels = {edge: f"{weight:.0f}" for edge, weight in top_edges}
        nx.draw_networkx_edge_labels(
            subgraph,
            pos,
            edge_labels=edge_labels,
            font_size=7,
            font_color='darkblue',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
            ax=ax
        )

    # Draw labels
    labels = {node: node.replace("ORN_", "") for node in subgraph.nodes()}
    nx.draw_networkx_labels(
        subgraph,
        pos,
        labels,
        font_size=9,
        font_weight='bold',
        ax=ax
    )

    # Title
    ax.set_title(
        f'Odor-Specific Cross-Talk Network: {odorant_name}\n'
        f'({subgraph.number_of_nodes()} glomeruli, {subgraph.number_of_edges()} pathways)',
        fontsize=16,
        fontweight='bold'
    )

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Inhibitory (LN-mediated)'),
        Patch(facecolor='red', alpha=0.6, label='Excitatory (PN-mediated)'),
        Patch(facecolor='purple', alpha=0.6, label='Mixed'),
        Patch(facecolor='gray', alpha=0.6, label='Unknown')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

    ax.axis('off')
    plt.tight_layout()

    # Save
    plot_path = output_dir / f"subnetwork_{odorant_name.replace(' ', '_')}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved network visualization to {plot_path}")
    plt.close()


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-odorants option
    if args.list_odorants:
        try:
            mapper = OdorantMapper()
            odorants = mapper.list_all_odorants()

            print(f"\n📋 Available Odorants ({len(odorants)} total):\n")
            for idx, (name, inchikey) in enumerate(odorants, 1):
                print(f"  {idx:3d}. {name:<30} ({inchikey})")

            print("\n💡 Usage:")
            print('   python scripts/analysis_2_odor_subnetwork.py --odorant "<name>"')
            print("   Names are case-insensitive: '1-hexanol', '1-Hexanol', '1-HEXANOL' all work\n")
            sys.exit(0)
        except FileNotFoundError as e:
            logger.error(f"❌ {str(e)}")
            sys.exit(1)

    # Validate required arguments
    if not args.odorant:
        print("Error: --odorant is required (or use --list-odorants to see available odorants)")
        sys.exit(1)

    print()
    print("DoOR-FlyWire Integration: Analysis 2")
    print("=" * 70)
    print()

    # Initialize integrator
    logger.info("Initializing integrator...")
    integrator = DoORFlyWireIntegrator(
        door_cache=args.door_cache,
        connectomics_data=args.connectomics_data
    )

    # Extract subnetwork with error handling
    try:
        activated, subgraph, activations = extract_odor_subnetwork(
            integrator,
            args.odorant,
            args.activation_threshold,
            args.pathway_threshold
        )
    except ValueError as e:
        logger.error(f"❌ {str(e)}")
        logger.info("\n💡 Use --list-odorants to see all available odorant names")
        sys.exit(1)

    safe_odorant_name = args.odorant.replace(" ", "_")
    stats_dir = Path(args.output_dir)
    stats_dir.mkdir(parents=True, exist_ok=True)

    nodes = list(subgraph.nodes())
    degrees = dict(subgraph.degree())
    weights = [data.get('weight', 0) for _, _, data in subgraph.edges(data=True)]
    mean_degree = float(np.mean(list(degrees.values()))) if degrees else 0.0
    max_degree = max(degrees.values()) if degrees else 0
    hub_receptor = max(degrees, key=degrees.get) if degrees else None
    mean_weight = float(np.mean(weights)) if weights else 0.0
    median_weight = float(np.median(weights)) if weights else 0.0
    max_weight = max(weights) if weights else 0.0
    n_components = nx.number_connected_components(subgraph.to_undirected()) if nodes else 0

    stats_dict = {
        'odorant': args.odorant,
        'n_receptors_activated': len(activated),
        'n_receptors_mapped': len(nodes),
        'n_nodes': subgraph.number_of_nodes(),
        'n_edges': subgraph.number_of_edges(),
        'density': nx.density(subgraph) if subgraph.number_of_nodes() > 1 else 0.0,
        'mean_degree': mean_degree,
        'max_degree': max_degree,
        'hub_receptor': hub_receptor,
        'hub_degree': degrees.get(hub_receptor, 0) if hub_receptor else 0,
        'mean_weight': mean_weight,
        'median_weight': median_weight,
        'max_weight': max_weight,
        'n_components': n_components
    }

    stats_file = stats_dir / f"network_stats_{safe_odorant_name}.csv"
    pd.DataFrame([stats_dict]).to_csv(stats_file, index=False)
    logger.info("✅ Saved network statistics to %s", stats_file)

    # Analyze
    analyze_subnetwork(subgraph, args.odorant)

    # Visualize
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_subnetwork(subgraph, args.odorant, output_dir, args.layout)

    # Save data
    # Save activated receptors
    activated_df = pd.DataFrame({
        'receptor': activated,
        'activation': [activations.get(r, 0) for r in activated]
    })
    csv_path = output_dir / f"activated_receptors_{args.odorant.replace(' ', '_')}.csv"
    activated_df.to_csv(csv_path, index=False)
    logger.info(f"Saved activated receptors to {csv_path}")

    # Save edge list
    if subgraph.number_of_edges() > 0:
        edges_data = []
        for u, v, data in subgraph.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'synapses': data.get('weight', 0),
                'pathway_type': data.get('pathway_type', 'unknown')
            })
        edges_df = pd.DataFrame(edges_data)
        edges_path = output_dir / f"subnetwork_edges_{args.odorant.replace(' ', '_')}.csv"
        edges_df.to_csv(edges_path, index=False)
        logger.info(f"Saved edge list to {edges_path}")

    print()
    print("=" * 70)
    print("Analysis 2 Complete!")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
