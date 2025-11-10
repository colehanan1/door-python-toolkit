#!/usr/bin/env python3
"""
Custom DoOR Toolkit Analysis CLI
=================================

Flexible command-line tool for analyzing specific glomeruli, ORNs, or odorants.
Combines odorant encoding and connectomics network analysis.

Examples:
    # Analyze a specific glomerulus
    python custom_analysis.py --mode single-orn --glomerulus DL5 --threshold 10

    # Compare two glomeruli
    python custom_analysis.py --mode compare --glomeruli DL5 VA1v --threshold 5

    # Find pathways between glomeruli
    python custom_analysis.py --mode pathway --source DL5 --target VA1v

    # Analyze odorant responses across specific receptors
    python custom_analysis.py --mode odorant --odorants "acetic acid" "ethanol" --receptors Or42b Or47b

    # Full network analysis
    python custom_analysis.py --mode network --threshold 10 --detect-hubs

    # Search odorants and analyze their pathways
    python custom_analysis.py --mode odorant-pathway --odorant-pattern "alcohol" --top-n 5
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


def setup_parser():
    """Create argument parser with all CLI options."""
    parser = argparse.ArgumentParser(
        description="Custom DoOR Toolkit Analysis - Flexible CLI for glomeruli and odorant analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Analysis Modes:
  single-orn       Analyze all pathways from one glomerulus/ORN
  compare          Compare pathways between two glomeruli/ORNs
  pathway          Find specific pathways between source and target
  network          Full network analysis with statistics
  odorant          Encode odorants and analyze receptor responses
  odorant-pathway  Find best odorants and trace their pathways

Examples:
  # Analyze DL5 glomerulus with threshold of 10 synapses
  python custom_analysis.py --mode single-orn --glomerulus DL5 --threshold 10

  # Compare DL5 and VA1v bidirectional pathways
  python custom_analysis.py --mode compare --glomeruli DL5 VA1v

  # Find pathways from VM7v to D
  python custom_analysis.py --mode pathway --source VM7v --target D --max-paths 5

  # Network analysis with hub detection
  python custom_analysis.py --mode network --threshold 10 --detect-hubs --detect-communities

  # Analyze specific odorants
  python custom_analysis.py --mode odorant --odorants "acetic acid" "ethanol" --receptors Or42b Or47b

  # Find best odorants for Or47b and analyze pathways
  python custom_analysis.py --mode odorant-pathway --receptor Or47b --top-n 10
        """,
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["single-orn", "compare", "pathway", "network", "odorant", "odorant-pathway"],
        help="Analysis mode to run",
    )

    # Connectomics options
    parser.add_argument(
        "--connectomics-data",
        type=str,
        default="data/interglomerular_crosstalk_pathways.csv",
        help="Path to connectomics CSV file (default: interglomerular_crosstalk_pathways.csv)",
    )

    parser.add_argument(
        "--glomerulus",
        type=str,
        help="Single glomerulus/ORN to analyze (for single-orn mode)",
    )

    parser.add_argument(
        "--glomeruli",
        type=str,
        nargs=2,
        metavar=("GLOM1", "GLOM2"),
        help="Two glomeruli to compare (for compare mode)",
    )

    parser.add_argument(
        "--source",
        type=str,
        help="Source glomerulus/ORN (for pathway mode)",
    )

    parser.add_argument(
        "--target",
        type=str,
        help="Target glomerulus/ORN (for pathway mode)",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Minimum synapse count threshold (default: 5)",
    )

    parser.add_argument(
        "--max-paths",
        type=int,
        default=10,
        help="Maximum number of shortest paths to find (default: 10)",
    )

    parser.add_argument(
        "--by-glomerulus",
        action="store_true",
        help="Analyze at glomerulus level instead of individual neurons",
    )

    # Network analysis options
    parser.add_argument(
        "--detect-hubs",
        action="store_true",
        help="Detect hub neurons in network analysis",
    )

    parser.add_argument(
        "--detect-communities",
        action="store_true",
        help="Detect communities in network analysis",
    )

    parser.add_argument(
        "--hub-method",
        type=str,
        default="degree",
        choices=["degree", "betweenness", "eigenvector"],
        help="Method for hub detection (default: degree)",
    )

    parser.add_argument(
        "--top-n-hubs",
        type=int,
        default=10,
        help="Number of top hubs to report (default: 10)",
    )

    # Odorant encoding options
    parser.add_argument(
        "--door-cache",
        type=str,
        default="door_cache",
        help="Path to DoOR cache directory (default: door_cache)",
    )

    parser.add_argument(
        "--odorants",
        type=str,
        nargs="+",
        help="Specific odorant names to analyze",
    )

    parser.add_argument(
        "--odorant-pattern",
        type=str,
        help="Search pattern for odorants (e.g., 'alcohol', 'acetate')",
    )

    parser.add_argument(
        "--receptors",
        type=str,
        nargs="+",
        help="Specific receptors to analyze",
    )

    parser.add_argument(
        "--receptor",
        type=str,
        help="Single receptor for odorant-pathway mode",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top results to show (default: 10)",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/custom_analysis",
        help="Directory for output files (default: output/custom_analysis)",
    )

    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="Export results to CSV files",
    )

    parser.add_argument(
        "--export-graph",
        action="store_true",
        help="Export network graph (GraphML format)",
    )

    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots",
    )

    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Suppress file output (only print to console)",
    )

    # General options
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser


def normalize_glomerulus_name(glom_name: str) -> str:
    """Normalize glomerulus name to match data format (ORN_XXX)."""
    if glom_name.startswith("ORN_"):
        return glom_name
    else:
        return f"ORN_{glom_name}"


def list_available_glomeruli(network) -> List[str]:
    """Get list of available glomeruli from network."""
    try:
        glomeruli = sorted(list(network.data.glomeruli))
        return glomeruli
    except Exception as e:
        logger.warning(f"Could not get glomeruli list: {e}")
        return []


def analyze_single_orn(args):
    """Analyze pathways from a single ORN/glomerulus."""
    from door_toolkit.connectomics import CrossTalkNetwork, analyze_single_orn

    if not args.glomerulus:
        print("ERROR: --glomerulus required for single-orn mode")
        sys.exit(1)

    # Load network first to check available glomeruli
    network = CrossTalkNetwork.from_csv(args.connectomics_data)
    network.set_min_synapse_threshold(args.threshold)

    # Normalize glomerulus name (auto-add ORN_ prefix if needed)
    glomerulus = normalize_glomerulus_name(args.glomerulus)

    # Check if glomerulus exists
    available_glomeruli = list_available_glomeruli(network)
    if glomerulus not in available_glomeruli:
        print(f"ERROR: Glomerulus '{glomerulus}' not found in network.")
        print(f"\nDid you mean one of these? (showing first 20)")
        for g in available_glomeruli[:20]:
            # Show without ORN_ prefix for readability
            display_name = g.replace("ORN_", "")
            print(f"  {display_name} (full name: {g})")
        print(f"\nTotal available: {len(available_glomeruli)} glomeruli")
        print("\nTip: You can use short names like 'DL5' instead of 'ORN_DL5'")
        sys.exit(1)

    print("=" * 70)
    print(f"Single ORN Analysis: {glomerulus}")
    print("=" * 70)

    if args.verbose:
        print(f"\nNetwork: {network.summary()}")
        print(f"Threshold: {args.threshold} synapses")
        print(f"Level: {'Glomerulus' if args.by_glomerulus else 'Individual neurons'}\n")

    # Analyze
    try:
        results = analyze_single_orn(network, glomerulus, by_glomerulus=args.by_glomerulus)
    except Exception as e:
        print(f"\nERROR during analysis: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Check if any pathways were found
    if results.num_pathways == 0:
        print(f"\nNo pathways found for {glomerulus} with threshold {args.threshold}")
        print(f"\nTroubleshooting:")
        print(f"  1. Try lowering --threshold (current: {args.threshold})")
        print(f"  2. Try --by-glomerulus flag if not already set")
        print(f"  3. Check if glomerulus has connectivity data")
        sys.exit(1)

    # Print results
    print("\n" + results.summary())
    print()

    # Top pathways
    print(f"Top {args.top_n} strongest pathways:")
    print("-" * 70)
    strongest = results.get_strongest_pathways(n=args.top_n)
    for idx, row in strongest.iterrows():
        print(
            f"{row['level1_type']} → {row['level2_type']} "
            f"(synapses: {row['synapse_count_step2']})"
        )

    # Target distribution
    print("\nTarget glomeruli distribution:")
    print("-" * 70)
    targets = results.get_targets_by_glomerulus()
    for target, count in sorted(targets.items(), key=lambda x: x[1], reverse=True)[: args.top_n]:
        print(f"{target}: {count} pathways")

    # Export
    if not args.no_output:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.export_csv:
            csv_path = output_dir / f"{args.glomerulus}_pathways.csv"
            results.to_dataframe().to_csv(csv_path, index=False)
            print(f"\nExported pathways to {csv_path}")

        if args.visualize:
            from door_toolkit.connectomics.visualization import plot_orn_pathways

            plot_path = output_dir / f"{args.glomerulus}_pathways.png"
            plot_orn_pathways(results, output_path=str(plot_path))
            print(f"Saved visualization to {plot_path}")


def analyze_orn_pair(args):
    """Compare pathways between two ORNs/glomeruli."""
    from door_toolkit.connectomics import CrossTalkNetwork, compare_orn_pair

    if not args.glomeruli:
        print("ERROR: --glomeruli GLOM1 GLOM2 required for compare mode")
        sys.exit(1)

    # Load network
    network = CrossTalkNetwork.from_csv(args.connectomics_data)
    network.set_min_synapse_threshold(args.threshold)

    # Normalize glomerulus names
    glom1 = normalize_glomerulus_name(args.glomeruli[0])
    glom2 = normalize_glomerulus_name(args.glomeruli[1])

    # Check if glomeruli exist
    available_glomeruli = list_available_glomeruli(network)
    for glom in [glom1, glom2]:
        if glom not in available_glomeruli:
            print(f"ERROR: Glomerulus '{glom}' not found in network.")
            print(f"\nAvailable glomeruli (showing first 20):")
            for g in available_glomeruli[:20]:
                display_name = g.replace("ORN_", "")
                print(f"  {display_name}")
            sys.exit(1)

    print("=" * 70)
    print(f"ORN Pair Comparison: {glom1} ↔ {glom2}")
    print("=" * 70)

    if args.verbose:
        print(f"\nNetwork: {network.summary()}")
        print(f"Threshold: {args.threshold} synapses\n")

    # Compare
    try:
        comparison = compare_orn_pair(network, glom1, glom2, by_glomerulus=args.by_glomerulus)
    except Exception as e:
        print(f"\nERROR during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    # Print results
    print("\n" + comparison.summary())

    # Export
    if not args.no_output:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.export_csv:
            csv_path = output_dir / f"{glom1}_vs_{glom2}_comparison.csv"
            df = pd.DataFrame(
                {
                    "direction": ["forward", "reverse"],
                    "pathway_count": [
                        comparison.forward_count,
                        comparison.reverse_count,
                    ],
                    "mean_strength": [
                        comparison.forward_mean_strength,
                        comparison.reverse_mean_strength,
                    ],
                    "asymmetry_index": [
                        comparison.asymmetry_index,
                        comparison.asymmetry_index,
                    ],
                }
            )
            df.to_csv(csv_path, index=False)
            print(f"\nExported comparison to {csv_path}")

        if args.visualize:
            from door_toolkit.connectomics.visualization import plot_orn_pair_comparison

            plot_path = output_dir / f"{glom1}_vs_{glom2}_comparison.png"
            plot_orn_pair_comparison(comparison, output_path=str(plot_path))
            print(f"Saved visualization to {plot_path}")


def analyze_pathway(args):
    """Find pathways between source and target."""
    from door_toolkit.connectomics import CrossTalkNetwork, find_pathways

    if not args.source or not args.target:
        print("ERROR: --source and --target required for pathway mode")
        sys.exit(1)

    # Load network
    network = CrossTalkNetwork.from_csv(args.connectomics_data)
    network.set_min_synapse_threshold(args.threshold)

    # Normalize glomerulus names
    source = normalize_glomerulus_name(args.source)
    target = normalize_glomerulus_name(args.target)

    # Check if glomeruli exist
    available_glomeruli = list_available_glomeruli(network)
    for glom in [source, target]:
        if glom not in available_glomeruli:
            print(f"ERROR: Glomerulus '{glom}' not found in network.")
            print(f"\nAvailable glomeruli (showing first 20):")
            for g in available_glomeruli[:20]:
                display_name = g.replace("ORN_", "")
                print(f"  {display_name}")
            sys.exit(1)

    print("=" * 70)
    print(f"Pathway Search: {source} → {target}")
    print("=" * 70)

    if args.verbose:
        print(f"\nNetwork: {network.summary()}")
        print(f"Threshold: {args.threshold} synapses")
        print(f"Max paths: {args.max_paths}\n")

    # Find pathways
    try:
        pathways = find_pathways(
            network, source, target, by_glomerulus=args.by_glomerulus, max_paths=args.max_paths
        )
    except Exception as e:
        print(f"\nERROR during pathway search: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    print(f"\nFound {len(pathways)} pathways\n")

    # Print pathways
    for i, pathway in enumerate(pathways[: args.top_n], 1):
        print(f"Pathway {i}:")
        print(f"  Length: {pathway['length']} steps")
        print(f"  Path: {' → '.join(pathway['path'])}")
        print(f"  Total synapses: {pathway['total_synapses']}")
        print()

    # Export
    if not args.no_output and args.export_csv:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / f"{source}_to_{target}_pathways.csv"
        df = pd.DataFrame(pathways)
        df.to_csv(csv_path, index=False)
        print(f"Exported pathways to {csv_path}")


def analyze_network(args):
    """Full network analysis with statistics."""
    from door_toolkit.connectomics import CrossTalkNetwork
    from door_toolkit.connectomics.statistics import NetworkStatistics

    print("=" * 70)
    print("Full Network Analysis")
    print("=" * 70)

    # Load network
    network = CrossTalkNetwork.from_csv(args.connectomics_data)
    network.set_min_synapse_threshold(args.threshold)

    print(f"\nNetwork: {network.summary()}\n")

    # Statistics
    stats = NetworkStatistics(network)

    # Hub detection
    if args.detect_hubs:
        print(f"\n{'=' * 70}")
        print(f"Hub Detection ({args.hub_method} method)")
        print("=" * 70)

        hubs = stats.detect_hub_neurons(method=args.hub_method, threshold_percentile=90.0)

        print(f"\nTop {args.top_n_hubs} hub neurons:")
        # hubs is already a sorted list of (neuron, score) tuples
        for i, (neuron, score) in enumerate(hubs[: args.top_n_hubs], 1):
            # Get neuron name from graph
            node_data = network.graph.nodes.get(neuron, {})
            # Try 'label' first (ORNs), then 'type' (LNs/PNs)
            name = node_data.get('label') or node_data.get('type', str(neuron))
            category = node_data.get('category', '')
            glomerulus = node_data.get('glomerulus', '')

            # Format display name
            if glomerulus:
                display_name = f"{name} ({category}, {glomerulus})"
            else:
                display_name = f"{name} ({category})" if category else name

            print(f"{i:2d}. {display_name:50s} - Score: {score:.2f}")

    # Community detection
    if args.detect_communities:
        print(f"\n{'=' * 70}")
        print("Community Detection")
        print("=" * 70)

        communities = stats.detect_communities(algorithm="louvain", level="glomerulus")

        print(f"\nDetected {max(communities.values()) + 1} communities\n")

        # Group by community
        comm_groups = {}
        for node, comm_id in communities.items():
            comm_groups.setdefault(comm_id, []).append(node)

        for comm_id in sorted(comm_groups.keys()):
            members = comm_groups[comm_id]
            print(f"Community {comm_id} ({len(members)} members):")
            print(f"  {', '.join(sorted(members)[:10])}")
            if len(members) > 10:
                print(f"  ... and {len(members) - 10} more")
            print()

    # General statistics
    network_stats = network.get_network_statistics()
    print(f"\n{'=' * 70}")
    print("Network Statistics")
    print("=" * 70)
    for key, value in network_stats.items():
        print(f"{key}: {value}")

    # Export
    if not args.no_output:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.export_graph:
            graph_path = output_dir / "network.graphml"
            network.export_to_graphml(str(graph_path))
            print(f"\nExported network graph to {graph_path}")

        if args.visualize:
            from door_toolkit.connectomics.visualization import NetworkVisualizer

            visualizer = NetworkVisualizer(network)
            plot_path = output_dir / "network_heatmap.png"
            visualizer.plot_glomerulus_heatmap(output_path=str(plot_path))
            print(f"Saved heatmap to {plot_path}")


def analyze_odorant(args):
    """Analyze odorant receptor responses."""
    from door_toolkit import DoOREncoder

    print("=" * 70)
    print("Odorant Response Analysis")
    print("=" * 70)

    # Load encoder
    encoder = DoOREncoder(args.door_cache, use_torch=False)

    # Get odorants to analyze
    if args.odorants:
        odorants = args.odorants
    elif args.odorant_pattern:
        odorants = encoder.list_available_odorants(pattern=args.odorant_pattern)
        print(f"\nFound {len(odorants)} odorants matching '{args.odorant_pattern}'")
        odorants = odorants[: args.top_n]
    else:
        print("ERROR: Either --odorants or --odorant-pattern required for odorant mode")
        sys.exit(1)

    print(f"\nAnalyzing {len(odorants)} odorants\n")

    # Encode odorants
    results = []
    for odorant in odorants:
        try:
            response = encoder.encode(odorant)

            # Get receptor coverage
            coverage = encoder.get_receptor_coverage(odorant)

            # Filter to specific receptors if requested
            if args.receptors:
                response_dict = {rec: response[i] for i, rec in enumerate(encoder.receptor_names)}
                filtered = {k: v for k, v in response_dict.items() if k in args.receptors}
            else:
                response_dict = {rec: response[i] for i, rec in enumerate(encoder.receptor_names)}
                # Show top responding receptors
                filtered = dict(
                    sorted(response_dict.items(), key=lambda x: x[1], reverse=True)[: args.top_n]
                )

            results.append(
                {
                    "odorant": odorant,
                    "coverage": coverage,
                    "responses": filtered,
                }
            )

        except Exception as e:
            if args.verbose:
                print(f"Warning: Could not encode '{odorant}': {e}")

    # Print results
    for result in results:
        print(f"\n{result['odorant']}")
        print("-" * 70)
        print(f"Receptor coverage: {result['coverage']:.1%}")
        print(f"\nTop responding receptors:")

        for receptor, response in result["responses"].items():
            print(f"  {receptor:20s}: {response:6.3f}")

    # Export
    if not args.no_output and args.export_csv:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Flatten for CSV export
        rows = []
        for result in results:
            for receptor, response in result["responses"].items():
                rows.append(
                    {
                        "odorant": result["odorant"],
                        "receptor": receptor,
                        "response": response,
                        "coverage": result["coverage"],
                    }
                )

        df = pd.DataFrame(rows)
        csv_path = output_dir / "odorant_responses.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nExported responses to {csv_path}")


def analyze_odorant_pathway(args):
    """Find best odorants for a receptor and analyze their pathways."""
    from door_toolkit.pathways import PathwayAnalyzer

    if not args.receptor:
        print("ERROR: --receptor required for odorant-pathway mode")
        sys.exit(1)

    print("=" * 70)
    print(f"Odorant-Pathway Analysis: {args.receptor}")
    print("=" * 70)

    analyzer = PathwayAnalyzer(args.door_cache)

    # Get odorants
    odorants = analyzer.encoder.odorant_names

    if args.odorant_pattern:
        odorants = [o for o in odorants if args.odorant_pattern.lower() in o.lower()]
        print(f"\nFiltered to {len(odorants)} odorants containing '{args.odorant_pattern}'")

    print(f"\nTesting {len(odorants)} odorants with {args.receptor}...\n")

    # Test each odorant
    results = []
    for odorant in odorants:
        try:
            pathway = analyzer.trace_custom_pathway(
                receptors=[args.receptor], odorants=[odorant], behavior="detection"
            )

            if pathway.strength > 0:
                results.append(
                    {
                        "odorant": odorant,
                        "strength": pathway.strength,
                        "pathway": pathway,
                    }
                )

        except Exception as e:
            if args.verbose:
                print(f"Warning: Could not analyze '{odorant}': {e}")

    if not results:
        print(f"\nNo responsive odorants found for {args.receptor}")
        return

    # Sort by strength
    results.sort(key=lambda x: x["strength"], reverse=True)

    # Print top results
    print(f"\nTop {args.top_n} odorants for {args.receptor}:")
    print("=" * 70)

    for i, result in enumerate(results[: args.top_n], 1):
        print(f"{i:2d}. {result['odorant']:40s} - Strength: {result['strength']:.3f}")

        if args.verbose:
            pathway = result["pathway"]
            print(f"    Pathway: {pathway.receptor} → PN ({pathway.pn_activation:.3f})")

    # Statistics
    print(f"\n{'=' * 70}")
    print("Summary Statistics")
    print("=" * 70)

    strengths = [r["strength"] for r in results]
    print(f"Total responsive odorants: {len(results)}")
    print(f"Mean strength: {sum(strengths) / len(strengths):.3f}")
    print(f"Max strength: {max(strengths):.3f}")
    print(f"Min strength: {min(strengths):.3f}")

    # Distribution
    strong = sum(1 for s in strengths if s >= 0.5)
    moderate = sum(1 for s in strengths if 0.2 <= s < 0.5)
    weak = sum(1 for s in strengths if s < 0.2)

    print(f"\nStrength distribution:")
    print(f"  Strong (≥0.5):      {strong:3d} ({100*strong/len(results):5.1f}%)")
    print(f"  Moderate (0.2-0.5): {moderate:3d} ({100*moderate/len(results):5.1f}%)")
    print(f"  Weak (<0.2):        {weak:3d} ({100*weak/len(results):5.1f}%)")

    # Export
    if not args.no_output and args.export_csv:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame([{"odorant": r["odorant"], "strength": r["strength"]} for r in results])
        csv_path = output_dir / f"{args.receptor}_odorant_responses.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nExported results to {csv_path}")


def main():
    """Main entry point."""
    parser = setup_parser()
    args = parser.parse_args()

    # Route to appropriate analysis function
    if args.mode == "single-orn":
        analyze_single_orn(args)
    elif args.mode == "compare":
        analyze_orn_pair(args)
    elif args.mode == "pathway":
        analyze_pathway(args)
    elif args.mode == "network":
        analyze_network(args)
    elif args.mode == "odorant":
        analyze_odorant(args)
    elif args.mode == "odorant-pathway":
        analyze_odorant_pathway(args)


if __name__ == "__main__":
    main()
