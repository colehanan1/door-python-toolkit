"""
Example 4: Pathway Search
==========================

This example demonstrates how to find specific pathways between neurons or glomeruli.

Use cases:
- Testing hypotheses about specific cross-talk connections
- Finding the strongest pathways between two glomeruli
- Identifying intermediate neurons in specific connections
- Shortest path analysis
"""

from pathlib import Path
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import find_pathways
import pandas as pd

# Configuration
DATA_FILE = "interglomerular_crosstalk_pathways.csv"
OUTPUT_DIR = Path("output/connectomics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("=" * 70)
    print("Example 4: Pathway Search")
    print("=" * 70)
    print()

    # Load network
    print("Loading network...")
    network = CrossTalkNetwork.from_csv(DATA_FILE)
    # Use lower threshold for lateral inhibition pathways
    network.set_min_synapse_threshold(5)
    print()

    # --- Search 1: Find all pathways between two glomeruli ---
    print("=" * 70)
    print("SEARCH 1: All pathways between two glomeruli")
    print("=" * 70)
    print()

    # Use glomeruli that actually have pathways (from asymmetry analysis)
    source_glom = "ORN_VM7v"
    target_glom = "ORN_D"

    print(f"Finding pathways from {source_glom} to {target_glom}...")
    results = find_pathways(
        network,
        source_glom,
        target_glom,
        by_glomerulus=True
    )

    print(f"\nFound {results['num_pathways']} pathways")
    print(f"Total synapses: {results['statistics']['total_synapses']}")
    print(f"Mean synapses per pathway: {results['statistics']['mean_synapses_per_pathway']:.2f}")
    print(f"Shortest path length: {results['statistics'].get('shortest_path_length', 'N/A')}")
    print()

    print("Intermediate neurons:")
    print(f"  Local Neurons: {len(results['intermediate_neurons']['LNs'])}")
    print(f"  Projection Neurons: {len(results['intermediate_neurons']['PNs'])}")
    print()

    # Show top pathways
    if results['pathways']:
        print("Top 10 strongest pathways:")
        print("-" * 70)
        df = pd.DataFrame(results['pathways'])
        top_pathways = df.nlargest(10, 'synapse_count_step2')

        for idx, row in top_pathways.iterrows():
            print(f"{row['orn_glomerulus']} → {row['level1_type']} → "
                  f"{row['level2_glomerulus']} (synapses: {row['synapse_count_step2']})")
        print()

        # Export pathways
        output_csv = OUTPUT_DIR / f"{source_glom}_to_{target_glom}_pathways.csv"
        df.to_csv(output_csv, index=False)
        print(f"Exported pathways to {output_csv}")
    print()

    # --- Search 2: Find pathways with specific intermediate neuron types ---
    print("=" * 70)
    print("SEARCH 2: Filter by intermediate neuron type")
    print("=" * 70)
    print()

    print(f"Finding pathways from {source_glom} to {target_glom} via Local Neurons...")
    results_ln = find_pathways(network, source_glom, target_glom, by_glomerulus=True)

    # Filter for LN-mediated pathways
    df_all = pd.DataFrame(results_ln['pathways'])
    df_ln_only = df_all[df_all['level1_category'] == 'Local_Neuron']

    print(f"Found {len(df_ln_only)} LN-mediated pathways (out of {len(df_all)} total)")
    print()

    # Group by LN type
    if len(df_ln_only) > 0:
        ln_type_counts = df_ln_only['level1_type'].value_counts()
        print("Most common LN types:")
        for ln_type, count in ln_type_counts.head(10).items():
            print(f"  {ln_type}: {count} pathways")
    print()

    # --- Search 3: Find strongest cross-talk between multiple pairs ---
    print("=" * 70)
    print("SEARCH 3: Matrix search - multiple glomerulus pairs")
    print("=" * 70)
    print()

    # Define glomeruli of interest
    glomeruli_of_interest = ["ORN_DL5", "ORN_VA1v", "ORN_DA1", "ORN_V"]

    print(f"Searching all pairs among: {glomeruli_of_interest}")
    print()

    # Build connectivity matrix
    connectivity_results = []

    for source in glomeruli_of_interest:
        for target in glomeruli_of_interest:
            if source == target:
                continue

            results = find_pathways(
                network,
                source,
                target,
                by_glomerulus=True,
                max_pathways=None  # Get all
            )

            if results['num_pathways'] > 0:
                connectivity_results.append({
                    'source': source,
                    'target': target,
                    'num_pathways': results['num_pathways'],
                    'total_synapses': results['statistics']['total_synapses'],
                    'mean_synapses': results['statistics']['mean_synapses_per_pathway'],
                    'num_LNs': len(results['intermediate_neurons']['LNs']),
                    'num_PNs': len(results['intermediate_neurons']['PNs']),
                })

    # Create DataFrame and sort by strength
    df_connectivity = pd.DataFrame(connectivity_results)
    df_connectivity = df_connectivity.sort_values('total_synapses', ascending=False)

    print("Connectivity matrix (sorted by total synapses):")
    print("=" * 70)
    print(df_connectivity.to_string(index=False))
    print()

    # Save connectivity matrix
    output_matrix = OUTPUT_DIR / "connectivity_matrix.csv"
    df_connectivity.to_csv(output_matrix, index=False)
    print(f"Saved connectivity matrix to {output_matrix}")
    print()

    # --- Search 4: Shortest paths ---
    print("=" * 70)
    print("SEARCH 4: Shortest paths between neurons")
    print("=" * 70)
    print()

    # Get neurons from two glomeruli
    source_neurons = network.get_glomerulus_neurons("ORN_DL5")
    target_neurons = network.get_glomerulus_neurons("ORN_VA1v")

    if source_neurons and target_neurons:
        source_neuron = source_neurons[0]
        target_neuron = target_neurons[0]

        print(f"Finding shortest paths from {source_neuron} to {target_neuron}...")

        shortest_paths = network.find_shortest_paths(
            source_neuron,
            target_neuron,
            max_paths=5
        )

        if shortest_paths:
            print(f"Found {len(shortest_paths)} shortest path(s) of length {len(shortest_paths[0]) - 1}")
            print()

            for i, path in enumerate(shortest_paths):
                print(f"Path {i+1}:")
                for j, neuron_id in enumerate(path):
                    neuron_info = network.get_neuron_info(neuron_id)
                    if neuron_info:
                        cell_type = neuron_info.get('type', neuron_id)
                        category = neuron_info.get('category', '')
                        print(f"  {j}. {cell_type} ({category})")
                    else:
                        print(f"  {j}. {neuron_id}")

                    if j < len(path) - 1:
                        edge_data = network.graph[path[j]][path[j+1]]
                        print(f"      ↓ (synapses: {edge_data['synapse_count']})")
                print()
        else:
            print("No paths found")
    print()

    # --- Search 5: Find hubs in pathway ---
    print("=" * 70)
    print("SEARCH 5: Identify hub LNs connecting multiple glomeruli")
    print("=" * 70)
    print()

    # Count how many target glomeruli each LN connects to from source
    ln_target_count = {}

    for target in glomeruli_of_interest:
        if target == source_glom:
            continue

        results = find_pathways(network, source_glom, target, by_glomerulus=True)

        for ln_id in results['intermediate_neurons']['LNs']:
            if ln_id not in ln_target_count:
                ln_target_count[ln_id] = set()
            ln_target_count[ln_id].add(target)

    # Sort by number of targets
    hub_lns = sorted(
        [(ln, len(targets)) for ln, targets in ln_target_count.items()],
        key=lambda x: x[1],
        reverse=True
    )

    print(f"Top hub LNs connecting {source_glom} to multiple glomeruli:")
    print("-" * 70)
    for i, (ln_id, num_targets) in enumerate(hub_lns[:10]):
        ln_info = network.get_neuron_info(ln_id)
        ln_type = ln_info['type'] if ln_info else ln_id
        print(f"  {i+1}. {ln_type}: connects to {num_targets} glomeruli")
    print()

    print("Analysis complete!")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
