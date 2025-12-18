import os
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re

def extract_root_id_prefix_and_columns(sample_df):
    """Extract the root ID prefix and identify correct column names"""
    columns = sample_df.columns.tolist()
    
    # Find pre and post root ID columns
    pre_col = None
    post_col = None
    prefix = None
    
    for col in columns:
        if 'pre_root_id' in col:
            pre_col = col
            # Extract prefix number (e.g., 720575940 from 'pre_root_id_720575940')
            prefix_match = re.search(r'pre_root_id_(\d+)', col)
            if prefix_match:
                prefix = prefix_match.group(1)
        elif 'post_root_id' in col:
            post_col = col
    
    return pre_col, post_col, prefix

def reconstruct_full_root_id(partial_id, prefix):
    """Reconstruct full root ID by adding prefix"""
    if pd.isna(partial_id):
        return None
    return int(str(prefix) + str(int(partial_id)))

def convert_to_partial_root_id(full_root_id, prefix):
    """Convert full root ID to partial format for searching"""
    source_str = str(full_root_id)
    if source_str.startswith(prefix):
        return int(source_str[len(prefix):])
    else:
        # Try removing prefix length from the beginning
        if len(source_str) > len(prefix):
            return int(source_str[-len(source_str) + len(prefix):])
        return int(source_str)

def process_connectivity_for_all_neurons(synapse_file_path, target_neurons, pre_col, post_col, prefix):
    """Process connectivity for all neurons efficiently by reading file once"""
    
    print(f"   🔍 Scanning entire synapse table for {len(target_neurons)} neurons...")
    print("   This will take several minutes but is memory-efficient...")
    
    # Convert target neurons to partial format for efficient searching
    target_partials = {}
    for full_id in target_neurons:
        partial_id = convert_to_partial_root_id(full_id, prefix)
        target_partials[partial_id] = full_id
    
    print(f"   🎯 Looking for partial IDs: {list(target_partials.keys())[:10]}... (showing first 10)")
    
    # Store results for each neuron
    neuron_connections = {full_id: defaultdict(int) for full_id in target_neurons}
    neuron_neuropils = {full_id: defaultdict(set) for full_id in target_neurons}
    
    # Read file in chunks and collect connections
    chunk_count = 0
    total_matches = 0
    
    with gzip.open(synapse_file_path, 'rt') as f:
        # Read in chunks
        for chunk in pd.read_csv(f, chunksize=200000):
            chunk_count += 1
            
            # Filter chunk to only our target neurons
            matching_synapses = chunk[chunk[pre_col].isin(target_partials.keys())]
            
            if len(matching_synapses) > 0:
                total_matches += len(matching_synapses)
                
                # Process each synapse
                for _, synapse in matching_synapses.iterrows():
                    pre_partial = synapse[pre_col]
                    post_partial = synapse[post_col]
                    neuropil = synapse['neuropil']
                    
                    # Convert back to full IDs
                    pre_full_id = target_partials[pre_partial]
                    post_full_id = reconstruct_full_root_id(post_partial, prefix)
                    
                    if post_full_id:
                        neuron_connections[pre_full_id][post_full_id] += 1
                        # Clean neuropil value and add to set
                        if pd.notna(neuropil):
                            neuron_neuropils[pre_full_id][post_full_id].add(str(neuropil))
            
            # Progress update every 1000 chunks (~200M synapses)
            if chunk_count % 1000 == 0:
                print(f"      Processed {chunk_count} chunks ({chunk_count * 200000:,} synapses), "
                      f"found {total_matches:,} relevant synapses")
    
    print(f"   ✅ Completed! Found {total_matches:,} relevant synapses across {chunk_count} chunks")
    
    return neuron_connections, neuron_neuropils

def format_connectivity_results(connections_dict, neuropils_dict):
    """Format connectivity results for CSV output"""
    
    if not connections_dict:
        return {
            'output_cell_ids': '',
            'output_synapse_counts': '',
            'total_output_synapses': 0,
            'num_output_targets': 0,
            'top_targets': '',
            'neuropils_involved': ''
        }
    
    # Sort by synapse count
    sorted_connections = sorted(connections_dict.items(), key=lambda x: x[1], reverse=True)
    
    # Format outputs
    output_cell_ids = ';'.join([str(cell_id) for cell_id, _ in sorted_connections])
    output_synapse_counts = ';'.join([str(count) for _, count in sorted_connections])
    total_synapses = sum(connections_dict.values())
    num_targets = len(connections_dict)
    
    # Top 5 targets with synapse counts
    top_5 = sorted_connections[:5]
    top_targets = ';'.join([f"{cell_id}:{count}" for cell_id, count in top_5])
    
    # Neuropils involved - FIX THE ERROR HERE
    all_neuropils = set()
    for neuropil_set in neuropils_dict.values():
        for neuropil in neuropil_set:
            if pd.notna(neuropil) and neuropil != '':
                all_neuropils.add(str(neuropil))  # Ensure all are strings
    
    # Remove any empty strings and sort safely
    clean_neuropils = [n for n in all_neuropils if n.strip()]
    neuropils_str = ';'.join(sorted(clean_neuropils)) if clean_neuropils else ''
    
    return {
        'output_cell_ids': output_cell_ids,
        'output_synapse_counts': output_synapse_counts,
        'total_output_synapses': total_synapses,
        'num_output_targets': num_targets,
        'top_targets': top_targets,
        'neuropils_involved': neuropils_str
    }

def main():
    print("🔗 Adding Output Connectivity from Raw Synapse Table")
    print("=" * 60)
    
    # Load existing annotated data
    input_file = "selected_glomerulus_with_annotations.csv"
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        print("Please run the glomerulus annotation script first.")
        return
    
    print(f"📊 Loading existing data from {input_file}...")
    existing_data = pd.read_csv(input_file)
    print(f"   Found {len(existing_data)} neurons to analyze")
    
    # Find synapse table file
    data_dir = "data/flywire"
    synapse_file = None
    
    for filename in os.listdir(data_dir):
        if 'synapse' in filename.lower() and filename.endswith('.csv.gz'):
            synapse_file = filename
            break
    
    if not synapse_file:
        print("❌ No synapse table file found!")
        return
    
    synapse_path = os.path.join(data_dir, synapse_file)
    print(f"\n🧠 Using synapse table: {synapse_file}")
    print(f"   File size: {os.path.getsize(synapse_path) / (1024**3):.1f} GB")
    
    # Load small sample to get structure
    print("\n🔍 Analyzing synapse table structure...")
    with gzip.open(synapse_path, 'rt') as f:
        sample_df = pd.read_csv(f, nrows=1000)
    
    pre_col, post_col, prefix = extract_root_id_prefix_and_columns(sample_df)
    
    print(f"   ✅ Found columns: {pre_col}, {post_col}")
    print(f"   ✅ Root ID prefix: {prefix}")
    
    if not all([pre_col, post_col, prefix]):
        print("❌ Could not identify synapse table structure")
        return
    
    # Get list of all target neurons to search for
    target_neurons = existing_data['source_root_id'].tolist()
    
    # Process all connectivity in one pass through the file
    neuron_connections, neuron_neuropils = process_connectivity_for_all_neurons(
        synapse_path, target_neurons, pre_col, post_col, prefix
    )
    
    # Format results for each neuron
    print(f"\n📊 Formatting results for {len(existing_data)} neurons...")
    
    connectivity_info = []
    neurons_processed = 0
    
    for idx, row in existing_data.iterrows():
        source_root_id = row['source_root_id']
        
        connections = neuron_connections.get(source_root_id, {})
        neuropils = neuron_neuropils.get(source_root_id, {})
        
        conn_summary = format_connectivity_results(connections, neuropils)
        connectivity_info.append(conn_summary)
        
        neurons_processed += 1
        if neurons_processed % 100 == 0:
            print(f"   Processed {neurons_processed}/{len(existing_data)} neurons...")
    
    # Combine with existing data
    connectivity_df = pd.DataFrame(connectivity_info)
    result_df = pd.concat([existing_data, connectivity_df], axis=1)
    
    # Save results
    output_file = "selected_glomerulus_with_full_connectivity.csv"
    result_df.to_csv(output_file, index=False)
    
    # Print summary
    print("\n📊 FINAL CONNECTIVITY SUMMARY")
    print("=" * 45)
    
    total_neurons = len(result_df)
    neurons_with_outputs = (result_df['num_output_targets'] > 0).sum()
    
    print(f"✅ Total neurons analyzed: {total_neurons}")
    print(f"🔗 Neurons with outputs found: {neurons_with_outputs}")
    print(f"🚫 Neurons without outputs: {total_neurons - neurons_with_outputs}")
    
    if neurons_with_outputs > 0:
        connected_neurons = result_df[result_df['num_output_targets'] > 0]
        avg_targets = connected_neurons['num_output_targets'].mean()
        avg_synapses = connected_neurons['total_output_synapses'].mean()
        max_targets = connected_neurons['num_output_targets'].max()
        max_synapses = connected_neurons['total_output_synapses'].max()
        
        print(f"\n📈 CONNECTIVITY STATISTICS:")
        print(f"   Average targets per connected neuron: {avg_targets:.1f}")
        print(f"   Average synapses per connected neuron: {avg_synapses:.0f}")
        print(f"   Max targets from one neuron: {max_targets}")
        print(f"   Max synapses from one neuron: {max_synapses}")
        
        # Show top connected glomeruli (only those with known glomerulus)
        print(f"\n🎯 TOP CONNECTED GLOMERULI:")
        known_glom = result_df[(result_df['num_output_targets'] > 0) & (result_df['glomerulus'] != 'UNKNOWN')]
        if len(known_glom) > 0:
            glom_stats = known_glom.groupby('glomerulus').agg({
                'num_output_targets': ['count', 'mean'],
                'total_output_synapses': 'mean'
            }).round(1)
            
            # Flatten column names
            glom_stats.columns = ['neuron_count', 'avg_targets', 'avg_synapses']
            glom_stats = glom_stats.sort_values('avg_synapses', ascending=False)
            
            for glom in glom_stats.head(10).index:
                stats = glom_stats.loc[glom]
                print(f"   {glom}: {stats['neuron_count']:.0f} neurons, "
                      f"avg {stats['avg_targets']:.0f} targets, "
                      f"avg {stats['avg_synapses']:.0f} synapses")
        
        # Show most connected individual neurons
        print(f"\n🌟 MOST CONNECTED INDIVIDUAL NEURONS:")
        top_connected = result_df.nlargest(5, 'total_output_synapses')
        for _, row in top_connected.iterrows():
            print(f"   {row['glomerulus']} (ID: {row['source_root_id']}): "
                  f"{row['num_output_targets']} targets, {row['total_output_synapses']} synapses")
    
    print(f"\n✅ Complete connectivity analysis saved: {output_file}")
    print("   Columns added:")
    print("   • output_cell_ids: Downstream neuron IDs (semicolon-separated)")
    print("   • output_synapse_counts: Synapse counts for each target")
    print("   • total_output_synapses: Total synapses from this neuron")
    print("   • num_output_targets: Number of downstream partners")
    print("   • top_targets: Top 5 targets with counts (ID:count format)")
    print("   • neuropils_involved: Brain regions where connections occur")
    print("\n🎯 Ready for ORN→LN→ORN pathway analysis!")

if __name__ == "__main__":
    main()
