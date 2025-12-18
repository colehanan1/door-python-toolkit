import os
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import re

def load_csv_gz(file_path):
    """Load compressed CSV file"""
    try:
        with gzip.open(file_path, 'rt') as f:
            return pd.read_csv(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def extract_root_id_prefix_and_columns(sample_df):
    """Extract the root ID prefix and identify correct column names"""
    columns = sample_df.columns.tolist()
    pre_col = None
    post_col = None
    prefix = None
    
    for col in columns:
        if 'pre_root_id' in col:
            pre_col = col
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
        if len(source_str) > len(prefix):
            return int(source_str[-len(source_str) + len(prefix):])
        return int(source_str)

def load_annotation_data(data_dir):
    """Load cell type annotation files"""
    annotation_files = {
        "consolidated_cell_types.csv.gz": "primary_type",
        "processed_labels.csv.gz": "processed_labels"
    }
    
    annotations = {}
    for filename, column in annotation_files.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            print(f"   Loading {filename}...")
            df = load_csv_gz(file_path)
            if df is not None:
                annotations[filename] = df.set_index('root_id')[column].to_dict()
                print(f"   ✅ Loaded {len(df):,} annotations")
    
    return annotations

def get_cell_type(root_id, annotations):
    """Get cell type for a given root_id"""
    if "consolidated_cell_types.csv.gz" in annotations:
        cell_type = annotations["consolidated_cell_types.csv.gz"].get(root_id)
        if cell_type and pd.notna(cell_type):
            return str(cell_type)
    
    if "processed_labels.csv.gz" in annotations:
        cell_type = annotations["processed_labels.csv.gz"].get(root_id)
        if cell_type and pd.notna(cell_type):
            return str(cell_type)
    
    return "UNKNOWN"

def classify_cell_type(cell_type_str):
    """Classify cell type into broad category"""
    if pd.isna(cell_type_str) or cell_type_str == 'UNKNOWN':
        return 'UNKNOWN'
    
    cell_type = str(cell_type_str).strip()
    
    if re.match(r'^ORN_', cell_type):
        return 'ORN'
    if cell_type.endswith('PN'):
        return 'Projection_Neuron'
    if re.match(r'^[lvi]*LN|^LN[0-9]', cell_type):
        return 'Local_Neuron'
    if re.match(r'^KC', cell_type):
        return 'Kenyon_Cell'
    if re.match(r'^MBON', cell_type):
        return 'MBON'
    
    return 'Other'

def process_focused_downstream(source_neurons, synapse_file_path, pre_col, post_col, prefix, 
                               annotations, level_name, filter_targets=None):
    """Find downstream connections, optionally filtering target types"""
    
    print(f"\n   🔍 Level {level_name}: Processing {len(source_neurons):,} source neurons...")
    if filter_targets:
        print(f"      Filtering to only: {filter_targets}")
    
    target_partials = {}
    for full_id in source_neurons:
        partial_id = convert_to_partial_root_id(full_id, prefix)
        target_partials[partial_id] = full_id
    
    neuron_connections = {full_id: defaultdict(int) for full_id in source_neurons}
    
    chunk_count = 0
    total_matches = 0
    
    with gzip.open(synapse_file_path, 'rt') as f:
        for chunk in pd.read_csv(f, chunksize=200000):
            chunk_count += 1
            matching_synapses = chunk[chunk[pre_col].isin(target_partials.keys())]
            
            if len(matching_synapses) > 0:
                total_matches += len(matching_synapses)
                
                for _, synapse in matching_synapses.iterrows():
                    pre_partial = synapse[pre_col]
                    post_partial = synapse[post_col]
                    
                    pre_full_id = target_partials[pre_partial]
                    post_full_id = reconstruct_full_root_id(post_partial, prefix)
                    
                    if post_full_id:
                        neuron_connections[pre_full_id][post_full_id] += 1
            
            if chunk_count % 1000 == 0:
                print(f"      Chunk {chunk_count}: {total_matches:,} synapses found")
    
    print(f"   ✅ Level {level_name}: Found {total_matches:,} synapses")
    
    # Format results with filtering
    results = []
    all_downstream_ids = set()
    filtered_count = 0
    
    for source_id, connections in neuron_connections.items():
        if connections:
            for target_id, syn_count in connections.items():
                target_cell_type = get_cell_type(target_id, annotations)
                target_category = classify_cell_type(target_cell_type)
                
                # Apply filter if specified
                if filter_targets and target_category not in filter_targets:
                    filtered_count += 1
                    continue
                
                results.append({
                    'source_root_id': source_id,
                    'target_root_id': target_id,
                    'target_cell_type': target_cell_type,
                    'target_category': target_category,
                    'synapse_count': syn_count
                })
                all_downstream_ids.add(target_id)
    
    if filter_targets and filtered_count > 0:
        print(f"   🔽 Filtered out {filtered_count:,} connections to non-target cell types")
    
    return pd.DataFrame(results), all_downstream_ids

def main():
    print("🧠 FOCUSED Interglomerular Cross-Talk Pathway Analysis")
    print("=" * 70)
    print("Strategy: ORN → LN/PN (Level1) → ORN/PN (Level2) ONLY")
    print("=" * 70)
    
    # Load level 0 data
    level0_file = "output_cells_annotated.csv"
    if not os.path.exists(level0_file):
        print(f"❌ Error: {level0_file} not found!")
        return
    
    print(f"\n📊 Loading Level 0 (ORN outputs)...")
    level0_data = pd.read_csv(level0_file)
    print(f"   Total Level 0 connections: {len(level0_data):,}")
    
    # Show breakdown
    level0_breakdown = level0_data['broad_category'].value_counts()
    print(f"\n   📈 Level 0 Target Breakdown:")
    for cat, count in level0_breakdown.items():
        print(f"      {cat}: {count:,}")
    
    # Filter Level 1 to only LNs and PNs
    level1_relevant = level0_data[
        level0_data['broad_category'].isin(['Local_Neuron', 'Projection_Neuron'])
    ]
    level1_neurons = level1_relevant['output_root_id'].unique().tolist()
    
    print(f"\n   🎯 FOCUSING on biologically relevant Level 1 neurons:")
    print(f"      Local Neurons (LNs): {(level0_data['broad_category'] == 'Local_Neuron').sum():,} connections")
    print(f"      Projection Neurons (PNs): {(level0_data['broad_category'] == 'Projection_Neuron').sum():,} connections")
    print(f"      Total Level 1 neurons to trace: {len(level1_neurons):,}")
    
    # Setup
    data_dir = "data/flywire"
    synapse_file = None
    for filename in os.listdir(data_dir):
        if 'synapse' in filename.lower() and filename.endswith('.csv.gz'):
            synapse_file = filename
            break
    
    if not synapse_file:
        print("❌ No synapse table found!")
        return
    
    synapse_path = os.path.join(data_dir, synapse_file)
    
    # Get synapse table structure
    print(f"\n🔍 Analyzing synapse table...")
    with gzip.open(synapse_path, 'rt') as f:
        sample_df = pd.read_csv(f, nrows=1000)
    
    pre_col, post_col, prefix = extract_root_id_prefix_and_columns(sample_df)
    print(f"   ✅ Structure identified")
    
    # Load annotations
    print(f"\n📂 Loading cell type annotations...")
    annotations = load_annotation_data(data_dir)
    
    # Process Level 1 → Level 2 (FILTERED)
    print(f"\n{'='*70}")
    print(f"LEVEL 1→2: LN/PN outputs (FILTERED to ORN/PN/LN targets only)")
    print(f"{'='*70}")
    
    level1_to_2, level2_neurons = process_focused_downstream(
        level1_neurons, synapse_path, pre_col, post_col, prefix, annotations, 
        "1→2", filter_targets=['ORN', 'Projection_Neuron', 'Local_Neuron']
    )
    
    print(f"\n   📊 Level 1→2 Results:")
    print(f"      Total connections: {len(level1_to_2):,}")
    print(f"      Target categories:")
    for cat, count in level1_to_2['target_category'].value_counts().items():
        print(f"         {cat}: {count:,}")
    
    # Build complete pathway database - FIXED MERGE
    print(f"\n🔗 Building complete cross-talk pathway database...")
    
    # Create lookup dictionaries from level 0 to avoid merge issues
    level0_lookup = {}
    for _, row in level0_data.iterrows():
        level0_lookup[row['output_root_id']] = {
            'orn_root_id': row['source_root_id'],
            'orn_label': row['source_label'],
            'orn_glomerulus': row['source_glomerulus'],
            'level1_cell_type': row['output_cell_type'],
            'level1_category': row['broad_category'],
            'synapse_count_step1': row['synapse_count']
        }
    
    # Build complete pathway rows
    pathway_rows = []
    for _, row in level1_to_2.iterrows():
        level1_id = row['source_root_id']
        level2_id = row['target_root_id']
        
        # Get level 0 info
        if level1_id in level0_lookup:
            level0_info = level0_lookup[level1_id]
            
            pathway_rows.append({
                'orn_root_id': level0_info['orn_root_id'],
                'orn_label': level0_info['orn_label'],
                'orn_glomerulus': level0_info['orn_glomerulus'],
                'level1_root_id': level1_id,
                'level1_cell_type': level0_info['level1_cell_type'],
                'level1_category': level0_info['level1_category'],
                'level2_root_id': level2_id,
                'level2_cell_type': row['target_cell_type'],
                'level2_category': row['target_category'],
                'synapse_count_step1': level0_info['synapse_count_step1'],
                'synapse_count_step2': row['synapse_count']
            })
    
    crosstalk_pathways = pd.DataFrame(pathway_rows)
    
    # Sort by source glomerulus
    crosstalk_pathways = crosstalk_pathways.sort_values(['orn_glomerulus', 'level1_cell_type'])
    
    # Save complete pathway file
    output_file = "interglomerular_crosstalk_pathways.csv"
    crosstalk_pathways.to_csv(output_file, index=False)
    
    # Analysis
    print(f"\n📊 INTERGLOMERULAR CROSS-TALK ANALYSIS")
    print("=" * 50)
    
    print(f"✅ Total 2-step pathways: {len(crosstalk_pathways):,}")
    print(f"✅ Source ORN glomeruli: {crosstalk_pathways['orn_glomerulus'].nunique()}")
    print(f"✅ Intermediate neurons (Level 1): {crosstalk_pathways['level1_root_id'].nunique():,}")
    print(f"✅ Final target neurons (Level 2): {crosstalk_pathways['level2_root_id'].nunique():,}")
    
    # Pathway type breakdown
    print(f"\n🔬 PATHWAY TYPE BREAKDOWN:")
    pathway_types = crosstalk_pathways.groupby(['level1_category', 'level2_category']).size().sort_values(ascending=False)
    for (intermediate, target), count in pathway_types.items():
        print(f"   ORN → {intermediate} → {target}: {count:,} pathways")
    
    # ORN→LN→ORN (LATERAL INHIBITION - KEY!)
    print(f"\n🎯 ORN→LN→ORN CROSS-TALK PATHWAYS (Lateral Inhibition):")
    orn_ln_orn = crosstalk_pathways[
        (crosstalk_pathways['level1_category'] == 'Local_Neuron') &
        (crosstalk_pathways['level2_category'] == 'ORN')
    ]
    print(f"   Total pathways: {len(orn_ln_orn):,}")
    orn_ln_orn.to_csv("crosstalk_ORN_LN_ORN.csv", index=False)
    
    if len(orn_ln_orn) > 0:
        print(f"\n   🔬 Top 20 ORN→LN→ORN Cross-Talk Pathways:")
        top_crosstalk = orn_ln_orn.nlargest(20, 'synapse_count_step2')
        for _, row in top_crosstalk.iterrows():
            print(f"      {row['orn_glomerulus']} → {row['level1_cell_type']} → {row['level2_cell_type']}")
            print(f"         (Step1: {row['synapse_count_step1']} syn, Step2: {row['synapse_count_step2']} syn)")
    
    # ORN→LN→PN (LATERAL MODULATION)
    print(f"\n🎯 ORN→LN→PN PATHWAYS (Lateral Modulation of Projection Neurons):")
    orn_ln_pn = crosstalk_pathways[
        (crosstalk_pathways['level1_category'] == 'Local_Neuron') &
        (crosstalk_pathways['level2_category'] == 'Projection_Neuron')
    ]
    print(f"   Total pathways: {len(orn_ln_pn):,}")
    orn_ln_pn.to_csv("crosstalk_ORN_LN_PN.csv", index=False)
    
    if len(orn_ln_pn) > 0:
        print(f"\n   🔬 Top 20 ORN→LN→PN Modulation Pathways:")
        top_modulation = orn_ln_pn.nlargest(20, 'synapse_count_step2')
        for _, row in top_modulation.iterrows():
            print(f"      {row['orn_glomerulus']} → {row['level1_cell_type']} → {row['level2_cell_type']}")
            print(f"         (Step1: {row['synapse_count_step1']} syn, Step2: {row['synapse_count_step2']} syn)")
    
    # ORN→PN→ORN (DIRECT PN FEEDBACK)
    print(f"\n🎯 ORN→PN→ORN/LN PATHWAYS (Direct PN Feedback):")
    orn_pn_feedback = crosstalk_pathways[
        (crosstalk_pathways['level1_category'] == 'Projection_Neuron') &
        (crosstalk_pathways['level2_category'].isin(['ORN', 'Local_Neuron']))
    ]
    print(f"   Total pathways: {len(orn_pn_feedback):,}")
    orn_pn_feedback.to_csv("crosstalk_ORN_PN_feedback.csv", index=False)
    
    # Cross-talk matrix (glomerulus to glomerulus via LN)
    print(f"\n📊 GLOMERULAR CROSS-TALK MATRIX:")
    
    def extract_glomerulus(cell_type):
        if pd.isna(cell_type):
            return 'UNKNOWN'
        match = re.match(r'ORN_([A-Z0-9]+)', str(cell_type))
        return match.group(1) if match else str(cell_type)
    
    orn_ln_orn_copy = orn_ln_orn.copy()
    orn_ln_orn_copy['source_glom'] = orn_ln_orn_copy['orn_glomerulus'].str.replace('ORN_', '')
    orn_ln_orn_copy['target_glom'] = orn_ln_orn_copy['level2_cell_type'].apply(extract_glomerulus)
    
    crosstalk_matrix = orn_ln_orn_copy.groupby(['source_glom', 'target_glom'])['synapse_count_step2'].sum().reset_index()
    crosstalk_matrix = crosstalk_matrix.sort_values('synapse_count_step2', ascending=False)
    crosstalk_matrix.to_csv("crosstalk_matrix_glomerulus.csv", index=False)
    
    print(f"   Unique source glomeruli: {crosstalk_matrix['source_glom'].nunique()}")
    print(f"   Unique target glomeruli: {crosstalk_matrix['target_glom'].nunique()}")
    print(f"\n   Top 25 Glomerulus→Glomerulus Cross-Talk Pairs:")
    for idx, row in crosstalk_matrix.head(25).iterrows():
        print(f"      {row['source_glom']} → {row['target_glom']}: {int(row['synapse_count_step2'])} synapses")
    
    print(f"\n✅ FILES CREATED (FOCUSED FOR CROSS-TALK):")
    print(f"   📄 interglomerular_crosstalk_pathways.csv: {len(crosstalk_pathways):,} complete 2-step pathways")
    print(f"   📄 crosstalk_ORN_LN_ORN.csv: {len(orn_ln_orn):,} lateral inhibition pathways ⭐")
    print(f"   📄 crosstalk_ORN_LN_PN.csv: {len(orn_ln_pn):,} lateral modulation pathways ⭐")
    print(f"   📄 crosstalk_ORN_PN_feedback.csv: {len(orn_pn_feedback):,} PN feedback pathways")
    print(f"   📄 crosstalk_matrix_glomerulus.csv: Glomerulus cross-talk strength matrix ⭐")
    
    print(f"\n🎯 READY FOR CROSS-TALK VISUALIZATION!")
    print(f"   Focus on crosstalk_ORN_LN_ORN.csv for lateral inhibition analysis")
    print(f"   Use crosstalk_matrix_glomerulus.csv for network visualization")

if __name__ == "__main__":
    main()
