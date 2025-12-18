import os
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_csv_gz(file_path):
    """Load compressed CSV file"""
    try:
        with gzip.open(file_path, 'rt') as f:
            return pd.read_csv(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

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
                # Create lookup: root_id -> cell_type
                annotations[filename] = df.set_index('root_id')[column].to_dict()
                print(f"   ✅ Loaded {len(df):,} cell type annotations from {column}")
    
    return annotations

def get_cell_type(root_id, annotations):
    """Get cell type for a given root_id from multiple sources"""
    
    # Try consolidated_cell_types first
    if "consolidated_cell_types.csv.gz" in annotations:
        cell_type = annotations["consolidated_cell_types.csv.gz"].get(root_id)
        if cell_type and pd.notna(cell_type):
            return str(cell_type), "consolidated_cell_types"
    
    # Try processed_labels second
    if "processed_labels.csv.gz" in annotations:
        cell_type = annotations["processed_labels.csv.gz"].get(root_id)
        if cell_type and pd.notna(cell_type):
            return str(cell_type), "processed_labels"
    
    return "UNKNOWN", "NOT_FOUND"

def main():
    print("🔬 Creating Flattened Output Cells CSV with Cell Type Annotations")
    print("=" * 70)
    
    # Load the connectivity file
    input_file = "selected_glomerulus_with_full_connectivity.csv"
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return
    
    print(f"📊 Loading connectivity data from {input_file}...")
    connectivity_data = pd.read_csv(input_file)
    print(f"   Found {len(connectivity_data)} source neurons")
    
    # Load annotation data for cell type lookup
    data_dir = "data/flywire"
    print(f"\n📂 Loading cell type annotations from {data_dir}...")
    annotations = load_annotation_data(data_dir)
    
    if not annotations:
        print("❌ No annotation files found!")
        return
    
    # Flatten the data - one row per output cell
    print("\n🔄 Flattening data to one row per output cell...")
    
    output_rows = []
    
    for idx, row in tqdm(connectivity_data.iterrows(), total=len(connectivity_data), desc="Processing neurons"):
        source_root_id = row['source_root_id']
        source_label = row.get('label', 'UNKNOWN')
        source_glomerulus = row.get('glomerulus', 'UNKNOWN')
        
        # Parse output cells
        output_ids = row.get('output_cell_ids', '')
        output_syn_counts = row.get('output_synapse_counts', '')
        
        if not output_ids or pd.isna(output_ids) or output_ids == '':
            continue
        
        # Split into lists
        output_id_list = str(output_ids).split(';')
        output_syn_list = str(output_syn_counts).split(';')
        
        # Create one row per output cell
        for output_id, syn_count in zip(output_id_list, output_syn_list):
            if output_id and output_id.strip():
                try:
                    output_root_id = int(output_id.strip())
                    synapse_count = int(syn_count.strip())
                    
                    # Get cell type for output cell
                    cell_type, cell_type_source = get_cell_type(output_root_id, annotations)
                    
                    output_rows.append({
                        'source_root_id': source_root_id,
                        'source_label': source_label,
                        'source_glomerulus': source_glomerulus,
                        'output_root_id': output_root_id,
                        'output_cell_type': cell_type,
                        'cell_type_source': cell_type_source,
                        'synapse_count': synapse_count
                    })
                except ValueError:
                    continue
    
    # Create DataFrame
    print(f"\n📋 Creating flattened output cells DataFrame...")
    output_df = pd.DataFrame(output_rows)
    
    if len(output_df) == 0:
        print("❌ No output cells found!")
        return
    
    # Sort by glomerulus, then by output cell ID
    print("   Sorting by glomerulus and cell ID...")
    output_df = output_df.sort_values(['source_glomerulus', 'output_root_id'])
    
    # Save results
    output_file = "output_cells_with_types.csv"
    output_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n📊 OUTPUT CELLS SUMMARY")
    print("=" * 45)
    print(f"✅ Total output connections: {len(output_df):,}")
    print(f"✅ Unique output cells: {output_df['output_root_id'].nunique():,}")
    print(f"✅ Source glomeruli: {output_df['source_glomerulus'].nunique()}")
    
    # Cell type statistics
    cell_types_found = (output_df['output_cell_type'] != 'UNKNOWN').sum()
    print(f"\n🔬 CELL TYPE ANNOTATION:")
    print(f"   Cells with type identified: {cell_types_found:,} ({100*cell_types_found/len(output_df):.1f}%)")
    print(f"   Unknown cell types: {len(output_df) - cell_types_found:,}")
    
    # Show top output cell types
    print(f"\n🎯 TOP OUTPUT CELL TYPES:")
    top_types = output_df[output_df['output_cell_type'] != 'UNKNOWN']['output_cell_type'].value_counts().head(15)
    for cell_type, count in top_types.items():
        avg_synapses = output_df[output_df['output_cell_type'] == cell_type]['synapse_count'].mean()
        print(f"   {cell_type}: {count:,} connections, avg {avg_synapses:.0f} synapses")
    
    # Show glomerulus-specific breakdown
    print(f"\n🧬 CONNECTIONS BY SOURCE GLOMERULUS:")
    glom_stats = output_df.groupby('source_glomerulus').agg({
        'output_root_id': 'count',
        'synapse_count': 'sum'
    }).sort_values('synapse_count', ascending=False)
    
    for glom in glom_stats.head(10).index:
        if glom != 'UNKNOWN':
            stats = glom_stats.loc[glom]
            print(f"   {glom}: {stats['output_root_id']:,} connections, "
                  f"{stats['synapse_count']:,} total synapses")
    
    # Check for DL5_adPN specifically
    print(f"\n🔍 CHECKING FOR DL5_adPN (Or7a target):")
    dl5_adpn = output_df[output_df['output_cell_type'].str.contains('DL5_adPN', case=False, na=False)]
    if len(dl5_adpn) > 0:
        print(f"   ✅ Found {len(dl5_adpn)} DL5_adPN connections!")
        for _, row in dl5_adpn.iterrows():
            print(f"      Source: {row['source_glomerulus']} (ID: {row['source_root_id']})")
            print(f"      Target: {row['output_cell_type']} (ID: {row['output_root_id']})")
            print(f"      Synapses: {row['synapse_count']}")
            print()
    else:
        print("   ❌ No DL5_adPN cells found in output types")
        print("   Searching for similar patterns...")
        dl5_related = output_df[output_df['output_cell_type'].str.contains('DL5', case=False, na=False)]
        if len(dl5_related) > 0:
            print(f"   Found {len(dl5_related)} DL5-related cells:")
            for cell_type in dl5_related['output_cell_type'].unique()[:10]:
                print(f"      • {cell_type}")
    
    print(f"\n✅ Flattened output cells file saved: {output_file}")
    print("   Format: One row per connection")
    print("   Columns:")
    print("   • source_root_id: Source neuron ID")
    print("   • source_label: Original label (e.g., Or7a)")
    print("   • source_glomerulus: Glomerulus ID (e.g., ORN_DL5)")
    print("   • output_root_id: Downstream neuron ID")
    print("   • output_cell_type: Cell type annotation (e.g., DL5_adPN, lLN2, etc.)")
    print("   • cell_type_source: Where annotation came from")
    print("   • synapse_count: Number of synapses in connection")
    print("\n🎯 Ready for ORN→LN→ORN pathway analysis!")

if __name__ == "__main__":
    main()
