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

def extract_glomerulus_from_labels(label_text):
    """Extract glomerulus ID from processed_labels text"""
    if pd.isna(label_text):
        return None
    
    label_str = str(label_text).upper()
    
    # Common patterns for glomerulus identification
    glomerulus_patterns = [
        # ORN patterns (ORN_XX, ORN_XXX)
        'ORN_D[ACMLOV][0-9]+[A-Z]*',  # ORN_DL5, ORN_DA1, etc.
        'ORN_V[ACMLOV][0-9]+[A-Z]*',  # ORN_VA7m, ORN_VC1, etc.
        
        # Direct glomerulus names
        'DL[0-9]+[A-Z]*', 'DA[0-9]+[A-Z]*', 'DC[0-9]+[A-Z]*',
        'DM[0-9]+[A-Z]*', 'DP[0-9]+[A-Z]*', 'DO[0-9]+[A-Z]*',
        'VA[0-9]+[A-Z]*', 'VC[0-9]+[A-Z]*', 'VL[0-9]+[A-Z]*',
        'VM[0-9]+[A-Z]*', 'VP[0-9]+[A-Z]*',
        
        # AC/AB patterns
        'AC[0-9]+[A-Z]*', 'AB[0-9]+[A-Z]*', 'PB[0-9]+[A-Z]*'
    ]
    
    import re
    for pattern in glomerulus_patterns:
        match = re.search(pattern, label_str)
        if match:
            return match.group()
    
    return None

def main():
    print("🧠 Adding Glomerulus Information to Selected Neurons")
    print("=" * 60)
    
    # Load your existing cell ID file
    input_file = "data/flywire/selected_glomerulus_and_targets.csv"
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        print("Please run the previous script first to generate this file.")
        return
    
    print(f"📊 Loading existing data from {input_file}...")
    existing_data = pd.read_csv(input_file)
    print(f"   Found {len(existing_data)} neurons to annotate")
    
    # Define FlyWire data files to check
    data_dir = "data/flywire"
    annotation_files = {
        "consolidated_cell_types.csv.gz": "primary_type",
        "processed_labels.csv.gz": "processed_labels"
    }
    
    # Load annotation data
    print("\n📂 Loading FlyWire annotation files...")
    annotation_data = {}
    
    for filename, column in annotation_files.items():
        file_path = os.path.join(data_dir, filename)
        if os.path.exists(file_path):
            print(f"   Loading {filename}...")
            df = load_csv_gz(file_path)
            if df is not None:
                # Create lookup dictionary: root_id -> annotation
                annotation_data[filename] = df.set_index('root_id')[column].to_dict()
                print(f"   ✅ Loaded {len(df)} annotations from {filename}")
            else:
                print(f"   ❌ Failed to load {filename}")
        else:
            print(f"   ⚠️  File not found: {filename}")
    
    if not annotation_data:
        print("❌ No annotation files found! Check your data/flywire directory.")
        return
    
    # Add glomerulus information to existing data
    print("\n🔍 Looking up glomerulus information for each neuron...")
    
    glomerulus_info = []
    
    for idx, row in tqdm(existing_data.iterrows(), total=len(existing_data)):
        root_id = row['source_root_id']
        glomerulus = None
        source_file = None
        
        # Try to find glomerulus from consolidated_cell_types first
        if "consolidated_cell_types.csv.gz" in annotation_data:
            primary_type = annotation_data["consolidated_cell_types.csv.gz"].get(root_id)
            if primary_type and 'ORN_' in str(primary_type).upper():
                glomerulus = str(primary_type)
                source_file = "consolidated_cell_types.csv.gz"
        
        # If not found, try processed_labels
        if glomerulus is None and "processed_labels.csv.gz" in annotation_data:
            processed_label = annotation_data["processed_labels.csv.gz"].get(root_id)
            extracted_glomerulus = extract_glomerulus_from_labels(processed_label)
            if extracted_glomerulus:
                glomerulus = extracted_glomerulus
                source_file = "processed_labels.csv.gz"
        
        # If still not found, try to extract from the original label
        if glomerulus is None:
            original_label = row.get('label', '')
            extracted_glomerulus = extract_glomerulus_from_labels(original_label)
            if extracted_glomerulus:
                glomerulus = extracted_glomerulus
                source_file = "original_label"
        
        glomerulus_info.append({
            'glomerulus': glomerulus if glomerulus else 'UNKNOWN',
            'glomerulus_source': source_file if source_file else 'NOT_FOUND'
        })
    
    # Add glomerulus information to the dataframe
    glomerulus_df = pd.DataFrame(glomerulus_info)
    result_df = pd.concat([existing_data, glomerulus_df], axis=1)
    
    # Save enhanced results
    output_file = "selected_glomerulus_with_annotations.csv"
    result_df.to_csv(output_file, index=False)
    
    # Print summary statistics
    print("\n📊 SUMMARY RESULTS")
    print("=" * 40)
    print(f"✅ Total neurons processed: {len(result_df)}")
    print(f"✅ Glomerulus identified: {(result_df['glomerulus'] != 'UNKNOWN').sum()}")
    print(f"❌ Unknown glomerulus: {(result_df['glomerulus'] == 'UNKNOWN').sum()}")
    
    print(f"\n📁 Results saved to: {output_file}")
    
    # Show breakdown by glomerulus
    print("\n🎯 Glomerulus Breakdown:")
    glomerulus_counts = result_df['glomerulus'].value_counts()
    for glom, count in glomerulus_counts.head(15).items():
        if glom != 'UNKNOWN':
            print(f"   {glom}: {count} neurons")
    
    if 'UNKNOWN' in glomerulus_counts:
        print(f"   UNKNOWN: {glomerulus_counts['UNKNOWN']} neurons")
    
    # Show source breakdown
    print("\n📂 Data Source Breakdown:")
    source_counts = result_df['glomerulus_source'].value_counts()
    for source, count in source_counts.items():
        print(f"   {source}: {count} neurons")
    
    print(f"\n✅ Enhanced CSV file created: {output_file}")
    print("   New columns added: 'glomerulus', 'glomerulus_source'")

if __name__ == "__main__":
    main()
