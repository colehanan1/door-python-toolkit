import os
import pandas as pd
import re

def classify_cell_type(cell_type_str):
    """Classify cell type into broad categories with detailed descriptions"""
    
    if pd.isna(cell_type_str) or cell_type_str == 'UNKNOWN':
        return {
            'broad_category': 'UNKNOWN',
            'functional_role': 'Unknown',
            'description': 'No annotation available'
        }
    
    cell_type = str(cell_type_str).strip()
    
    # Olfactory Receptor Neurons
    if re.match(r'^ORN_', cell_type):
        glom = cell_type.replace('ORN_', '')
        return {
            'broad_category': 'ORN',
            'functional_role': 'Sensory Input',
            'description': f'Olfactory Receptor Neuron targeting glomerulus {glom}'
        }
    
    # Projection Neurons - FIXED: Check if ends with PN (most general rule)
    if cell_type.endswith('PN'):
        # Parse glomerulus and projection tract
        if '_' in cell_type:
            parts = cell_type.split('_')
            glom = parts[0]
            projection = '_'.join(parts[1:])  # Handle multi-part like VP1l+_lvPN
            
            # Map projection tract abbreviations
            proj_map = {
                'adPN': 'anterodorsal projection neuron',
                'lPN': 'lateral projection neuron',
                'vPN': 'ventral projection neuron',
                'lvPN': 'lateroventral projection neuron',
                'lv2PN': 'lateroventral type 2 projection neuron',
                'l2PN': 'lateral type 2 projection neuron',
                'ilPN': 'inferolateral projection neuron',
                'imPN': 'inferomedial projection neuron'
            }
            proj_desc = proj_map.get(projection, projection)
            
            return {
                'broad_category': 'Projection_Neuron',
                'functional_role': 'AL→MB/LH Output',
                'description': f'Projection neuron from {glom} glomerulus via {proj_desc} to mushroom body and lateral horn'
            }
        else:
            return {
                'broad_category': 'Projection_Neuron',
                'functional_role': 'AL→MB/LH Output',
                'description': f'Projection neuron {cell_type} from antennal lobe to mushroom body and lateral horn'
            }
    
    # Local Neurons
    if re.match(r'^[lvi]*LN|^LN[0-9]', cell_type):
        ln_type = cell_type.split('_')[0] if '_' in cell_type else cell_type
        return {
            'broad_category': 'Local_Neuron',
            'functional_role': 'Lateral/Interglomerular Processing',
            'description': f'Antennal lobe local interneuron {ln_type} mediating cross-talk between glomeruli'
        }
    
    # Kenyon Cells
    if re.match(r'^KC', cell_type):
        return {
            'broad_category': 'Kenyon_Cell',
            'functional_role': 'Mushroom Body Intrinsic',
            'description': 'Mushroom body Kenyon cell for olfactory learning and memory'
        }
    
    # Mushroom Body Output Neurons
    if re.match(r'^MBON', cell_type):
        return {
            'broad_category': 'MBON',
            'functional_role': 'Mushroom Body Output',
            'description': 'Mushroom body output neuron encoding learned behavioral choices'
        }
    
    # Dopaminergic neurons
    if re.match(r'^PAM|^PPL|^PAL', cell_type):
        cluster_map = {
            'PAM': 'Protocerebral Anterior Medial (reward)',
            'PPL': 'Protocerebral Posterior Lateral (punishment/reward)',
            'PAL': 'Protocerebral Anterior Lateral'
        }
        cluster = cell_type[:3]
        cluster_desc = cluster_map.get(cluster, cluster)
        return {
            'broad_category': 'DAN',
            'functional_role': 'Dopaminergic Neuromodulation',
            'description': f'Dopaminergic neuron {cell_type} from {cluster_desc} cluster'
        }
    
    # Lateral Horn neurons
    if re.match(r'^LH', cell_type):
        return {
            'broad_category': 'Lateral_Horn',
            'functional_role': 'Innate Olfactory Behavior',
            'description': f'Lateral horn neuron {cell_type} for innate valence and behavior'
        }
    
    # Antennal lobe neurons (other)
    if re.match(r'^mAL|^AL-|^ALIN|^ALON|^ALBN', cell_type):
        return {
            'broad_category': 'AL_Interneuron',
            'functional_role': 'Antennal Lobe Modulation',
            'description': f'Antennal lobe interneuron {cell_type} for local processing'
        }
    
    # Central Complex neurons
    cx_patterns = ['Delta', 'EPG', 'PEN', 'PFL', 'PFN', 'PFR', 'PFGs', 'FB', 'EB', 'PB', 'NO', 
                   'ExR', 'ER', 'FC', 'FR', 'FS', 'PEG']
    for pattern in cx_patterns:
        if pattern in cell_type:
            return {
                'broad_category': 'Central_Complex',
                'functional_role': 'Navigation/Spatial Memory',
                'description': f'Central complex neuron {cell_type} for head direction, navigation, and motor control'
            }
    
    # Visual system
    visual_patterns = ['LC', 'LT', 'LPLC', 'LLPC', 'LPC', 'MC', 'aMe', 'CT', 'HSN', 'HSE', 
                      'HSS', 'VS', 'DCH', 'VCH', 'Li', 'HBeyelet']
    for pattern in visual_patterns:
        if cell_type.startswith(pattern):
            return {
                'broad_category': 'Visual_System',
                'functional_role': 'Visual Processing',
                'description': f'Optic lobe neuron {cell_type} for motion and visual feature detection'
            }
    
    # Descending neurons
    if re.match(r'^DN[abgdp]|^DNES|^Giant_Fiber|^MDN', cell_type):
        return {
            'broad_category': 'Descending_Neuron',
            'functional_role': 'Brain→VNC Motor Command',
            'description': f'Descending neuron {cell_type} transmitting motor commands to ventral nerve cord'
        }
    
    # Neuromodulatory neurons
    if re.match(r'^OA-|^5-HT', cell_type):
        return {
            'broad_category': 'Neuromodulatory',
            'functional_role': 'Neuromodulation',
            'description': f'Neuromodulatory neuron {cell_type} (octopamine/serotonin)'
        }
    
    # Higher order protocerebrum neurons
    protocerebrum_regions = {
        'SMP': 'Superior Medial Protocerebrum',
        'SLP': 'Superior Lateral Protocerebrum',
        'SIP': 'Superior Intermediate Protocerebrum',
        'AVLP': 'Anterior Ventrolateral Protocerebrum',
        'PVLP': 'Posterior Ventrolateral Protocerebrum',
        'PLP': 'Posterior Lateral Protocerebrum',
        'ATL': 'Antler',
        'AOTU': 'Anterior Optic Tubercle',
        'CRE': 'Crepine',
        'LAL': 'Lateral Accessory Lobe',
        'BU': 'Bulb',
        'WED': 'Wedge',
        'VES': 'Vest',
        'SAD': 'Saddle',
        'CL': 'Clamp',
        'IB': 'Inferior Bridge',
        'PS': 'Posterior Slope'
    }
    
    for region, full_name in protocerebrum_regions.items():
        if cell_type.startswith(region):
            return {
                'broad_category': f'{region}_Neuron',
                'functional_role': 'Higher Order Integration',
                'description': f'{full_name} neuron {cell_type} for multimodal integration'
            }
    
    # Default for unclassified
    return {
        'broad_category': 'Other',
        'functional_role': 'Unclassified',
        'description': f'Neuron type {cell_type}'
    }

def main():
    print("🏷️  Adding Cell Type Categories and Descriptions")
    print("=" * 60)
    
    # Load the output cells file
    input_file = "output_cells_with_types.csv"
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return
    
    print(f"📊 Loading output cells data from {input_file}...")
    output_cells = pd.read_csv(input_file)
    print(f"   Found {len(output_cells):,} output connections")
    
    # Add classification for each cell type
    print(f"\n🔍 Classifying {output_cells['output_cell_type'].nunique()} unique cell types...")
    
    classifications = []
    for cell_type in output_cells['output_cell_type']:
        classification = classify_cell_type(cell_type)
        classifications.append(classification)
    
    classification_df = pd.DataFrame(classifications)
    
    # Add to output cells dataframe
    result_df = pd.concat([output_cells, classification_df], axis=1)
    
    # Reorder columns for better readability
    column_order = [
        'source_root_id', 'source_label', 'source_glomerulus',
        'output_root_id', 'output_cell_type', 'broad_category', 
        'functional_role', 'description',
        'synapse_count', 'cell_type_source'
    ]
    result_df = result_df[column_order]
    
    # Sort by source glomerulus, then output cell ID
    result_df = result_df.sort_values(['source_glomerulus', 'output_root_id'])
    
    # Save annotated file
    output_file = "output_cells_annotated.csv"
    result_df.to_csv(output_file, index=False)
    
    # Print comprehensive summary
    print("\n📊 CELL TYPE CLASSIFICATION SUMMARY")
    print("=" * 50)
    
    print(f"✅ Total connections analyzed: {len(result_df):,}")
    print(f"✅ Unique output cell types: {result_df['output_cell_type'].nunique()}")
    print(f"✅ Source glomeruli: {result_df['source_glomerulus'].nunique()}")
    
    # Breakdown by broad category
    print(f"\n🧬 CONNECTIONS BY CELL TYPE CATEGORY:")
    category_stats = result_df.groupby('broad_category').agg({
        'output_root_id': 'count',
        'synapse_count': ['sum', 'mean']
    }).round(1)
    
    category_stats.columns = ['num_connections', 'total_synapses', 'avg_synapses']
    category_stats = category_stats.sort_values('total_synapses', ascending=False)
    
    for category in category_stats.index:
        stats = category_stats.loc[category]
        pct = 100 * stats['num_connections'] / len(result_df)
        print(f"   {category}: {int(stats['num_connections']):,} connections ({pct:.1f}%), "
              f"{int(stats['total_synapses']):,} synapses, avg {stats['avg_synapses']:.0f}")
    
    # Show key olfactory connections
    print(f"\n🔬 KEY OLFACTORY PATHWAY CONNECTIONS:")
    olfactory_categories = ['Projection_Neuron', 'Local_Neuron', 'ORN', 'AL_Interneuron']
    olfactory = result_df[result_df['broad_category'].isin(olfactory_categories)]
    
    if len(olfactory) > 0:
        print(f"   Total olfactory connections: {len(olfactory):,}")
        print(f"   Total olfactory synapses: {olfactory['synapse_count'].sum():,}")
        
        # Breakdown by type
        olfactory_by_type = olfactory.groupby('broad_category')['synapse_count'].agg(['count', 'sum'])
        for cat in olfactory_by_type.index:
            stats = olfactory_by_type.loc[cat]
            print(f"      {cat}: {int(stats['count']):,} connections, "
                  f"{int(stats['sum']):,} synapses")
    
    # Check for DL5_adPN specifically
    print(f"\n🎯 CHECKING FOR DL5_adPN (Or7a → Projection Neuron):")
    dl5_adpn = result_df[result_df['output_cell_type'] == 'DL5_adPN']
    if len(dl5_adpn) > 0:
        print(f"   ✅ Found {len(dl5_adpn)} DL5_adPN connections!")
        print(f"   ✅ Found {dl5_adpn['output_root_id'].nunique()} unique DL5_adPN cells")
        
        for _, row in dl5_adpn.head(10).iterrows():
            print(f"      From: {row['source_glomerulus']} (ID: {row['source_root_id']})")
            print(f"      To: {row['output_cell_type']} (ID: {row['output_root_id']})")
            print(f"      Synapses: {row['synapse_count']}")
            print(f"      Role: {row['functional_role']}")
            print()
    
    # Show top source glomeruli to LN connections (for lateral inhibition)
    print(f"\n🔗 TOP ORN→LOCAL NEURON CONNECTIONS (Lateral Inhibition):")
    orn_to_ln = result_df[result_df['broad_category'] == 'Local_Neuron']
    if len(orn_to_ln) > 0:
        top_ln_connections = orn_to_ln.groupby(['source_glomerulus', 'output_cell_type'])['synapse_count'].sum().sort_values(ascending=False).head(15)
        
        for (glom, ln_type), synapses in top_ln_connections.items():
            print(f"   {glom} → {ln_type}: {int(synapses)} synapses")
    
    # Show top source glomeruli to PN connections
    print(f"\n🔗 TOP ORN→PROJECTION NEURON CONNECTIONS:")
    orn_to_pn = result_df[result_df['broad_category'] == 'Projection_Neuron']
    if len(orn_to_pn) > 0:
        top_pn_connections = orn_to_pn.groupby(['source_glomerulus', 'output_cell_type'])['synapse_count'].sum().sort_values(ascending=False).head(15)
        
        for (glom, pn_type), synapses in top_pn_connections.items():
            print(f"   {glom} → {pn_type}: {int(synapses)} synapses")
    
    # Show all PN types found
    print(f"\n📋 ALL PROJECTION NEURON TYPES IDENTIFIED:")
    pn_types = result_df[result_df['broad_category'] == 'Projection_Neuron']['output_cell_type'].unique()
    print(f"   Total unique PN types: {len(pn_types)}")
    for pn in sorted(pn_types)[:20]:
        count = len(result_df[result_df['output_cell_type'] == pn])
        print(f"   • {pn}: {count} connections")
    if len(pn_types) > 20:
        print(f"   ... and {len(pn_types) - 20} more")
    
    print(f"\n✅ Annotated output cells file saved: {output_file}")
    print("   New columns added:")
    print("   • broad_category: Cell type classification (ORN, PN, LN, etc.)")
    print("   • functional_role: Functional role in circuit")
    print("   • description: Detailed biological description")
    print("\n🎯 Ready for ORN→LN→ORN pathway mapping and cross-talk analysis!")

if __name__ == "__main__":
    main()
