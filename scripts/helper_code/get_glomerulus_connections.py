import pandas as pd
import os
import gzip

# Step 1: Put your target labels here
TARGET_LABELS = [
    'ac1A', 'ac1B', 'ac2A', 'ac2B', 'ac3A', 'ac3B', 'ac4', 'Or1a', 'Or2a',
    'Or7a', 'Or9a', 'Or10a', 'Or13a', 'Or19a', 'Or22a', 'Or22b', 'Or22c', 'Or23a',
    'Or24a', 'Or30a', 'Or33a', 'Or33b', 'Or33c', 'Or35a', 'Or42a', 'Or42b', 'Or43a', 'Or43b',
    'Or45a', 'Or45b', 'Or46a', 'Or47a', 'Or47b', 'Or49a', 'Or49b', 'Or59a','Or59b', 'Or59c',
    'Or65a', 'Or67a', 'Or67b', 'Or67c', 'Or67d', 'Or71a', 'Or74a', 'Or82a', 'Or85a', 'Or85b',
    'Or85c', 'Or85d', 'Or85e', 'Or85f', 'Or88a', 'Or92a', 'Or94a', 'Or94b', 'Or98a', 'Gr21a.Gr63a',
    'ab2B', 'ab4B', 'ab5B', 'pb2A', 'Or69a', 'ac1', 'ac2', 'ac3_noOr35a', 'Ir31a', 'Ir41a', 'Ir75a',
    'Ir75d', 'Ir76a', 'Ir84a', 'Ir92a', 'Ir64a.DC4', 'Ir64a.DP1m', 'ac1BC', 'ac2BC', 'Or83c'
]

data_dir = "data/flywire"
labels_path = os.path.join(data_dir, "processed_labels.csv.gz")

# Step 2: Load labels file (handles gz automatically)
if labels_path.endswith('.gz'):
    with gzip.open(labels_path, 'rt') as f:
        labels = pd.read_csv(f)
else:
    labels = pd.read_csv(labels_path)

# Step 3: Find all matching neurons
matched = pd.DataFrame()
for label in TARGET_LABELS:
    hits = labels[labels['processed_labels'].str.contains(label, case=False, na=False)].copy()
    hits['matched_label'] = label
    matched = pd.concat([matched, hits], ignore_index=True)

matched = matched.drop_duplicates(subset=['root_id'])

# Step 4: Optionally, look for connectivity data
connectivity = None
for fname in os.listdir(data_dir):
    if "connect" in fname.lower() and fname.endswith('.csv'):
        connectivity = pd.read_csv(os.path.join(data_dir, fname))
        break

# Step 5: For each source neuron, find postsynaptic partners if possible
results = []
for idx, row in matched.iterrows():
    cell_id = row['root_id']
    label = row['matched_label']
    inputs = []
    if connectivity is not None:
        # Assuming columns: 'pre_root_id', 'post_root_id'
        postsynaptic = connectivity[connectivity['pre_root_id'] == cell_id]['post_root_id'].unique().tolist()
        inputs = postsynaptic
    results.append({
        'source_root_id': cell_id,
        'label': label,
        'connected_postsynaptic_root_ids': ';'.join(str(i) for i in inputs)
    })

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(data_dir, "selected_glomerulus_and_targets.csv"), index=False)

print(f"Saved {len(results_df)} entries to selected_glomerulus_and_targets.csv")
