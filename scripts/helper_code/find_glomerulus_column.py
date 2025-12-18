import os
import gzip
import pandas as pd

data_dir = 'data/flywire'
target_glomerulus = 'ORN_DL5'

def probe_csv_gz_for_value(file_path, value):
    found_cols = []
    with gzip.open(file_path, 'rt') as f:
        try:
            df = pd.read_csv(f, nrows=10000)  # Read 10k rows max for speed
        except Exception as e:
            print(f"Failed to read {file_path}: {e}")
            return []
        for col in df.columns:
            if df[col].astype(str).str.contains(value, na=False).any():
                found_cols.append(col)
    return found_cols

results = []

for fname in os.listdir(data_dir):
    if fname.endswith('.csv.gz'):
        print(f"Checking {fname}...")
        path = os.path.join(data_dir, fname)
        cols = probe_csv_gz_for_value(path, target_glomerulus)
        if cols:
            results.append({'file': fname, 'columns_with_value': cols})
            print(f"  FOUND in columns: {cols}")
        else:
            print("  Not found in first 10,000 rows.")

print("\nSummary of files with the target glomerulus:")
for res in results:
    print(f"File: {res['file']} | Columns: {res['columns_with_value']}")
