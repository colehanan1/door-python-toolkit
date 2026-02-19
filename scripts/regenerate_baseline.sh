#!/bin/bash
# Regenerate baseline weights for LOOCV intersection mode (no citral)
#
# This script:
# 1. Runs multicond LOOCV on 6 odors (no citral) with intersection feature set (13 receptors)
# 2. Extracts baseline weights from opto_AIR control condition
# 3. Saves to /tmp/baseline_weights_intersection.csv
#
# Usage:
#   bash scripts/regenerate_baseline.sh

set -e

cd "$(dirname "$0")/.."

echo "========================================================================"
echo "REGENERATING BASELINE WEIGHTS: 6 odors (no citral), 13 receptors (intersection)"
echo "========================================================================"

# Create no-citral CSV if it doesn't exist
if [ ! -f /tmp/reaction_rates_no_citral.csv ]; then
    echo ""
    echo "Creating /tmp/reaction_rates_no_citral.csv (removing Citral column)..."
    python << 'PYEOF'
import pandas as pd
csv_path = '/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions/reaction_rates_summary_unordered.csv'
df = pd.read_csv(csv_path)
if 'Citral' in df.columns:
    df = df.drop(columns=['Citral'])
    df.to_csv('/tmp/reaction_rates_no_citral.csv', index=False)
    print("✓ Created /tmp/reaction_rates_no_citral.csv")
else:
    print("ERROR: Citral column not found!")
    exit(1)
PYEOF
fi

# Run LOOCV
echo ""
echo "========================================================================"
echo "STEP 1: Running multicond LOOCV"
echo "========================================================================"
echo ""

python scripts/run_multicond_loocv.py \
  --csv /tmp/reaction_rates_no_citral.csv \
  --control-row opto_AIR \
  --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct \
  --model elasticnet \
  --feature-set intersection \
  --activation-threshold 0.05 \
  --l1-ratio 0.3 \
  --outdir out/multicond_loocv_baseline \
  --seed 0

# Extract and save baseline weights
echo ""
echo "========================================================================"
echo "STEP 2: Extracting baseline weights from opto_AIR control"
echo "========================================================================"
echo ""

python << 'PYEOF'
import pandas as pd

# Load control weights
control_weights = pd.read_csv('out/multicond_loocv_baseline/weights_mean_opto_AIR.csv')

# Rename column for clarity
control_weights = control_weights.rename(columns={'mean_w': 'baseline_w'})

# Save to standard location
output_path = '/tmp/baseline_weights_intersection.csv'
control_weights[['feature', 'baseline_w']].to_csv(output_path, index=False)

print(f"✓ Saved baseline weights to: {output_path}")
print("")
print("Baseline weights summary:")
n_nonzero = (control_weights['baseline_w'] != 0).sum()
print(f"  Non-zero weights: {n_nonzero}/{len(control_weights)}")
print("")
print("Top 5 by absolute value:")
top5 = control_weights.reindex(
    control_weights['baseline_w'].abs().argsort(ascending=False)[:5]
)
for _, row in top5.iterrows():
    print(f"    {row['feature']:20s}: {row['baseline_w']:10.6f}")
PYEOF

echo ""
echo "========================================================================"
echo "COMPLETE!"
echo "========================================================================"
echo ""
echo "Baseline weights file: /tmp/baseline_weights_intersection.csv"
echo "LOOCV outputs: out/multicond_loocv_baseline/"
echo ""
echo "Next step: Use baseline weights in future LOOCV runs:"
echo ""
echo "  python scripts/run_multicond_loocv.py \\"
echo "    --csv /tmp/reaction_rates_no_citral.csv \\"
echo "    --plot \\"
echo "    --plot-baseline-weights /tmp/baseline_weights_intersection.csv \\"
echo "    ... other args ..."
