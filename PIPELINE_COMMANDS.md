# Multi-Condition LOOCV Pipeline - Complete Commands

## Overview
This document contains all commands for running the multi-condition LOOCV regression pipeline with baseline weight comparison and prediction visualization.

**Configuration Used:**
- Data: 6 odors (no citral)
- Features: 13 receptors (intersection mode, threshold=0.05)
- Model: ElasticNet (l1_ratio=0.3)
- LOOCV: Leave-one-odor-out (6 folds)

---

## 1. Regenerate Baseline Weights (One-time Setup)

This creates the baseline weights from the control condition (opto_AIR).

```bash
# Option A: Using the provided script
bash scripts/regenerate_baseline.sh

# Option B: Manual command sequence
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

# Then extract baseline weights:
python << 'EOF'
import pandas as pd
df = pd.read_csv('out/multicond_loocv_baseline/weights_mean_opto_AIR.csv')
df = df.rename(columns={'mean_w': 'baseline_w'})
df[['feature', 'baseline_w']].to_csv('/tmp/baseline_weights_intersection.csv', index=False)
print(f'✓ Baseline weights saved to /tmp/baseline_weights_intersection.csv')
EOF
```

**Output:**
- Baseline weights: `/tmp/baseline_weights_intersection.csv`
- LOOCV results: `out/multicond_loocv_baseline/`

---

## 2. Run Full LOOCV with Baseline Weights and Plots

Runs LOOCV with visualizations showing baseline vs delta weights.

```bash
python scripts/run_multicond_loocv.py \
  --csv /tmp/reaction_rates_no_citral.csv \
  --control-row opto_AIR \
  --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct \
  --model elasticnet \
  --feature-set intersection \
  --activation-threshold 0.05 \
  --l1-ratio 0.3 \
  --plot \
  --plot-top-n 13 \
  --plot-baseline-weights /tmp/baseline_weights_intersection.csv \
  --plot-comparison \
  --outdir out/multicond_loocv_best \
  --seed 0
```

**Output:**
- LOOCV results: `out/multicond_loocv_best/`
- Plots (baseline in purple): `out/multicond_loocv_best/plots/*.png`
  - `weights_deltaper_*.png` (6 odor-specific plots with baseline vs delta weights)
  - `weights_all_conditions.png` (cross-condition comparison)

---

## 3. Make Predictions with Averaged Weights

Uses the mean weights from LOOCV to predict ΔPER and compare to true values.

```bash
python scripts/predict_with_avg_weights.py \
  --loocv-dir out/multicond_loocv_best \
  --csv /tmp/reaction_rates_no_citral.csv \
  --control-row opto_AIR \
  --conditions opto_EB,opto_hex,opto_benz_1,opto_3-oct \
  --feature-set intersection \
  --activation-threshold 0.05 \
  --outdir out/prediction_plots_best
```

**Output:**
- Prediction plots: `out/prediction_plots_best/`
  - `predictions_vs_true.png` (bar plots per condition)
  - `predictions_scatter.png` (scatter: true vs predicted)
  - `weights_comparison.png` (averaged weights across conditions)
- Predictions CSV: `out/prediction_plots_best/predictions.csv`

**CSV Columns:**
- `condition`: trained condition name
- `odor`: odor name
- `true_delta_per`: true ΔPER value
- `predicted_delta_per`: predicted ΔPER value
- `true_centered`: true ΔPER (mean-centered)
- `predicted_centered`: predicted ΔPER (mean-centered)

---

## 4. Quick Full Pipeline (All Steps)

Run everything in sequence:

```bash
#!/bin/bash

# Step 1: Regenerate baseline
bash scripts/regenerate_baseline.sh

# Step 2: Run LOOCV with plots
python scripts/run_multicond_loocv.py \
  --csv /tmp/reaction_rates_no_citral.csv \
  --control-row opto_AIR \
  --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct \
  --model elasticnet \
  --feature-set intersection \
  --activation-threshold 0.05 \
  --l1-ratio 0.3 \
  --plot --plot-top-n 13 \
  --plot-baseline-weights /tmp/baseline_weights_intersection.csv \
  --plot-comparison \
  --outdir out/multicond_loocv_best \
  --seed 0

# Step 3: Make predictions
python scripts/predict_with_avg_weights.py \
  --loocv-dir out/multicond_loocv_best \
  --csv /tmp/reaction_rates_no_citral.csv \
  --control-row opto_AIR \
  --conditions opto_EB,opto_hex,opto_benz_1,opto_3-oct \
  --feature-set intersection \
  --activation-threshold 0.05 \
  --outdir out/prediction_plots_best

echo "✓ Complete!"
echo ""
echo "Results:"
echo "  LOOCV plots: out/multicond_loocv_best/plots/"
echo "  Predictions: out/prediction_plots_best/"
```

---

## 5. View Results

```bash
# List all plots
ls -lh out/multicond_loocv_best/plots/
ls -lh out/prediction_plots_best/

# View predictions CSV
head -20 out/prediction_plots_best/predictions.csv

# View baseline weights
head -10 /tmp/baseline_weights_intersection.csv
```

---

## Key Parameters Explained

| Parameter | Value | Reason |
|-----------|-------|--------|
| `--csv` | `/tmp/reaction_rates_no_citral.csv` | 6 odors (citral removed) |
| `--feature-set` | `intersection` | Only receptors active in all 6 odors |
| `--activation-threshold` | `0.05` | Only responses > 0.05 count as "active" |
| `--model` | `elasticnet` | Balanced L1/L2 regularization |
| `--l1-ratio` | `0.3` | 30% L1 (Lasso), 70% L2 (Ridge) |
| `--plot-top-n` | `13` | Show all 13 intersection receptors |

---

## Feature Set Details

**Current Configuration:**
- Total receptors in dataset: 60
- Receptors with any non-zero response: 57 (no_blanks)
- Receptors active in ALL 6 odors: 13 (intersection)

**The 13 Intersection Receptors:**
```
Or19a, Or22a, Or2a, Or35a, Or47b, Or67b, Or7a, Or83c,
Or85b, Or98a, ac1, ac2, ac3_noOr35a
```

---

## Expected Performance

With 13 receptors (intersection) and 6 odors (no citral):

| Condition | R² | MSE | Notes |
|-----------|----|----|-------|
| opto_EB | 0.40 | 0.009 | Good prediction |
| opto_hex | 0.44 | 0.032 | Best prediction |
| opto_benz_1 | 0.30 | 0.037 | Moderate |
| opto_3-oct | 0.27 | 0.004 | Moderate |

**Overall R²: ~0.35** (averaged across 4 trained conditions)

---

## Troubleshooting

### Baseline weights are all zero
**Problem:** Using baseline from opto_AIR control when it has poor fit (R² < 0)

**Solution:** Ensure you extracted baseline AFTER fitting the LOOCV
```bash
python scripts/run_multicond_loocv.py ... --outdir out/multicond_loocv_baseline
# Then extract from that directory's weights_mean_opto_AIR.csv
```

### File not found errors
**Problem:** Script looking for weights_mean_opto_3_oct.csv but file is weights_mean_opto_3-oct.csv

**Solution:** Use consistent naming - the script handles `opto_3-oct` → `opto_3_oct` conversion

### No citral CSV not found
**Problem:** `/tmp/reaction_rates_no_citral.csv` doesn't exist

**Solution:** Create it:
```bash
python << 'EOF'
import pandas as pd
df = pd.read_csv('/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions/reaction_rates_summary_unordered.csv')
df = df.drop(columns=['Citral'])
df.to_csv('/tmp/reaction_rates_no_citral.csv', index=False)
EOF
```

---

## File Structure

```
out/
├── multicond_loocv_baseline/      (Initial baseline run)
│   ├── conditions_overview.csv
│   ├── weights_mean_opto_AIR.csv  ← Extract baseline from here
│   ├── weights_mean_opto_EB.csv
│   └── ...
├── multicond_loocv_best/          (Main LOOCV run with plots)
│   ├── conditions_overview.csv
│   ├── plots/
│   │   ├── weights_deltaper_*.png
│   │   └── weights_all_conditions.png
│   ├── weights_mean_*.csv
│   └── ...
└── prediction_plots_best/         (Predictions from averaged weights)
    ├── predictions_vs_true.png
    ├── predictions_scatter.png
    ├── weights_comparison.png
    └── predictions.csv

/tmp/
└── baseline_weights_intersection.csv  (Baseline weights for plotting)
```

---

## Notes

- All commands assume you're in the door-python-toolkit root directory
- The `--seed 0` ensures reproducible results
- Baseline weights are computed from **control condition only** (opto_AIR raw PER)
- Delta weights show **change from baseline** in each trained condition
- Predictions use **averaged LOOCV weights** (mean across 6 folds)
