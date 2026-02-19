# Quick Start: Regenerate Baseline & Run Pipeline

## One-Line Commands

### 1. Regenerate Baseline (first time only)
```bash
bash scripts/regenerate_baseline.sh
```

### 2. Run Full Pipeline (LOOCV + Plots + Predictions)
```bash
# LOOCV with plots
python scripts/run_multicond_loocv.py --csv /tmp/reaction_rates_no_citral.csv --control-row opto_AIR --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct --model elasticnet --feature-set intersection --activation-threshold 0.05 --l1-ratio 0.3 --plot --plot-top-n 13 --plot-baseline-weights /tmp/baseline_weights_intersection.csv --plot-comparison --outdir out/multicond_loocv_best --seed 0

# Predictions
python scripts/predict_with_avg_weights.py --loocv-dir out/multicond_loocv_best --csv /tmp/reaction_rates_no_citral.csv --control-row opto_AIR --conditions opto_EB,opto_hex,opto_benz_1,opto_3-oct --feature-set intersection --activation-threshold 0.05 --outdir out/prediction_plots_best
```

## What Each Step Does

### Baseline Regeneration
Creates baseline weights from control condition for comparison in plots.

**Command:**
```bash
bash scripts/regenerate_baseline.sh
```

**Creates:**
- `/tmp/baseline_weights_intersection.csv` - Baseline weights from opto_AIR

**What's in baseline weights:**
- All 13 intersection receptors with their control weights
- Shows "normal" weight for each receptor (in purple on plots)

---

### LOOCV + Plots
Runs leave-one-odor-out cross-validation and generates comparison plots.

**Command:**
```bash
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
```

**Outputs:**
```
out/multicond_loocv_best/
├── plots/
│   ├── weights_deltaper_3-octonol.png          ← Purple bars = baseline
│   ├── weights_deltaper_apple_cider_vinegar.png
│   ├── weights_deltaper_benzaldehyde.png
│   ├── weights_deltaper_ethyl_butyrate.png
│   ├── weights_deltaper_hexanol.png
│   ├── weights_deltaper_linalool.png
│   └── weights_all_conditions.png
├── predictions_opto_*.csv
├── weights_mean_opto_*.csv
└── conditions_overview.csv
```

**Plots Show:**
- **Purple bars:** Baseline weights (from opto_AIR)
- **Colored bars:** Delta weights for each trained condition
- **Bottom subplot:** Mean-centered ΔPER response

---

### Make Predictions
Uses averaged LOOCV weights to predict ΔPER and compare to truth.

**Command:**
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

**Outputs:**
```
out/prediction_plots_best/
├── predictions_vs_true.png    ← Bar plots (predicted vs true per condition)
├── predictions_scatter.png    ← Scatter plot (overall R² shown)
├── weights_comparison.png     ← Weights for all 13 receptors across conditions
└── predictions.csv            ← Detailed prediction values
```

---

## Key Files

### Input Files
```
/tmp/reaction_rates_no_citral.csv          ← PER data (6 odors, no citral)
/tmp/baseline_weights_intersection.csv     ← Baseline weights (generated)
```

### Output Files
```
out/multicond_loocv_best/plots/*.png       ← LOOCV comparison plots
out/prediction_plots_best/*.png            ← Prediction comparison plots
out/prediction_plots_best/predictions.csv  ← Detailed predictions
```

---

## Expected Results

### LOOCV Performance (R² per condition)
```
opto_EB:      R² = 0.40  (best)
opto_hex:     R² = 0.44  (best)
opto_benz_1:  R² = 0.30  (moderate)
opto_3-oct:   R² = 0.27  (moderate)
Overall:      R² = 0.35
```

### Features Used
- **13 receptors** (intersection mode)
- Only receptors active in **all 6 odors** (threshold > 0.05)

```
Or19a, Or22a, Or2a, Or35a, Or47b, Or67b, Or7a, Or83c, Or85b, Or98a, ac1, ac2, ac3_noOr35a
```

---

## Interpretation Guide

### LOOCV Plots (weights_deltaper_*.png)

**Top subplot (weights):**
- **Purple**: Baseline weight for that receptor
- **Blue**: Change in opto_EB
- **Orange**: Change in opto_hex
- **Green**: Change in opto_benz_1
- **Red**: Change in opto_3-oct

**Bottom subplot (ΔPER):**
- Mean-centered behavioral response for each condition
- Shows which odor activates the neuron the most

### Prediction Plots

**predictions_vs_true.png:**
- Blue bars = true ΔPER
- Orange bars = predicted ΔPER
- Good fit when bars align

**predictions_scatter.png:**
- Each dot = one prediction
- Points on diagonal = perfect prediction
- R² measures fit quality

**weights_comparison.png:**
- Shows which receptors are most important for each condition
- Larger bar = stronger weight

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No citral CSV not found" | Run: `python << 'EOF'\nimport pandas as pd\ndf = pd.read_csv('/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions/reaction_rates_summary_unordered.csv')\ndf = df.drop(columns=['Citral'])\ndf.to_csv('/tmp/reaction_rates_no_citral.csv', index=False)\nEOF` |
| "Baseline weights all zero" | Make sure you extracted from the LOOCV output directory, not from a previous run |
| "File weights_mean_opto_3_oct.csv not found" | The script converts `opto_3-oct` → `opto_3_oct`. Make sure condition name is exactly `opto_3-oct` with hyphen |

---

## Commands by Purpose

### I want to...

**...regenerate baseline weights**
```bash
bash scripts/regenerate_baseline.sh
```

**...run LOOCV only**
```bash
python scripts/run_multicond_loocv.py --csv /tmp/reaction_rates_no_citral.csv --control-row opto_AIR --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct --model elasticnet --feature-set intersection --activation-threshold 0.05 --outdir out/multicond_loocv_test
```

**...run LOOCV with plots**
```bash
python scripts/run_multicond_loocv.py --csv /tmp/reaction_rates_no_citral.csv --control-row opto_AIR --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct --model elasticnet --feature-set intersection --activation-threshold 0.05 --l1-ratio 0.3 --plot --plot-top-n 13 --plot-baseline-weights /tmp/baseline_weights_intersection.csv --plot-comparison --outdir out/multicond_loocv_best
```

**...make predictions from existing LOOCV**
```bash
python scripts/predict_with_avg_weights.py --loocv-dir out/multicond_loocv_best --csv /tmp/reaction_rates_no_citral.csv --control-row opto_AIR --conditions opto_EB,opto_hex,opto_benz_1,opto_3-oct --feature-set intersection --activation-threshold 0.05 --outdir out/prediction_plots_best
```

**...check predictions**
```bash
head -20 out/prediction_plots_best/predictions.csv
cat out/prediction_plots_best/predictions.csv | awk -F, '{print $1, $3, $4}' | column -t
```

---

## Next Steps

1. Run baseline regeneration: `bash scripts/regenerate_baseline.sh`
2. Run full pipeline: Use commands above
3. Check plots in `out/multicond_loocv_best/plots/`
4. View predictions in `out/prediction_plots_best/predictions.csv`
5. Adjust parameters as needed (model, l1-ratio, threshold, etc.)
