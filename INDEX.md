# Multi-Condition LOOCV Pipeline - Complete Index

## 📖 Documentation

Start here based on your needs:

### For Quick Copy-Paste Commands
→ **[QUICK_START.md](QUICK_START.md)**
- One-line commands for each step
- "I want to..." lookup table
- Troubleshooting guide

### For Complete Reference
→ **[PIPELINE_COMMANDS.md](PIPELINE_COMMANDS.md)**
- Detailed step-by-step explanation
- Parameter descriptions
- Expected performance metrics
- File structure diagram

---

## 🔧 Scripts

### 1. Regenerate Baseline Weights
**File:** `scripts/regenerate_baseline.sh`

Extracts baseline weights from the control condition for use in plot comparison.

```bash
bash scripts/regenerate_baseline.sh
```

**Output:** `/tmp/baseline_weights_intersection.csv`

---

### 2. Run LOOCV with Plots
**File:** `scripts/run_multicond_loocv.py` (existing, enhanced)

Runs leave-one-odor-out cross-validation with visualization.

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
  --outdir out/multicond_loocv_best
```

**Outputs:**
- `out/multicond_loocv_best/plots/` - 7 comparison plots
- `out/multicond_loocv_best/weights_mean_*.csv` - Averaged LOOCV weights
- `out/multicond_loocv_best/conditions_overview.csv` - Summary metrics

---

### 3. Make Predictions from Averaged Weights
**File:** `scripts/predict_with_avg_weights.py`

Uses the mean LOOCV weights to predict ΔPER and compare to true values.

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
- `out/prediction_plots_best/predictions_vs_true.png` - Per-condition bar plots
- `out/prediction_plots_best/predictions_scatter.png` - Scatter plot (overall R²)
- `out/prediction_plots_best/weights_comparison.png` - Weight comparison
- `out/prediction_plots_best/predictions.csv` - Detailed predictions (24 rows)

---

## 📊 Output Files

### LOOCV Results
```
out/multicond_loocv_best/
├── plots/                           ← 7 PNG files
│   ├── weights_deltaper_3-octonol.png
│   ├── weights_deltaper_apple_cider_vinegar.png
│   ├── weights_deltaper_benzaldehyde.png
│   ├── weights_deltaper_ethyl_butyrate.png
│   ├── weights_deltaper_hexanol.png
│   ├── weights_deltaper_linalool.png
│   └── weights_all_conditions.png
├── predictions_opto_*.csv           ← Per-condition LOOCV predictions
├── weights_mean_opto_*.csv          ← Averaged weights (13 receptors)
├── weights_folds_opto_*.csv         ← Fold-specific weights
├── conditions_overview.csv          ← Summary: R², MSE, n_features
└── summary_opto_*.json              ← Detailed metrics per condition
```

### Prediction Results
```
out/prediction_plots_best/
├── predictions_vs_true.png          ← Bar plots (4 conditions)
├── predictions_scatter.png          ← Scatter plot with R²
├── weights_comparison.png           ← Bar chart (13 receptors)
└── predictions.csv                  ← 24 rows (6 odors × 4 conditions)
```

### Baseline Weights
```
/tmp/baseline_weights_intersection.csv
├── feature: Or19a, Or22a, ..., ac3_noOr35a (13 receptors)
└── baseline_w: mean weights from opto_AIR control
```

---

## 📈 Performance Summary

**Configuration:** 6 odors (no citral), 13 receptors (intersection), ElasticNet

| Condition | R² | MSE | Folds |
|-----------|----|----|-------|
| opto_EB | 0.40 | 0.009 | 6 |
| opto_hex | 0.44 | 0.032 | 6 |
| opto_benz_1 | 0.30 | 0.037 | 6 |
| opto_3-oct | 0.27 | 0.004 | 6 |
| **Overall** | **0.35** | - | - |

---

## 🔑 Key Features Used (13 Receptors)

```
Or19a, Or22a, Or2a, Or35a, Or47b, Or67b, Or7a, Or83c,
Or85b, Or98a, ac1, ac2, ac3_noOr35a
```

These are the only receptors active (threshold > 0.05) across **all 6 odors**.

---

## 🎯 How to Use This Pipeline

### Option A: Run Everything at Once
```bash
# 1. Regenerate baseline
bash scripts/regenerate_baseline.sh

# 2. Run LOOCV (copy from QUICK_START.md)
python scripts/run_multicond_loocv.py ...

# 3. Make predictions
python scripts/predict_with_avg_weights.py ...
```

### Option B: Run Step-by-Step
1. Read [QUICK_START.md](QUICK_START.md) for copy-paste commands
2. Copy Step 1 command and run
3. Copy Step 2 command and run
4. Copy Step 3 command and run
5. View results in output directories

### Option C: Reference Only
Use this as a reference while working on your own variations.

---

## 📝 Example Output Interpretation

### LOOCV Plot (weights_deltaper_3-octonol.png)

**Top Subplot (Weights):**
- **Purple bars** = Baseline weights (opto_AIR control)
- **Blue/Orange/Green/Red bars** = Delta weights for each trained condition
- Height = Receptor contribution to prediction

**Bottom Subplot (ΔPER):**
- Mean-centered behavioral response (what we're trying to predict)
- Shows which condition activates neuron the most

### Prediction Plot (predictions_scatter.png)

- **X-axis** = True ΔPER (centered)
- **Y-axis** = Predicted ΔPER (centered)
- **Diagonal line** = Perfect prediction
- **R² value** = Goodness of fit (higher is better, max = 1.0)

### Predictions CSV

Each row shows:
```
condition,odor,true_delta_per,predicted_delta_per,true_centered,predicted_centered
opto_EB,3-Octonol,0.1097,0.0894,0.0052,0.0189
```

This means: For opto_EB + 3-Octonol odor, we predicted 0.0894 when truth was 0.1097

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Baseline weights all zero | Make sure you ran `bash scripts/regenerate_baseline.sh` first |
| File not found error | Check that `/tmp/reaction_rates_no_citral.csv` exists |
| Wrong number of receptors | Verify `--feature-set intersection --activation-threshold 0.05` |
| Plots missing baseline bars | Pass `--plot-baseline-weights /tmp/baseline_weights_intersection.csv` |

See [QUICK_START.md](QUICK_START.md#troubleshooting) for more details.

---

## 📚 Related Files

### Input Data
- `/tmp/reaction_rates_no_citral.csv` - PER data (6 odors)
- `data/mappings/door_to_flywire_mapping.csv` - Receptor mapping

### Source Code
- `src/door_toolkit/multicond_loocv.py` - LOOCV implementation
- `src/door_toolkit/glomerulus_features.py` - Feature matrix builder
- `src/door_toolkit/encoder.py` - DoOR encoder

---

## 🔄 Next Steps

1. Choose which documentation to read:
   - **Quick**: [QUICK_START.md](QUICK_START.md) (5 min read)
   - **Complete**: [PIPELINE_COMMANDS.md](PIPELINE_COMMANDS.md) (15 min read)

2. Run baseline regeneration:
   ```bash
   bash scripts/regenerate_baseline.sh
   ```

3. Run LOOCV + predictions (copy command from docs)

4. View plots and results in output directories

---

## 📧 Questions?

- For command examples: See [QUICK_START.md](QUICK_START.md)
- For detailed explanation: See [PIPELINE_COMMANDS.md](PIPELINE_COMMANDS.md)
- For troubleshooting: See sections above or in QUICK_START.md

---

**Last Updated:** 2026-02-18
**Configuration:** 6 odors (no citral), 13 receptors (intersection), ElasticNet
**Status:** ✓ Complete and ready to use
