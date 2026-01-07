# Dataset-level PER Rate Model (Option 1)

This document describes the dataset-level (aggregated) PER rate model used to reproduce a behavioral matrix like `reaction_rates_summary_unordered.csv` (rows = datasets/conditions, columns = test odors, values in `[0,1]`).

## Key definitions

- **Dataset**: a training condition (e.g., `opto_hex`, `hex_control`, `opto_EB_6_training`).
- **Trained odor**: the odor paired with reward for `opto_*` datasets (or the matched odor for controls).
- **PER rate**: the aggregate probability of proboscis extension response for a (dataset, test odor) cell, computed as the mean of binary outcomes across flies in that condition. This is a **dataset-level summary statistic**, NOT per-fly dynamics or trial-by-trial probabilities.
- **Observed cell**: a cell in the behavioral matrix that had measurements (non-NaN in input CSV).
- **Extrapolated cell**: a cell that was NaN in the input (no data), but for which the model can predict based on DoOR profiles.

## Model

We fit a sparse generalized linear model (GLM) on DoOR receptor-response features:

```
logit(p̂) = b + b_reward * reward_flag
            + w_test · x_test
            + w_train · x_train
            + w_int · (x_test ⊙ x_train)
            + w_diff · (x_test - x_train)   (optional)
```

Where:
- `x_test` = DoOR receptor-response vector for the test odor (length = #receptors, e.g. 78).
- `x_train` = DoOR receptor-response vector for the trained odor for that dataset.
- `reward_flag` = `1` if dataset starts with `opto_`, else `0`.
- `⊙` = elementwise product.

## Feature Construction

For each (dataset, test_odor) cell:

1. **Test profile** (`x_test`): DoOR receptor-response vector for the test odor
2. **Trained profile** (`x_train`): DoOR vector for the trained odor (inferred from dataset name)
3. **Interaction** (`x_test ⊙ x_train`): Element-wise product (co-activation)
4. **Difference** (`x_test - x_train`, optional): Directional contrast

### Receptor Masking

- By default, uses 55 adult-only ORNs from `data/mappings/training_receptor_set.json`
- Excludes 23 larval/unmapped receptors
- Manifest saved to `adult_only_mask_manifest.json` with details

### Odor Name Resolution

- Behavioral CSV labels (e.g., "Hexanol") → DoOR canonical names ("1-hexanol")
- Uses InChIKey-based lookup via candidate mappings + fuzzy matching
- All resolution decisions logged to `odor_name_resolution.json`
- Control stimuli (air, paraffin oil) encoded as zero vectors (logged as control_stimulus)
- Unknown odors encoded as zero vectors (logged as NOT_FOUND)

## Decision → Evidence → Implementation notes

### BCE on fractional labels

- **Decision**: use binary cross-entropy (BCE) with logits on PER rates `y ∈ [0,1]`.
- **Evidence**: a PER rate is an aggregate mean of Bernoulli outcomes; BCE corresponds to the negative log-likelihood for Bernoulli targets, and fractional targets are a standard “soft label” relaxation.
- **Implementation**: training uses `torch.nn.functional.binary_cross_entropy_with_logits(logits, y)`.

### L1 (lasso-style) regularization

- **Decision**: apply L1 regularization to ORN weight vectors to encourage sparsity and interpretability.
- **Evidence**: sparsity yields simpler receptor circuits and reduces sensitivity to correlated features.
- **Implementation**: `loss = BCE + λ * (|w_test|_1 + |w_train|_1 + |w_int|_1 + |w_diff|_1)`.

### Ablation-based importance (preferred)

- **Decision**: prioritize ablation-based ORN importance for experiment planning.
- **Evidence**: correlated inputs can make weight magnitudes misleading; ablation measures sensitivity of model fit in the data regime.
- **Implementation**: for each ORN channel `i`, set channel `i` to 0 in both `x_test` and `x_train` (and derived features), recompute BCE on the full observed training table, and record `ΔBCE = BCE_ablated - BCE_baseline`.

### Trained-odor resolution

- **Decision**: resolve trained odors via an explicit mapping first, then deterministic parsing.
- **Evidence**: dataset naming conventions are not fully standardized (e.g., `opto_benz_1`).
- **Implementation**: `door_toolkit.pathways.behavior_rate_model.resolve_trained_odor_for_dataset()` uses a default mapping and common token parsing; unknown datasets raise a clear error unless you provide an override mapping.

### Input validation

- **Decision**: Auto-detect wide vs long CSV format and validate rate ranges.
- **Evidence**: User CSVs may use percentage [0,100] or fractional [0,1] rates; explicit validation prevents silent errors.
- **Implementation**: `load_behavior_matrix_csv()` detects format, rescales if needed (with warning), and logs metadata to `input_format_metadata.json`.

### Learning-effect computation

- **Decision**: ΔPER = PER_opto - PER_control for matched opto/control pairs.
- **Evidence**: Learning effect is the causal difference between optogenetic stimulation and matched control; requires exact dataset pairing.
- **Implementation**: `compute_learning_effect_metrics()` merges predictions for matched pairs (e.g., `opto_hex` vs `hex_control`), computes delta for both y_true and y_pred, reports per-odor errors.

## Outputs

The training script writes artifacts under `--output_dir`:

**Core predictions:**
- `predicted_rates.csv`: **Long format** with observed cells only. Columns: `[dataset, test_odor, y_true, y_pred, split, trained_odor, trained_odor_door, test_odor_door]`. `split` = "observed" for all rows. `y_true` = original measured PER rate (no NaN). `y_pred` = model prediction.
- `extrapolated_predictions.csv`: **Long format** with extrapolated cells (originally NaN in input). Same columns, but `split` = "extrapolated", `y_true` = NaN.

**Metrics:**
- `metrics.json`: Global fit metrics (BCE, R², Pearson r) on observed cells.
- `metrics_per_dataset.csv`: Breakdown by dataset (one row per dataset): `[dataset, n_cells, bce, r2, pearson_r, mae, trained_odor]`.
- `metrics_per_test_odor.csv`: Breakdown by test odor: `[test_odor, n_cells, bce, r2, pearson_r, mae, door_name]`.
- `learning_effect_metrics.csv`: ΔPER = opto - control for matched pairs: `[test_odor, delta_per_true, delta_per_pred, error, opto_dataset, control_dataset]` (only if matched controls exist).
- `predicted_learning_effect_opto_minus_control.csv`: Old wide-format learning effect table (for backward compatibility).

**Receptor importance:**
- `orn_importance_global.csv`: Combined weight-based and ablation-based importance.
- `orn_importance_by_dataset.csv`: Ablation ΔBCE per dataset.
- `orn_importance_by_test_odor.csv`: Ablation ΔBCE per test odor.
- `ablation_sweep.csv`: Top-K receptors with learning-effect deltas.

**Connectome-aware interpretation (optional):**
- `connectome_analysis/connectome_inputs.json`: paths, hashes, shapes, orientation/alignment reports used for connectome propagation.
- `connectome_analysis/orn_connectome_amplified_importance.csv`: ORN-level base importance + fanout metrics + connectome-amplified importance (ranked).
- `connectome_analysis/pn_influence.csv`: Top PNs by propagated influence.
- `connectome_analysis/kc_influence.csv`: Top KCs by propagated influence.
- `connectome_analysis/connectome_summary.json`: Concentration metrics (top-K fractions, gini-like summary) and top ORNs/PNs/KCs.
- `connectome_analysis/orn_to_pn_top_edges.csv`: (Optional) ORN→PN edge list for top ORNs by amplified importance.

**Provenance & logging:**
- `run_config.json`: CLI args, git SHA, hyperparameters.
- `input_format_metadata.json`: CSV format detection, rescaling log.
- `odor_name_resolution.json`: All odor name mappings (csv_name → door_name or NOT_FOUND).
- `adult_only_mask_manifest.json`: Receptor mask details (55 adult ORNs kept, 23 excluded).
- `training_loss.csv`: Training curve for diagnostics.

## Connectome-aware interpretation (post-training)

This repository also contains FlyWire-derived connectivity artifacts for ORN→PN and PN→KC. The behavior-rate model training objective is unchanged; after training we can **optionally** propagate the learned ORN importance through these matrices to get a first-order downstream readout.

### Inputs

Expected files (from `scripts/extract_flywire_connectivity.py`):
- `orn_pn_connectivity.pt`
- `pn_kc_connectivity.pt`
- `connectivity_metadata.json` (required for ORN ordering alignment; must include `receptor_names`)

By default, the training script auto-detects connectivity under:
- `data/pgcn_features/connectivity/` (preferred)
- `data/connectivity/`

Override with `--connectome_dir`, or skip with `--disable_connectome_analysis`.

### Propagation definition

Let:
- `s_orn` be the non-negative ORN importance vector from the trained GLM weights, where `s_orn[i] = |w_test[i]| + |w_train[i]| + |w_int[i]| (+ |w_diff[i]| if enabled)`.
- `A` be the ORN→PN matrix, oriented as `(n_pn, n_orn)` after detecting whether the stored file is ORN×PN vs PN×ORN.
- `B` be the PN→KC matrix, oriented as `(n_kc, n_pn)` after detecting whether the stored file is PN×KC vs KC×PN.

We compute:

```
s_pn = A @ s_orn
s_kc = B @ s_pn
```

These are not firing rates or dynamics; they are a linear "influence mass" readout under the assumption that larger ORN importance combined with stronger fanout yields larger downstream footprint.

### Connectome-amplified ORN importance

We define a simple amplification term based on **two-hop KC reach**:

```
fanout_kc[orn] = sum_kc ( (B @ A)[kc, orn] )
amp_factor = fanout_kc / mean(fanout_kc)
connectome_amplified_importance = s_orn * amp_factor
```

This answers: *do the ORNs that matter in the GLM also sit in wiring positions that project broadly into KCs?*

### Decision → Evidence → Implementation notes

- **Decision**: infer matrix orientation from shape compatibility (shared PN dimension), not hardcoded assumptions.
  - **Evidence**: exported connectivity artifacts may be stored as ORN×PN vs PN×ORN (and PN×KC vs KC×PN) depending on pipeline.
  - **Implementation**: `door_toolkit.pathways.connectome_analysis.orient_connectome()` orients matrices to `(PN, ORN)` and `(KC, PN)` by matching the shared PN dimension.

- **Decision**: require explicit receptor name mapping for ORN alignment.
  - **Evidence**: the behavior-rate model uses a 55-ORN adult-only subset in encoder order; connectome matrices often include additional receptors in a different order.
  - **Implementation**: `align_orn_connectome()` maps by name using `connectivity_metadata.json["receptor_names"]`; missing receptors raise a clear error (and auto-detected runs skip analysis rather than producing misaligned outputs).

## How to run

```
python scripts/train_behavior_rate_model.py \
  --behavior_csv /path/to/reaction_rates_summary_unordered.csv \
  --output_dir outputs/behavior_rate_model \
  --epochs 500 --lr 1e-2 --l1_lambda 1e-3
```

Default behavior:
- trains on datasets starting with `opto_`
- excludes `opto_AIR` from training

To train on all datasets in the CSV, set `--train_prefix ''`.
