# Baseline Drift Hypotheses

## Summary
No obvious in-place mutation or stochastic sources were found in the LASSO predictor or the ablation/focus scripts. The most plausible explanations are (1) accidental in-process mutation of a view derived from `X`, or (2) changes in target alignment when using ΔPER (control subtraction). Each hypothesis below includes file and function references.

## Hypotheses (with code locations)

### 1) View-based mutation risk (low likelihood)
- `src/door_toolkit/pathways/behavioral_prediction.py:1560` `restrict_to_receptors()` returns `X[:, kept_indices_sorted]` without `.copy()`.
  - This returns a view; if any downstream code mutates `X_restricted` in-place it could mutate the original `X` (and appear as baseline drift).
  - In `scripts/lasso_with_focus_mode.py:421` the view is only passed to `StandardScaler.fit` and LASSO fitting, which do not mutate input arrays, so this risk is theoretical but low.

### 2) In-place ablation (low likelihood)
- `src/door_toolkit/pathways/behavioral_prediction.py:1410` `apply_receptor_ablation()` explicitly copies `X` before ablation.
  - This is safe; baseline drift would require a different ablation path that modifies `X` in-place.
  - `scripts/lasso_with_ablations.py:456` uses `apply_receptor_ablation()` (safe).

### 3) Non-determinism in CV or lambda selection (unlikely)
- `src/door_toolkit/pathways/behavioral_prediction.py:915` `LassoCV(... random_state=42)`.
- `src/door_toolkit/pathways/behavioral_prediction.py:961` `cross_val_score` uses deterministic KFold (no shuffle).
  - Without shuffle, folds are deterministic and reproducible; no randomness expected.

### 4) Data alignment differences (likely for ΔPER vs raw)
- `src/door_toolkit/pathways/behavioral_prediction.py:873-926` control subtraction uses different masks depending on `missing_control_policy`.
  - ΔPER runs drop rows with NaNs in either opto or control (`skip`), or fill missing controls (`zero`).
  - This can change sample counts and target variance vs raw fits, potentially leading to different selected features.

### 5) Dataset label normalization changes (low impact)
- `src/door_toolkit/pathways/behavioral_prediction.py:726` `_resolve_dataset_name()` normalizes dataset labels.
  - If the CSV index has multiple labels that normalize to the same token, this can cause ambiguity errors; otherwise should not affect results.

## Notes
- No global caches or shared mutable matrices were found in the predictor; `get_receptor_profile()` returns fresh arrays.
- The diagnostic script added in this task will validate reproducibility and detect constant-prediction collapses.
