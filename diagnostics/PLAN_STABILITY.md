# Stability + Metrics Layer Plan

## Discovery summary
- Feature matrices X are built in `src/door_toolkit/pathways/behavioral_prediction.py` via:
  - `_extract_test_odorant_features`, `_extract_trained_odorant_features`, `_extract_interaction_features`.
- Receptor ordering comes from `DoOREncoder.response_matrix` column order in `src/door_toolkit/encoder.py` and is exposed as `predictor.masked_receptor_names` (or `encoder.receptor_names`).
- LASSO selection is in `LassoBehavioralPredictor.fit_behavior()` and `fit_lasso_with_fixed_scaler()` (same file).
- Ridge/ElasticNet CV logic exists in `diagnostics/run_postA_postB_audit.py` (LOOCV grid search).
- Audit outputs are under `diagnostics/postA_postB_audit_*/` with `audit_metrics.csv` + `audit_artifacts.json`.

## Files to add/change
- Add: `diagnostics/run_stability_and_metrics.py` (new stability + metrics runner).
- Add: `tests/test_stability_metrics.py` (determinism + schema + intercept-only flag tests).
- Update: `.gitignore` to allow tracked `diagnostics/*.py` and `diagnostics/*.md`.
- Update: `docs/BEHAVIORAL_PREDICTION_ANALYSIS.md` with 5-line “how to run stability layer”.

## Algorithms to implement
- Standardized metrics for each (condition, mode, modelclass):
  - y_std, y_var, y_min, y_max; pred_std, pred_min, pred_max; cv_mse; nmse; rmse_over_y_std;
    intercept_only_flag; intercept_only_mse (LOOCV mean predictor).
- ORN stability (LOOO):
  - For each fold: fit model on n-1 odorants (same scaling rules as baseline).
  - Record selected ORNs + coefficients; compute selection_frequency, sign_consistency,
    mean/std abs(weight), mean rank by abs(weight).
  - LASSO only if not intercept-only; ElasticNet for ΔPER when LASSO is intercept-only; Ridge uses rank stability.
- Experiment shortlist: top 5 ORNs by stability_score = selection_frequency * sign_consistency,
  plus confidence flags (nmse>=1, intercept-only, missing controls).

## Verification steps
- `pytest -q` (determinism + schema tests for stability outputs).
- Run stability script on real CSV + conditions with seed=1337; check outputs:
  - `stability_per_condition.csv`, `model_metrics.csv`, `EXPERIMENT_SHORTLIST.md`, `SUMMARY.md`, `RUN_COMMANDS.txt`.
