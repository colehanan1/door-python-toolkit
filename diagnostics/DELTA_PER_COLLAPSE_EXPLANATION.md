# ΔPER LASSO Collapse Explanation

Latest audit run: `diagnostics/postA_postB_audit_20260108_154458`

## Evidence of intercept-only collapse (LASSO)

- opto_hex / delta_base: INTERCEPT-ONLY; n_selected=0, pred_std=0, cv_mse=0.100622, intercept_only_mse=0.100622, y_std=0.271894
- opto_hex / delta_extended: INTERCEPT-ONLY; n_selected=0, pred_std=0, cv_mse=0.100622, intercept_only_mse=0.100622, y_std=0.271894
- opto_EB / delta_base: INTERCEPT-ONLY; n_selected=0, pred_std=0, cv_mse=0.0165606, intercept_only_mse=0.0165606, y_std=0.110304
- opto_EB / delta_extended: INTERCEPT-ONLY; n_selected=0, pred_std=0, cv_mse=0.0165606, intercept_only_mse=0.0165606, y_std=0.110304
- opto_benz_1 / delta_base: non-intercept; n_selected=1, pred_std=0.0544301, cv_mse=0.0382769, intercept_only_mse=0.0388923, y_std=0.169038
- opto_benz_1 / delta_extended: non-intercept; n_selected=1, pred_std=0.0544301, cv_mse=0.0382769, intercept_only_mse=0.0388923, y_std=0.169038

## If LASSO collapsed, how much better are Ridge/ElasticNet?

- opto_hex / delta_base: best alt = elasticnet_0.5 | Δcv_mse=0, Δnmse=0
- opto_hex / delta_extended: best alt = elasticnet_0.5 | Δcv_mse=0, Δnmse=0
- opto_EB / delta_base: best alt = ridge | Δcv_mse=0.00694191, Δnmse=0.570554
- opto_EB / delta_extended: best alt = ridge | Δcv_mse=0.00694192, Δnmse=0.570555

## Why LASSO collapses in ΔPER for opto_hex/opto_EB

The audit shows ΔPER LASSO selecting zero features with pred_std=0 and cv_mse equal to intercept-only MSE. This indicates the LASSO penalty dominates the signal at small n, so the best cross-validated model is the intercept-only baseline. Expanding the ΔPER lambda grid (delta_extended) does not change this for opto_hex/opto_EB, so it is not a grid-resolution artifact.

## Why low-range datasets look “perfect” in raw MSE

Raw MSE is scale-dependent: smaller y_std yields smaller MSE even when relative error is similar. Normalized metrics (nmse and rmse_over_y_std) in `diagnostics/delta_model_comparison.csv` should be used for cross-condition comparisons. This avoids misreading low-variance conditions as “perfect fits.”

## Recommended default model for ΔPER reporting

When LASSO is intercept-only (n_selected=0, pred_std=0, cv_mse==intercept_only_mse), report the best ElasticNet/Ridge by CV MSE. This is already surfaced in `audit_primary_models.csv` in the latest audit run.

## Reproducible commands

```bash
conda run -n DoOR python diagnostics/run_postA_postB_audit.py \
  --door_cache door_cache \
  --behavior_csv "/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions(Strictest)/reaction_rates_summary_unordered.csv" \
  --conditions opto_hex,opto_EB,opto_benz_1,opto_ACV,opto_3-oct \
  --prediction_mode test_odorant \
  --cv_folds 5 \
  --lambda_range 0.0001,0.001,0.01,0.1,1.0 \
  --lambda_range_delta 1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0 \
  --missing_control_policy skip
```