# Preflight Diagnosis Report

## Run Command
`python scripts/preflight_train_checks.py --constraint_tier 2 --tiny_overfit --label_shuffle`

## Tiny Overfit Subset Class Balance

- Flies: `december_05_batch_1_2, november_05_batch_2_1, november_07_batch_1_2`
- Trials: 30 | Pos: 15 | Neg: 15

| fly_id | pos | neg | only_one_class |
|---|---:|---:|:---:|
| `december_05_batch_1_2` | 5 | 5 |  |
| `november_05_batch_2_1` | 5 | 5 |  |
| `november_07_batch_1_2` | 5 | 5 |  |

## Legacy Tiny Subset (Pre-fix Selection)

- Legacy selection method: `legacy_first_k_after_shuffle(create_fly_wise_splits(seed))`
- Flies: `november_18_batch_1_3, december_09_batch_1_3, october_24_batch_1_rig_2_2`
- Trials: 30 | Pos: 3 | Neg: 27
- Flies with only one class: `october_24_batch_1_rig_2_2`

| fly_id | pos | neg | only_one_class |
|---|---:|---:|:---:|
| `november_18_batch_1_3` | 2 | 8 |  |
| `december_09_batch_1_3` | 1 | 9 |  |
| `october_24_batch_1_rig_2_2` | 0 | 10 | YES |

## Overfit Metrics (Best Train-Loss Epoch)

- Loss: 0.0012 (gate: < 0.05)
- AUROC: 1.0000
- AUPRC: 1.0000
- BalAcc@0.5: 1.0000 (fixed_threshold=0.5)
- BalAcc@opt: 1.0000 (optimized_threshold=0.5001; gate: >= 0.95)

## Overfit Metric Computation

- Loss: `BCEWithLogitsLoss` on training set (no `pos_weight`).
- Fixed balanced accuracy: threshold `0.5` applied to sigmoid probabilities.
- Optimized balanced accuracy: threshold chosen to maximize balanced accuracy on the tiny training set.

## Why The Original Overfit Gate Failed

- The old check gated on **balanced accuracy at a fixed 0.5 threshold**. On tiny, imbalanced subsets, the model can rank positives above negatives (high AUROC) while still keeping all probabilities below 0.5, yielding BalAcc≈0.5 (all-negative predictions).
- The previous tiny-subset selection took the **first K flies after a shuffle**, which can include flies with only one class (e.g., all-negative), amplifying threshold/imbalance issues.

## What Changed (Decision → Evidence → Implementation)

- Decision: Make the overfit preflight a memorization sanity check, not a calibration check.
- Evidence: Fixed-threshold BalAcc can be 0.5 even with high AUROC on imbalanced tiny-N.
- Implementation:
  - Added loss-based memorization gate (train_loss < 0.05 by default).
  - Added optimized-threshold balanced accuracy gate (BalAcc@opt ≥ 0.95 by default).
  - Made tiny subset fly selection deterministic + class-count constrained (min pos/neg).

## Next Steps

- [ ] Re-run preflight for tiers 0/1/2 with desired settings
- [ ] If all pass, proceed to full training runs
- [ ] If overfit fails, inspect diagnostic block for class balance and gate values

## Label Shuffle Summary

- AUROC: 0.4760 | BalAcc@0.5: 0.5000 | Passed: True

