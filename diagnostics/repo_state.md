# Repo State Snapshot

## Commands

### git status -sb
```
## feature/lasso-subtract-control...origin/feature/lasso-subtract-control
```

### git log -n 20 --oneline
```
64c79ce feat: Implement control subtraction for LASSO behavioral prediction and add corresponding tests
3225982 Merge pull request #2 from colehanan1/feature/lasso-ablation-analysis
1ed3db4 refactor: Save ablation_summary and comparison plot to ablations/ subfolder
8183dbc docs: Add ablation and focus mode CLI usage to documentation
4278a80 feat: Add LASSO focus mode analysis for receptor circuit sufficiency
026dad9 feat: Add LASSO ablation analysis for receptor circuit robustness
bc9234f feat: Add support for strict mode in connectome analysis and deprecate Shapley importance method in favor of Shapley-proxy
e5574e8 Merge pull request #1 from colehanan1/audit/codex_repo_analysis
1b83066 feat: Add synthetic importance audit scripts for connectome, GLM, LASSO, and Shapley methods
8093664 Add threshold calibration utilities and corresponding tests
bd12a4b feat: Update .gitignore to include 'outputs/', 'helper-code/', and 'flywire_orn_database/' directories
6f84d36 feat: Update .gitignore to include 'outputs/' and '.claude/' directories
3ac9598 Release v1.0.1
8ca9bf7 Add comprehensive test suites for mapping accounting and identifier resolution
862c0f2 Add receptor sensitivity diagnostics script
4219faa Release v1.0.0: Production-ready toolkit with mushroom body circuit validation
49ba71a feat: Add Mushroom Body Circuit Validation module and update README with new features
19b029b feat: Add FlyWire Mushroom Body Pathway Analysis script and Mushroom Body Tracer module
458cf39 Add comprehensive documentation for behavioral prediction analysis, connectomics module, custom pathway guide, and FlyWire integration notes
3b5f197 Add LASSO regression-based behavioral prediction and enhance existing predictor
```

### git diff
```
<no working tree diff>
```

### git diff --stat
```
<no working tree diff>
```

## Changed Files Relevant to Drift Investigation

### Behavioral prediction core
- `src/door_toolkit/pathways/behavioral_prediction.py`
  - Commit `64c79ce` adds control-subtraction in `fit_behavior`, plus dataset name normalization helpers and new metadata fields.
  - Helper utilities for ablation/focus mode are in this file (see `apply_receptor_ablation`, `fit_lasso_with_fixed_scaler`, `restrict_to_receptors`).

### Ablation/focus scripts
- `scripts/lasso_with_ablations.py` and `scripts/lasso_with_focus_mode.py` show no diffs in this branch vs main (branch equals main at `64c79ce`).

### Helpers for X construction / scaling / lambda selection
- `LassoBehavioralPredictor.fit_behavior()` in `behavioral_prediction.py` constructs X via `_extract_*` helpers.
- Scaling uses `StandardScaler.fit_transform` (new arrays, no in-place mutation).
- Lambda selection uses `LassoCV(random_state=42)` and `cross_val_score` with deterministic folds.

## Commit Stats (64c79ce)
```
docs/BEHAVIORAL_PREDICTION_ANALYSIS.md             |  20 ++
scripts/run_lasso_behavioral_prediction.py         | 239 +++++++++++++++++++++
src/door_toolkit/pathways/behavioral_prediction.py | 156 +++++++++++++-
tests/test_lasso_behavioral_prediction.py          |  86 ++++++++
```
