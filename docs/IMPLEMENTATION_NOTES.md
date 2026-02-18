# Glomerulus Weight Baseline Pipeline — Implementation Notes

## Overview

This document summarizes the implementation of a minimal, glomerulus-level single weight vector regression pipeline for baseline/control PER behavioral labels.

## Key Decisions & Trade-Offs

### 1. Ridge (not LASSO) as Default Model

**Decision**: Use Ridge regression as default; LASSO available but not recommended for 4 samples.

**Rationale**:
- With 4 samples and 25+ features, LASSO CV selects α >> 1, zeroing all weights
- Ridge CV selects weak α ≈ 1e-4, producing nonzero, meaningful weights
- For small-sample problems, smooth regularization (Ridge) beats aggressive sparsity (LASSO)
- User can still try LASSO/ElasticNet via CLI: `--model lasso`

### 2. Intersection Mode as Default Feature Set

**Decision**: Use intersection (glomeruli active in ALL 4 odors) as default; union/all available.

**Rationale**:
- Intersection: 25 features → max |weight| ≈ 0.024
- Union: 44 features → max |weight| ≈ 0.014
- Intersection is more interpretable (only "core" active glomeruli)
- Larger weights easier to rank and discuss
- User can explore union/all via CLI: `--feature-set union`

### 3. Small Weights (0.01–0.02) Are Correct

**Decision**: Document and explain why weights are small; do NOT artificially increase them.

**Rationale**:
- 4 samples (equations) + 25 features (unknowns) = 21 degrees of freedom
- Ridge distributes weights smoothly across all dimensions
- Mathematical consequence: each weight ≈ 0.01–0.02
- This is **correct behavior**, not a bug
- See `GLOMERULUS_WEIGHTS_EXPLANATION.md` for full math

### 4. Focus on Sign, Not Magnitude

**Decision**: Emphasize sign classification (+/−/0) over absolute weight values.

**Rationale**:
- With 4 samples, magnitude unreliable; sign more robust
- ORN_DM1 (+) means "attracts" in relative terms
- ORN_VA3 (−) means "avoids" in relative terms
- Ranking by |weight| is meaningful for prioritization
- Absolute magnitude meaningful only with 10+ samples

## Implementation Details

### Code Organization

```
src/door_toolkit/
├── glomerulus_features.py
│   ├── load_receptor_to_glomerulus_mapping()
│   ├── odor_to_receptor_vector()
│   ├── receptor_vector_to_glomerulus_vector()
│   └── build_design_matrix()
│
├── glomerulus_regression.py
│   ├── fit_glomerulus_weight_vector()
│   └── export_weight_report()
│
└── cli_glomerulus.py
    └── glomerulus_main()

scripts/
└── fit_glomerulus_weights.py  (CLI wrapper)

configs/
└── glomerulus_weight_baseline.yaml  (4 odors + defaults)

tests/
└── test_glomerulus_pipeline.py  (20 unit tests, synthetic fixtures)
```

### API Design Principles

1. **No Internet Dependency**: All tests use synthetic fixtures; DoOR data cached locally.
2. **Deterministic**: Fixed `random_state=0` ensures reproducible runs.
3. **Configurable**: YAML config + CLI flags override mechanism.
4. **Exportable**: CSV + JSON output; easily parsed/visualized.
5. **Reusable**: Components designed for future 2-pathway opponent model.

## Reused Code

All implementations follow existing repo patterns:

| Component | Reused From | Evidence |
|-----------|------------|----------|
| DoOR encoding | `encoder.py:126-173` | `DoOREncoder.encode()` |
| Receptor→glomerulus mapping | `data/mappings/door_to_flywire_mapping.csv` | 60 receptors → 49 glomeruli |
| Regression pattern | `behavioral_prediction.py:538-1338` | sklearn LassoCV + export structure |
| CLI pattern | `cli_pathways.py`, `cli_neural.py` | argparse + YAML config |
| Test pattern | `tests/` | pytest + synthetic fixtures |

## Output Format

### weights.csv

```csv
glomerulus,weight,sign,abs_weight,rank
ORN_DM1,0.023823,+1,0.023823,1
ORN_VA3,-0.023526,-1,0.023526,2
...
```

- **glomerulus**: FlyWire ORN label (e.g., ORN_DM1)
- **weight**: Regression coefficient (−0.024 to +0.024)
- **sign**: +1 (attractant), −1 (aversant), 0 (zero)
- **abs_weight**: Absolute magnitude (for ranking)
- **rank**: Ranked by |weight| descending (1 = most important)

### model_summary.json

```json
{
  "model": "ridge",
  "alpha": 1.00e-04,
  "r2": 1.0000,
  "mse": 4.62e-13,
  "n_samples": 4,
  "n_features": 25,
  "n_positive": 11,
  "n_negative": 14,
  "n_zero": 0,
  "config": { ... }
}
```

- **alpha**: CV-selected regularization strength
- **r2**: Coefficient of determination (perfect fit with 4 samples)
- **n_positive/negative/zero**: Sign distribution of weights
- **config**: Full pipeline metadata (odors, PER labels, thresholds, etc.)

## Verification

### Tests (20/20 passing)

```bash
pytest tests/test_glomerulus_pipeline.py -v -p no:napari -p no:npe2
```

Coverage:
- Mapping loading and filtering
- Receptor→glomerulus aggregation (max/mean/sum)
- Design matrix construction (all/union/intersection)
- Regression fitting and sign classification
- Export (CSV + JSON)
- Determinism (same seed → same output)

### Reproducibility

```bash
# Run 1
python scripts/fit_glomerulus_weights.py --config configs/glomerulus_weight_baseline.yaml --outdir out/test1

# Run 2
python scripts/fit_glomerulus_weights.py --config configs/glomerulus_weight_baseline.yaml --outdir out/test2

# Should be identical
diff out/test1/model_summary.json out/test2/model_summary.json  # (no output = same)
```

### Feature Sets Validation

| Mode | Features | Max |w| | + | − | 0 |
|------|----------|---------|---|---|---|
| Intersection | 25 | 0.024 | 11 | 14 | 0 |
| Union | 44 | 0.014 | 20 | 24 | 0 |
| All | 49 | 0.014 | 20 | 24 | 5 |

All produce valid outputs with expected feature counts.

## Future Extensions (Out of Scope)

### Two-Pathway Opponent Model

Currently: Single weight vector from baseline PER.

Future:
- Separate excitatory (approach) and inhibitory (avoidance) vectors
- Use opto/control PER contrasts
- Requires (~20+ odor pairs) for meaningful separation

### Temporal Dynamics

Currently: Static per-odor regression.

Future:
- Trial-by-trial variability
- Latency/rise-time effects
- Concentration dependence

### Integration with Connectomics

Currently: Only receptor→glomerulus mapping.

Future:
- Glomerulus→LNs (lateral neurons)
- GABA-mediated inhibition
- Higher-order circuit analysis

## Known Limitations & Workarounds

### Limitation 1: Only 4 Odors

**Issue**: Severe underdetermination (4 equations, 25 unknowns).

**Consequence**: Small weights, perfect R² fit, limited generalization.

**Workaround**:
1. Collect 10+ odors for larger, more stable weights
2. Use stronger manual regularization (e.g., `Ridge(alpha=0.1)`)
3. Focus on sign patterns, not absolute magnitudes

### Limitation 2: LASSO All-Zeros

**Issue**: LASSO CV selects high α for small n.

**Consequence**: All weights exactly zero, no signal.

**Workaround**: Use Ridge (default) instead. LASSO better for larger n (10+).

### Limitation 3: Some Glomeruli Unmapped

**Issue**: Not all 78 DoOR receptors map to glomeruli.

**Consequence**: Adult-only mapping covers 60/78 receptors.

**Workaround**: Handled by load_receptor_to_glomerulus_mapping(). Unmapped receptors logged in metadata.

## Lessons Learned

### For Similar Projects

1. **Small-sample regression**: Ridge > LASSO (unless n >> p)
2. **Underdetermined systems**: Expect small weights; focus on signs/rankings
3. **Perfect R²**: Expected with n << p; not a sign of good generalization
4. **Aggregation**: max() best for receptor→glomerulus (typically one dominant receptor per glomerulus)
5. **Reproducibility**: Always fix `random_state=0` in CV-based models

### For This Repo

1. DoOREncoder is robust for multi-odor encoding (fast, flexible)
2. Mapping CSV is comprehensive but needs continued curation (some ambiguities flagged)
3. Test fixtures with synthetic data + real data integration points valuable
4. YAML configs + CLI paradigm good for exploratory analysis pipelines

## References

- **DoOR Database**: Münch & Galizia 2016 (https://doi.org/10.1038/srep21841)
- **FlyWire Connectome**: Schlegel et al. 2023 (https://doi.org/10.1038/s41586-023-06173-7)
- **LASSO vs Ridge**: Hastie et al. 2009, "The Elements of Statistical Learning"
- **Small-sample ML**: Ng 2004 "Feature selection for SVMs"

---

**Last Updated**: 2026-02-17
**Status**: ✅ Complete & Tested
**Next PR**: Two-pathway opponent model (post-review)
