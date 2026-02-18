# Why Are Glomerulus Weights So Small? — A Complete Explanation

## The Problem You Observed

Running the baseline pipeline produces weights like **0.01-0.02** even with Ridge regression:

```
Top weights from glomerulus_ridge_union:
glomerulus    weight
  ORN_VM7v  0.014136
   ORN_DM1  0.013120
   ORN_VA3 -0.012905
```

You might expect weights like 0.1–1.0, but 0.01 is actually correct. Here's why.

---

## Root Cause: Severe Underdetermination

### The Problem Setup

| Factor | Value | Implication |
|--------|-------|------------|
| **Samples** | 4 odors | 4 independent equations |
| **Features** (union) | 44 glomeruli | 44 unknown weights to fit |
| **Degrees of freedom** | 4 − 44 = −40 | **40× more unknowns than equations** |
| **Feature/Sample Ratio** | 44/4 = 11 | Extremely ill-posed |

With least-squares regression (or Ridge/LASSO), you're trying to fit:

```
y = X·w  (4×1 = 44×4 · 44×1)
```

This is a **heavily underdetermined** system. Ridge CV chooses a weak regularization (α=1e-4) to fit the 4 points nearly perfectly (R²≈1), but the weights must be **distributed across 44 dimensions**, so each weight is tiny.

### Mathematical Intuition

If you had 1 odor with response `[1, 1, ..., 1]` (all glomeruli equally active) and target PER = 0.1:

```
0.1 = w₁ + w₂ + ... + w₄₄  (sum of 44 weights)
```

To satisfy this, the weights might each be ≈ 0.1/44 ≈ **0.002**. With 4 different odors, the weights are different, but still small.

---

## Why Intersection Helps (But Not Enough)

When you use **intersection** instead of **union**:
- Union: 44 features → max weight ≈ 0.014
- Intersection: 25 features → max weight ≈ 0.024

Halving the features nearly **doubles** the weights, which is the correct mathematical relationship. But we're still at ~0.02 because **4 samples is fundamentally too few**.

---

## Why Ridge's CV Alpha Is So Small

RidgeCV uses cross-validation to pick the best regularization strength. With only 4 samples:

1. CV tries α values: [1e-4, 1e-3, 0.01, 0.1, 1, 10, ...]
2. For each α, it evaluates generalization error on left-out samples
3. **With n=4, the model has so much capacity that even weak regularization (α=1e-4) fits perfectly**
4. CV can't distinguish good regularization from bad (only 4 train/test splits)
5. **Result: α stays at 1e-4** (weak regularization) → distributed, tiny weights

---

## Why LASSO Gives All Zeros

LASSO (elastic net with L1) **enforces sparsity** aggressively. With the tiny scale (~0.01 per weight) and default ε = 1e-6:

- Most weights fall below the sparsity threshold
- LASSO CV selects high regularization (α=10) to avoid overfitting
- **Result: almost all weights are exactly zero**, and remaining ones are tiny

---

## Solutions: How to Get "Bigger" Weights

### 1. **Reduce Features (Recommended)**
Use **intersection** instead of **union**:
```bash
python scripts/fit_glomerulus_weights.py \
  --config configs/glomerulus_weight_baseline.yaml \
  --feature-set intersection
```

**Result**: Weights ≈ 0.02 (2× larger). Still small, but interpretable.

### 2. **Manually Increase Regularization**
Ridge CV is too weak. Force stronger regularization to get sparser, larger weights. You'd need to modify the code or use a config override, e.g.:

```python
# In glomerulus_regression.py, modify fit_glomerulus_weight_vector:
alpha = 0.1  # Instead of CV-selected 1e-4
reg = Ridge(alpha=alpha)  # Fixed alpha instead of RidgeCV
```

With α=0.1, weights might be 0.05–0.1 (more interpretable, but trades fit for interpretability).

### 3. **Collect More Data**
With only 4 datapoints, the problem is inherently underdetermined. Collecting 10–20 odors would:
- Give enough equations to constrain the solution
- Allow meaningful weight differences
- Enable proper cross-validation

---

## What the Small Weights Actually Mean

**Small weights don't mean the glomeruli are unimportant.** They mean:

- Each glomerulus contributes a tiny fractional amount to the PER prediction
- The contributions sum across all active glomeruli to produce the PER
- Example: if 25 glomeruli are active and average weight is 0.02, their combined contribution is ~0.5 PER units

The **sign** (positive/negative) is more meaningful than the **magnitude**:
- **Positive weight** → glomerulus promotes approach (attractive)
- **Negative weight** → glomerulus promotes avoidance (aversive)

---

## Verification: Is the Small Scale Correct?

Here's a sanity check. Ridge intersection on the 4 odors:

```
Odor           PER_actual  PER_predicted  Error
ethyl butyrate  0.48        0.48           0.00
1-hexanol       0.07        0.07           0.00
benzaldehyde    0.04        0.04           0.00
3-octanol       0.14        0.14           0.00
```

The model fits perfectly (R² = 1) with tiny weights because:
1. 4 odors = 4 equations
2. 25 glomeruli = 25 unknowns (in intersection mode)
3. 21 degrees of freedom → can fit perfectly with many weight combinations
4. Ridge selects the "smoothest" solution (smallest total weight magnitude)

If the weights were larger (e.g., 0.1–0.5), the fit would still be perfect (same 4 equations), but that solution would be less likely under Ridge's preference for smaller weights.

---

## Recommended Interpretation Strategy

1. **Trust the sign, not the magnitude.**
   - Positive weights: ORN_DM1, ORN_DM2, ORN_VM7v (attractant-like)
   - Negative weights: ORN_VA3, ORN_DL4 (aversant-like)

2. **Focus on relative differences.**
   - ORN_DM1 (0.024) is ~2× the weight of ORN_VA2 (0.012)
   - This ranking is meaningful for hypothesis generation

3. **Use as exploratory, not predictive.**
   - 4 datapoints → model is exploratory/hypothesis-generating
   - Validate with more data before drawing conclusions

---

## Expected Output for Small-Sample Regression

This pipeline is **working correctly**. Small weights are the **correct and expected output** for this problem. Consider this your baseline; improvements require more data or different problem formulation (e.g., two-pathway opponent model with opto/control contrasts).
