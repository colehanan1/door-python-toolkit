# Training Preflight Documentation

## Overview

This document describes the preflight validation system that ensures the connectome-constrained RNN model trains on the correct adult receptor set with consistent ordering across all artifacts.

**Key Principle**: Fail fast on inconsistencies. Never train on misaligned data.

---

## 1. Training Receptor Set Definition

### Decision
Define training set as adult + mapped + non-ambiguous receptors by default.

### Evidence
- **Larval Exclusion**: Larval receptors target different brain circuits (Berck et al., eLife 2016; Jovanic et al., Current Biology 2016). Including them would mix developmental stages.
- **Mapping Requirement**: Unmapped receptors have no FlyWire connectivity constraints, defeating the model's purpose.
- **Ambiguity Exclusion**: Ambiguous mappings (e.g., Or33b → ORN_DM3 OR ORN_DM5) introduce uncertainty in connectome constraints.

### Implementation
- Script: `scripts/build_training_receptor_set.py`
- Output: `data/mappings/training_receptor_set.json`
- Filters applied:
  1. `life_stage == "Adult"`
  2. `is_mapped == "Yes"`
  3. `is_ambiguous == "No"` (unless `--include_ambiguous` flag)
  4. `flywire_target` starts with "ORN_*"

**Result**: 55 receptors (from 78 total DoOR receptors)
- Excluded: 11 larval, 12 unmapped/ambiguous

---

## 2. Receptor Ordering Consistency

### Decision
Validate receptor ordering across training set, connectivity matrices, and feature schema at training startup. Fail if inconsistent.

### Evidence
Silent ordering mismatches are a major source of bugs in neuroscience ML models (Botvinick et al., Trends Cogn Sci 2020). Better to crash early with a clear error than train on misaligned data that produces spurious results.

### Implementation
- Validator: `src/door_toolkit/data/training_receptor_validator.py`
- Function: `validate_receptor_ordering()`
- Checks:
  1. Training receptors are subset of connectivity receptors
  2. Index mapping is correct (`receptor_indices_in_connectivity`)
  3. No larval receptors in training set
  4. No unmapped receptors in training set
  5. Feature schema dimensions match connectivity matrices

**Usage**:
```python
from door_toolkit.data.training_receptor_validator import validate_receptor_ordering

report = validate_receptor_ordering(
    training_set_path="data/mappings/training_receptor_set.json",
    connectivity_metadata_path="data/pgcn_features/connectivity/connectivity_metadata.json",
    strict=True,  # Fail on any mismatch
)
```

---

## 3. Overfit Test

### Decision
Train model on 3-5 flies for 100 epochs. Expect near-perfect performance.

### Evidence
If a model cannot overfit a tiny dataset, there's a bug in:
- Model architecture
- Data loading
- Loss function
- Optimizer

This is a standard debugging tool in ML (Karpathy, Recipe for Training Neural Networks 2019).

### Implementation
- Script: `scripts/preflight_train_checks.py --tiny_overfit`
- Success criteria:
  - Train loss < 0.2
  - Train AUROC > 0.9
  - Train balanced accuracy > 0.85

**Failure modes**:
- Cannot overfit → Check model gradients, data alignment, loss function
- Overfits slowly → Check learning rate, model capacity

---

## 4. Label Shuffle Test

### Decision
Shuffle labels randomly and train for 10 epochs. Expect chance-level performance (~0.5 AUROC).

### Evidence
If model performs better than chance on shuffled labels, there's:
- Data leakage (e.g., labels encoded in features)
- Label-independent features (e.g., model predicting class imbalance)

This test catches common data bugs (Karpathy 2019; Lipton & Steinhardt, ICML 2018).

### Implementation
- Script: `scripts/preflight_train_checks.py --label_shuffle`
- Success criteria:
  - Val AUROC in [0.4, 0.6] (chance level)
  - Val balanced accuracy in [0.4, 0.6]

**Failure modes**:
- High performance on shuffled labels → Check for leakage, remove label-encoding features

---

## 5. Baseline Comparisons

### Decision
Train logistic regression baseline on same fly-wise splits as RNN.

### Evidence
Scientific claims require demonstrating that model complexity is justified (Lipton & Steinhardt, ICML 2018). If logistic regression performs similarly, the RNN's complexity isn't earning its keep.

### Implementation
- Script: `scripts/train_baseline_models.py`
- Baseline: sklearn LogisticRegression with balanced class weights
- Same data splits (fly-wise, no leakage)

**Interpretation**:
- RNN >> Baseline → Complexity justified (temporal dynamics matter)
- RNN ≈ Baseline → Consider simpler models first

---

## 6. Provenance Tracking

### Decision
Save SHA256 hashes of all critical artifacts (mapping, inventory, connectivity matrices) in config files.

### Evidence
Reproducibility requires exact artifact tracking. If connectivity matrices are accidentally modified, hashes will catch it (Hutson, Science 2018; Pineau et al., NeurIPS 2020 reproducibility checklist).

### Implementation
- Training config includes:
  - `inventory_hash`
  - `mapping_hash`
  - `connectivity_orn_pn_hash`
  - `connectivity_pn_kc_hash`
  - `training_receptor_set_hash`
- Stored in: `outputs/<run_name>/config.json`

---

## Files and Scripts

### Generation Scripts
1. `scripts/generate_complete_receptor_mapping.py` → `data/mappings/door_to_flywire_mapping.csv`
2. `scripts/generate_receptor_inventory.py` → `data/mappings/receptor_inventory.csv`
3. `scripts/build_training_receptor_set.py` → `data/mappings/training_receptor_set.json`

### Validation and Preflight
4. `scripts/preflight_train_checks.py` → Overfit + label shuffle tests
5. `src/door_toolkit/data/training_receptor_validator.py` → Ordering validation

### Training
6. `scripts/train_static_door_rnn.py` → Main training script (should integrate validation)
7. `scripts/train_baseline_models.py` → Baseline comparisons

---

## Running Preflight Pipeline

### Full Preflight Sequence
```bash
# Step 1: Regenerate mapping and inventory
python scripts/generate_complete_receptor_mapping.py
python scripts/generate_receptor_inventory.py

# Step 2: Build training receptor set
python scripts/build_training_receptor_set.py

# Step 3: Run preflight checks
python scripts/preflight_train_checks.py \
    --constraint_tier 2 \
    --tiny_overfit \
    --label_shuffle \
    --output_dir outputs/preflight

# Step 4: Train baselines
python scripts/train_baseline_models.py \
    --output_dir outputs/baselines

# Step 5: Train RNN (with validation integrated)
python scripts/train_static_door_rnn.py \
    --constraint_tier 0 \
    --output_dir outputs/tier0_validated \
    --epochs 50
```

---

## Acceptance Criteria

Before full-scale training, all of the following must pass:

- [ ] Training receptor set has 55 receptors (adult + mapped + non-ambiguous)
- [ ] All 55 receptors found in connectivity matrices
- [ ] Receptor ordering validation passes (no mismatches)
- [ ] Overfit test passes (loss < 0.2, AUROC > 0.9)
- [ ] Label shuffle test passes (AUROC ~0.5)
- [ ] Baseline trained on same splits
- [ ] No larval receptors in training set
- [ ] Provenance hashes saved in config

If any criterion fails, **DO NOT proceed with full training**. Debug first.

---

## References

- Berck, M. E. et al. (2016). The wiring diagram of a glomerular olfactory system. *eLife*, 5, e14859.
- Botvinick, M. et al. (2020). Deep Reinforcement Learning and Its Neuroscientific Implications. *Trends in Cognitive Sciences*, 24(2), 125-138.
- Hutson, M. (2018). Artificial intelligence faces reproducibility crisis. *Science*, 359(6377), 725-726.
- Jovanic, T. et al. (2016). Competitive Disinhibition Mediates Behavioral Choice and Sequences in *Drosophila*. *Current Biology*, 26(16), 2087-2097.
- Karpathy, A. (2019). A Recipe for Training Neural Networks. [Blog post]
- Lipton, Z. C., & Steinhardt, J. (2018). Troubling Trends in Machine Learning Scholarship. *ICML 2018 Debates*.
- Pineau, J. et al. (2020). Improving Reproducibility in Machine Learning Research. *NeurIPS 2020*.

---

## Contact

For questions about preflight validation, see:
- `docs/CONNECTOME_RNN_MODEL.md` for model design decisions
- `tests/test_training_receptor_set.py` for test examples
- GitHub issues for bug reports
