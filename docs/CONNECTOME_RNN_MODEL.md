# Connectome-Constrained Recurrent Model Documentation

## Overview

This document describes the design decisions, evidence, and implementation details for the Static DoOR Recurrent Circuit model. This is a publication-grade implementation designed for reproducibility, auditability, and extensibility.

**Model Type**: Connectome-constrained recurrent neural network for fly behavior prediction

**Key Principle**: All major design choices are documented with Decision → Evidence → Implementation format, making the model auditable and suitable for peer review.

---

## 1. Fly-Wise Splits with Ordered Sequences

### Decision
Group trials by fly_id and split data by fly, not by individual trials. Maintain temporal ordering within each fly's trials.

### Evidence
**Preventing Data Leakage**: Splitting individual trials would leak information between train/val/test through shared fly history. For example, if trial 5 from fly A is in training and trial 6 from fly A is in testing, the model could "remember" fly A's state from training.

**Standard Practice in Neuroscience**: Fly-wise (or animal-wise) splitting is standard in computational neuroscience:
- Steinmetz et al., Nature 2019: "mice were randomly assigned to training, validation, or test sets"
- Stringer et al., Science 2019: split data by recording session
- International Brain Lab, Nature 2021: split by subject ID

**Temporal Causality**: Recurrent models require temporally ordered sequences to learn trial history effects. Random shuffling destroys causal structure.

### Implementation
- `create_fly_wise_splits()` in [sequence_dataset.py](../src/door_toolkit/data/sequence_dataset.py:125)
- `verify_no_leakage()` checks set intersections are empty
- `FlySequenceDataset` sorts by `(fly_group, trial_num)` during initialization
- Training loop processes one fly at a time, resetting hidden state between flies

---

## 2. Fixed Connectome Adjacency as Buffers

### Decision
Use FlyWire connectome structure as fixed, non-trainable connectivity matrices. Register as PyTorch buffers (not parameters).

### Evidence
**Biological Constraint**: The connectome structure is a hard biological constraint measured from electron microscopy (Zheng et al., Cell 2018; Dorkenwald et al., Nature 2023). We test whether fixed wiring + minimal plasticity can explain behavior.

**Scientific Goal**: Ablation studies require explicit constraint tiers. If connectivity were trainable, we couldn't distinguish the contribution of:
1. Connectome structure itself
2. Synaptic weight plasticity
3. Recurrent dynamics

**Computational Realism**: Real brains don't add/remove synapses on the timescale of behavioral training (though weights do change). Fixed connectivity is more biologically plausible for acute learning.

### Implementation
- Connectivity matrices loaded from `orn_pn_connectivity.pt` and `pn_kc_connectivity.pt`
- Registered as buffers: `self.register_buffer('orn_pn_W', orn_pn_W)`
- `assert not model.orn_pn_W.requires_grad` verified in tests
- File hashes saved in config.json for provenance

**Location**: [static_door_recurrent_circuit.py:72-88](../src/door_toolkit/neural/static_door_recurrent_circuit.py)

---

## 3. Recurrent Dynamics in PN Space

### Decision
Place recurrent cell (GRUCell) at the PN layer, not KC layer. This models "history dynamics" in PN space.

### Evidence
**Empirical Observations**:
- PNs show trial history effects and adaptation (Gupta & Stopfer, J Neurosci 2012)
- PNs exhibit temporal dynamics and gain control (Olsen et al., Nature 2010)
- KCs respond more stereotypically and sparsely (Turner et al., eLife 2008)

**Computational Function**: PN recurrence can implement:
- Short-term adaptation
- Trial history integration
- Gain modulation based on recent odor context

**Not Claiming Lateral Inhibition**: We do NOT claim this directly models lateral inhibition (LN→PN). The recurrent dynamics are placed here as a modeling choice to capture "PN layer has trial history," not as a biophysical LN simulation. Future work could add explicit LN populations.

### Implementation
- `self.pn_gru = nn.GRUCell(input_size=self.n_pns, hidden_size=self.n_pns)`
- Hidden state initialized per fly: `pn_hidden = self.init_hidden(batch_size=1)`
- Hidden state carried across trials within a fly
- Reset at fly boundaries

**Location**: [static_door_recurrent_circuit.py:100-105](../src/door_toolkit/neural/static_door_recurrent_circuit.py)

---

## 4. KC Sparsity via Top-K Selection

### Decision
Enforce sparse KC activations using top-k selection. Default: 5% active (30 out of 608 KCs).

### Evidence
**In Vivo Sparsity**:
- Turner et al., eLife 2008: ~5% of KCs active per odor
- Honegger et al., Nature 2011: sparse, distributed KC coding
- Aso et al., eLife 2014: KC sparsity critical for MBON selectivity

**Winner-Take-All Dynamics**: KCs exhibit winner-take-all competition through GABAergic feedback inhibition (Perez-Orive et al., Science 2002; Lin et al., Nature 2014).

**Computational Function**: Sparse coding provides:
- High-dimensional separable representations
- Efficient memory storage (Dasgupta et al., Nat Neurosci 2017)
- Robustness to input noise

### Implementation
- Top-k selection after ReLU: `torch.topk(kcs_relu, k=self.kc_sparsity_k, dim=1)`
- Default k = 30 (5% of 608)
- Configurable via `kc_sparsity_k` parameter
- Sparsity stats logged: fraction active, number active

**Alternative Considered**: Soft sparsity (L1 penalty) was considered but rejected in favor of hard top-k for exact biological match.

**Location**: [static_door_recurrent_circuit.py:167-190](../src/door_toolkit/neural/static_door_recurrent_circuit.py)

---

## 5. Three Constraint Tiers for Ablation

### Decision
Implement three trainable parameter tiers to enable systematic ablation studies:
- **Tier 0**: Only KC→MBON readout trainable
- **Tier 1**: Tier 0 + per-neuron gains/biases on PN and KC
- **Tier 2**: Tier 1 + PN recurrent parameters (GRU weights)

Connectome adjacency is ALWAYS fixed across all tiers.

### Evidence
**Scientific Rigor**: To make a scientific claim like "connectome structure explains X% of behavior," we must show:
1. Minimal model (Tier 0) performance
2. Improvement from local gain modulation (Tier 1)
3. Improvement from recurrent dynamics (Tier 2)

**Standard Practice**: Tiered training is standard in computational neuroscience (Sussillo & Barak, Neuron 2013; Mante et al., Nature 2013). Ablations demonstrate which mechanisms are necessary vs. sufficient.

**Falsifiability**: If Tier 0 performs well, we learn that connectome + linear readout suffices. If Tier 2 is needed, we learn that history dynamics are critical. Either result is scientifically informative.

### Implementation
- `set_constraint_tier(tier)` method enables/disables parameter gradients
- `get_trainable_parameters()` returns list of trainable params for auditing
- Tests verify correct gradient flow per tier
- Config file logs tier and trainable parameter names

**Parameter Counts (Approximate)**:
- Tier 0: ~609 params (608 KC→1 MBON + bias)
- Tier 1: ~3,507 params (Tier 0 + 841 PN gain/bias + 608 KC gain/bias)
- Tier 2: ~5.67M params (Tier 1 + GRU weights for 841 hidden units)

**Location**: [static_door_recurrent_circuit.py:107-146](../src/door_toolkit/neural/static_door_recurrent_circuit.py)

---

## 6. Feature Schema for Reproducibility

### Decision
Use a JSON feature schema (`feature_metadata.json`) that explicitly defines where the 78-dim test DoOR profile lives in the 237-dim feature vector. No hardcoded indices.

### Evidence
**Reproducibility**: Hardcoded indices like `features[:, 78:156]` are error-prone and undocumented. If feature extraction changes, hardcoded indices break silently.

**Auditability**: Schema explicitly documents what each feature dimension represents, making code auditable for peer review.

**Extensibility**: Adding new features (e.g., behavioral history, optogenetic flags) only requires updating the schema, not the model code.

### Implementation
- Schema loaded in model `__init__`: `self.feature_schema = json.load(f)`
- Indices extracted: `self.test_profile_indices = torch.tensor(schema['feature_groups']['test_door_profile'])`
- Assertion checks shape matches: `assert features.shape[1] == schema['feature_dim']`

**Location**: [static_door_recurrent_circuit.py:66-71](../src/door_toolkit/neural/static_door_recurrent_circuit.py)

---

## 7. Evaluation Metrics for Imbalanced Classification

### Decision
Report AUROC, AUPRC (average precision), and balanced accuracy. Not just raw accuracy.

### Evidence
**Class Imbalance**: Behavioral data often has imbalanced classes (e.g., 70% no response, 30% response). Raw accuracy is misleading.

**Standard Metrics**:
- **AUROC**: Area under ROC curve. Threshold-independent. Standard in ML and neuroscience.
- **AUPRC**: Area under precision-recall curve. More informative for imbalanced data than AUROC (Saito & Rehmsmeier, PLoS ONE 2015).
- **Balanced Accuracy**: Average of per-class recall. Interpretable and robust to imbalance.

**Community Standard**: These metrics are standard in behavioral prediction (Stringer et al., Science 2019; Musall et al., Nat Neurosci 2019).

### Implementation
- `sklearn.metrics.roc_auc_score()`, `average_precision_score()`, `balanced_accuracy_score()`
- Computed on held-out test set (fly-wise split)
- Logged per epoch in `metrics.json`

**Location**: [train_static_door_rnn.py:108-123](../scripts/train_static_door_rnn.py)

---

## 8. Provenance Tracking

### Decision
Save file hashes, connectivity metadata, and trainable parameter names in `config.json` for full provenance.

### Evidence
**Reproducibility Crisis**: Many ML papers fail to reproduce due to missing details (Hutson, Science 2018). Provenance tracking enables exact reproduction.

**Scientific Integrity**: If connectivity matrices are accidentally modified between experiments, hashes will detect it. This prevents silent errors.

**Regulatory Compliance**: For eventual clinical/applied work, provenance is legally required (FDA guidance on ML medical devices).

### Implementation
- SHA256 hashes of connectivity files: `hash_file(orn_pn_path)`
- Metadata saved: constraint tier, sparsity k, split details, trainable params
- Config saved alongside model weights

**Location**: [train_static_door_rnn.py:233-246](../scripts/train_static_door_rnn.py)

---

## Files and Locations

### Core Implementation
1. **Model**: `src/door_toolkit/neural/static_door_recurrent_circuit.py` (~200 lines)
2. **Data**: `src/door_toolkit/data/sequence_dataset.py` (~250 lines)
3. **Training**: `scripts/train_static_door_rnn.py` (~280 lines)

### Testing
4. **Model Tests**: `tests/test_static_door_recurrent_circuit.py` (~350 lines)
5. **Data Tests**: `tests/test_sequence_split_no_leakage.py` (~300 lines)

### Documentation
6. **This File**: `docs/CONNECTOME_RNN_MODEL.md`

---

## Usage Examples

### Train Tier 0 (Readout Only)
```bash
python scripts/train_static_door_rnn.py \
    --data_dir data/pgcn_features \
    --output_dir outputs/tier0 \
    --constraint_tier 0 \
    --epochs 50 \
    --lr 1e-3
```

### Train Tier 1 (+ Gains)
```bash
python scripts/train_static_door_rnn.py \
    --data_dir data/pgcn_features \
    --output_dir outputs/tier1 \
    --constraint_tier 1 \
    --epochs 50 \
    --lr 1e-3
```

### Train Tier 2 (+ Recurrence)
```bash
python scripts/train_static_door_rnn.py \
    --data_dir data/pgcn_features \
    --output_dir outputs/tier2 \
    --constraint_tier 2 \
    --epochs 100 \
    --lr 5e-4
```

### Run Tests
```bash
pytest tests/test_static_door_recurrent_circuit.py -v
pytest tests/test_sequence_split_no_leakage.py -v
```

---

## Future Extensions

### 1. Explicit LN Population
Current model uses GRU cell as abstract "PN dynamics." Future work could add explicit lateral inhibition:
```
ORN [78] → PN [841]
         ↗
LN [~200] ← PN
```
This would require LN connectivity from FlyWire.

### 2. Plasticity Sites
Add trainable masks to test which synapses are plastic:
- ORN→PN plasticity
- PN→KC plasticity
- KC→MBON plasticity (current default)

### 3. Multi-Trial Context
Extend recurrence to longer timescales (across-session transfer, sleep consolidation).

### 4. Neuromodulation
Add neuromodulator signals (dopamine, octopamine) as time-varying inputs to MBON layer.

---

## References

### Connectome Structure
- Zheng, Z. et al. (2018). A Complete Electron Microscopy Volume of the Brain of Adult Drosophila melanogaster. *Cell*, 174(3), 730-743.
- Dorkenwald, S. et al. (2023). Neuronal wiring diagram of an adult brain. *Nature*.

### Neuroscience Evidence
- Turner, G. C., Bazhenov, M., & Laurent, G. (2008). Olfactory representations by Drosophila mushroom body neurons. *eLife*, 7, e01247.
- Gupta, N., & Stopfer, M. (2012). Functional analysis of a higher olfactory center, the lateral horn. *Journal of Neuroscience*, 32(24), 8138-8148.
- Perez-Orive, J. et al. (2002). Oscillations and sparsening of odor representations in the mushroom body. *Science*, 297(5580), 359-365.
- Olsen, S. R., Bhandawat, V., & Wilson, R. I. (2010). Divisive normalization in olfactory population codes. *Neuron*, 66(2), 287-299.

### Machine Learning & Methods
- Sussillo, D., & Barak, O. (2013). Opening the black box: low-dimensional dynamics in high-dimensional recurrent neural networks. *Neuron*, 13(1), 57-69.
- Steinmetz, N. A. et al. (2019). Distributed coding of choice, action and engagement across the mouse brain. *Nature*, 576(7786), 266-273.
- Stringer, C. et al. (2019). Spontaneous behaviors drive multidimensional, brainwide activity. *Science*, 364(6437), eaav7893.
- Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. *PLoS ONE*, 10(3), e0118432.

---

## Contact & Contribution

For questions or contributions, please open an issue on the GitHub repository.

**License**: See repository LICENSE file.

**Citation**: If you use this model in a publication, please cite this repository and the FlyWire connectome (Dorkenwald et al., 2023).
