# Smoke Test Summary - Tier 0

## Configuration
- **Constraint Tier**: 0 (readout only)
- **KC Sparsity**: 30 active KCs (5%)
- **Training**: 3 epochs, lr=1e-3
- **Splits**: 89 train / 29 val / 31 test flies

## Trainable Parameters (Tier 0)
1. `readout.weight`: [1, 608]
2. `readout.bias`: [1]

**Total**: 609 trainable parameters

## Results

### First Epoch
- Train: loss=0.631, AUROC=0.419, AUPRC=0.123
- Val: loss=0.575, AUROC=0.331, AUPRC=0.092

### Last Epoch (Epoch 3)
- Train: loss=0.505, AUROC=0.431, AUPRC=0.124
- Val: loss=0.468, AUROC=0.382, AUPRC=0.099

### Test Set (Final)
- **Loss**: 0.527
- **AUROC**: 0.390
- **AUPRC**: 0.157
- **Balanced Accuracy**: 0.500
- **KC Sparsity**: 4.93% active (30.0 KCs/trial)

## Verification
- Loss decreasing: ✓
- KC sparsity maintained at 5%: ✓
- Fly-wise splits: 89+29+31 = 149 flies total ✓
- No leakage between splits: ✓

## Next Steps
To fully train and compare tiers:

```bash
# Tier 0 (baseline)
python scripts/train_static_door_rnn.py \
    --output_dir outputs/tier0_full \
    --constraint_tier 0 \
    --epochs 50

# Tier 1 (+gains)
python scripts/train_static_door_rnn.py \
    --output_dir outputs/tier1_full \
    --constraint_tier 1 \
    --epochs 50

# Tier 2 (+recurrence)
python scripts/train_static_door_rnn.py \
    --output_dir outputs/tier2_full \
    --constraint_tier 2 \
    --epochs 100 \
    --lr 5e-4
```
