"""
Manual test script to verify model correctness.
"""

import sys
import json
import tempfile
from pathlib import Path
import torch
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, 'src')

from door_toolkit.neural.static_door_recurrent_circuit import StaticDoORRecurrentCircuit
from door_toolkit.data.sequence_dataset import (
    FlySequenceDataset,
    FlySequenceLoader,
    create_fly_wise_splits,
    verify_no_leakage,
)

def create_mock_data(tmpdir):
    """Create mock data for testing."""
    # Create mock data
    n_flies = 10
    trials_per_fly = 15
    n_trials = n_flies * trials_per_fly

    # Create mock features and labels
    features = torch.randn(n_trials, 237)
    labels = torch.randint(0, 2, (n_trials,)).float()
    torch.save(features, tmpdir / 'trial_features.pt')
    torch.save(labels, tmpdir / 'trial_labels.pt')

    # Create mock metadata
    metadata_records = []
    for fly_num in range(n_flies):
        fly_group = f'fly_{fly_num}'
        for trial_num in range(trials_per_fly):
            metadata_records.append({
                'fly': f'batch_{fly_num // 2}',
                'fly_number': fly_num,
                'trial_num': trial_num + 1,
                'condition': 'test',
                'trained_odor': 'odorA',
                'test_odor': 'odorB',
                'trial_within_fly': trial_num / trials_per_fly,
                'learning_history': 0.0,
                'prob_reaction': np.random.rand(),
                'fly_group': fly_group,
            })
    metadata = pd.DataFrame(metadata_records)
    metadata.to_csv(tmpdir / 'trial_metadata.csv', index=False)

    # Create feature schema
    feature_schema = {
        'feature_dim': 237,
        'feature_groups': {
            'test_door_profile': list(range(78, 156))
        }
    }
    with open(tmpdir / 'feature_metadata.json', 'w') as f:
        json.dump(feature_schema, f)

    # Create mock connectivity metadata
    connectivity_metadata = {
        'n_receptors': 78,
        'n_pns': 841,
        'n_kcs': 608,
    }
    with open(tmpdir / 'connectivity_metadata.json', 'w') as f:
        json.dump(connectivity_metadata, f)

    # Create mock connectivity matrices
    orn_pn_W = torch.randn(78, 841) * 0.1
    pn_kc_W = torch.randn(841, 608) * 0.1
    torch.save(orn_pn_W, tmpdir / 'orn_pn_connectivity.pt')
    torch.save(pn_kc_W, tmpdir / 'pn_kc_connectivity.pt')

    return n_flies, trials_per_fly


def test_model_initialization(tmpdir):
    """Test 1: Model initialization."""
    print("Test 1: Model initialization...", end=" ")
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )
    assert model.n_orns == 78
    assert model.n_pns == 841
    assert model.n_kcs == 608
    print("PASSED")


def test_forward_shapes(tmpdir):
    """Test 2: Forward pass shapes."""
    print("Test 2: Forward pass shapes...", end=" ")
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )
    features = torch.randn(5, 237)
    logits, pn_hidden, activations = model.forward(features)
    assert logits.shape == (5, 1)
    assert pn_hidden.shape == (5, 841)
    print("PASSED")


def test_constraint_tiers(tmpdir):
    """Test 3: Constraint tiers."""
    print("Test 3: Constraint tier gradients...", end=" ")

    # Tier 0
    model0 = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )
    trainable0 = [name for name, _ in model0.get_trainable_parameters()]
    assert 'readout.weight' in trainable0
    assert 'pn_gain' not in trainable0

    # Tier 1
    model1 = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=1,
    )
    trainable1 = [name for name, _ in model1.get_trainable_parameters()]
    assert 'readout.weight' in trainable1
    assert 'pn_gain' in trainable1

    # Tier 2
    model2 = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=2,
    )
    trainable2 = [name for name, _ in model2.get_trainable_parameters()]
    assert any('pn_gru' in name for name in trainable2)

    print("PASSED")


def test_connectivity_buffers(tmpdir):
    """Test 4: Connectivity matrices are buffers."""
    print("Test 4: Connectivity buffers (no grad)...", end=" ")
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=2,
    )
    assert not model.orn_pn_W.requires_grad
    assert not model.pn_kc_W.requires_grad
    print("PASSED")


def test_kc_sparsity(tmpdir):
    """Test 5: KC sparsity enforcement."""
    print("Test 5: KC sparsity enforcement...", end=" ")
    kc_sparsity_k = 30
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=0,
        kc_sparsity_k=kc_sparsity_k,
    )
    features = torch.randn(5, 237)
    logits, _, activations = model.forward(features)
    kcs_sparse = activations['kcs_sparse']
    n_active = (kcs_sparse > 0).sum(dim=1)
    assert torch.all(n_active == kc_sparsity_k)
    print("PASSED")


def test_fly_splits(tmpdir, n_flies):
    """Test 6: Fly-wise splits have no overlap."""
    print("Test 6: Fly-wise splits (no leakage)...", end=" ")
    metadata_path = tmpdir / 'trial_metadata.csv'
    train_flies, val_flies, test_flies = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
    )
    verify_no_leakage(train_flies, val_flies, test_flies)
    train_set = set(train_flies)
    val_set = set(val_flies)
    test_set = set(test_flies)
    assert len(train_set & val_set) == 0
    assert len(train_set & test_set) == 0
    assert len(val_set & test_set) == 0
    print("PASSED")


def test_dataset_ordering(tmpdir, trials_per_fly):
    """Test 7: Trials are temporally ordered."""
    print("Test 7: Temporal ordering within flies...", end=" ")
    dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(tmpdir / 'trial_metadata.csv'),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
    )
    for fly_id in dataset.fly_ids:
        fly_seq = dataset.get_fly_sequence(fly_id)
        trial_nums = fly_seq['trial_nums'].numpy()
        assert np.all(trial_nums[1:] > trial_nums[:-1])
    print("PASSED")


def test_backward_pass(tmpdir):
    """Test 8: Backward pass works."""
    print("Test 8: Backward pass...", end=" ")
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(tmpdir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(tmpdir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(tmpdir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )
    features = torch.randn(3, 237)
    labels = torch.tensor([0.0, 1.0, 0.0])
    logits, _, _ = model.forward(features)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(), labels
    )
    loss.backward()
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad
    print("PASSED")


def main():
    print("\n=== Running Manual Model Tests ===\n")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        print(f"Using temp directory: {tmpdir}\n")

        # Create mock data
        n_flies, trials_per_fly = create_mock_data(tmpdir)

        # Run tests
        test_model_initialization(tmpdir)
        test_forward_shapes(tmpdir)
        test_constraint_tiers(tmpdir)
        test_connectivity_buffers(tmpdir)
        test_kc_sparsity(tmpdir)
        test_fly_splits(tmpdir, n_flies)
        test_dataset_ordering(tmpdir, trials_per_fly)
        test_backward_pass(tmpdir)

        print("\n=== All Tests PASSED ===\n")


if __name__ == '__main__':
    main()
