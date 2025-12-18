"""
Tests for Sequence Dataset and Fly-Wise Splits

Critical validation for publication-grade code:
1. Verify fly-wise splits have no overlap (no leakage)
2. Verify trials are temporally ordered within each fly
3. Verify all flies are accounted for across splits
4. Verify dataset correctly filters by fly_ids
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import json
from pathlib import Path
import torch

from door_toolkit.data.sequence_dataset import (
    FlySequenceDataset,
    FlySequenceLoader,
    create_fly_wise_splits,
    verify_no_leakage,
)


@pytest.fixture
def mock_trial_data():
    """Create mock trial data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock data
        n_flies = 10
        trials_per_fly = 15
        n_trials = n_flies * trials_per_fly

        # Create mock features and labels
        features = torch.randn(n_trials, 237)
        labels = torch.randint(0, 2, (n_trials,)).float()
        torch.save(features, tmpdir / 'trial_features.pt')
        torch.save(labels, tmpdir / 'trial_labels.pt')

        # Create mock metadata with temporal ordering
        metadata_records = []
        idx = 0
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
                idx += 1

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

        yield tmpdir, n_flies, trials_per_fly


def test_fly_wise_splits_no_overlap(mock_trial_data):
    """Test that fly-wise splits have no overlapping flies."""
    tmpdir, n_flies, _ = mock_trial_data
    metadata_path = tmpdir / 'trial_metadata.csv'

    train_flies, val_flies, test_flies = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
    )

    # Verify no overlap
    assert verify_no_leakage(train_flies, val_flies, test_flies)

    # Check set intersections manually
    train_set = set(train_flies)
    val_set = set(val_flies)
    test_set = set(test_flies)

    assert len(train_set & val_set) == 0, "Train/val overlap detected"
    assert len(train_set & test_set) == 0, "Train/test overlap detected"
    assert len(val_set & test_set) == 0, "Val/test overlap detected"


def test_fly_wise_splits_coverage(mock_trial_data):
    """Test that all flies are assigned to exactly one split."""
    tmpdir, n_flies, _ = mock_trial_data
    metadata_path = tmpdir / 'trial_metadata.csv'
    metadata = pd.read_csv(metadata_path)

    train_flies, val_flies, test_flies = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=0.7,
        val_frac=0.15,
        test_frac=0.15,
        seed=42,
    )

    # All flies should be in exactly one split
    all_split_flies = set(train_flies) | set(val_flies) | set(test_flies)
    all_metadata_flies = set(metadata['fly_group'].unique())

    assert all_split_flies == all_metadata_flies, \
        "Not all flies are assigned to splits"


def test_fly_wise_splits_proportions(mock_trial_data):
    """Test that split proportions are approximately correct."""
    tmpdir, n_flies, _ = mock_trial_data
    metadata_path = tmpdir / 'trial_metadata.csv'

    train_frac = 0.6
    val_frac = 0.2
    test_frac = 0.2

    train_flies, val_flies, test_flies = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=train_frac,
        val_frac=val_frac,
        test_frac=test_frac,
        seed=42,
    )

    # Check proportions (allow ±1 fly tolerance due to rounding)
    expected_train = int(n_flies * train_frac)
    expected_val = int(n_flies * val_frac)

    assert abs(len(train_flies) - expected_train) <= 1
    assert abs(len(val_flies) - expected_val) <= 1


def test_dataset_temporal_ordering(mock_trial_data):
    """Test that trials are temporally ordered within each fly."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data

    dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(tmpdir / 'trial_metadata.csv'),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
    )

    # Check each fly's sequence is ordered
    for fly_id in dataset.fly_ids:
        fly_seq = dataset.get_fly_sequence(fly_id)
        trial_nums = fly_seq['trial_nums'].numpy()

        # Trial numbers should be strictly increasing
        assert np.all(trial_nums[1:] > trial_nums[:-1]), \
            f"Trials not ordered for {fly_id}: {trial_nums}"


def test_dataset_fly_filtering(mock_trial_data):
    """Test that dataset correctly filters by fly_ids."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data
    metadata_path = tmpdir / 'trial_metadata.csv'

    # Create splits
    train_flies, val_flies, test_flies = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
    )

    # Create train dataset
    train_dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(metadata_path),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
        fly_ids=train_flies,
    )

    # Verify only train flies are present
    assert set(train_dataset.fly_ids) == set(train_flies)

    # Verify correct number of trials
    expected_trials = len(train_flies) * trials_per_fly
    assert len(train_dataset) == expected_trials


def test_dataset_fly_boundaries(mock_trial_data):
    """Test that fly boundaries are correctly computed."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data

    dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(tmpdir / 'trial_metadata.csv'),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
    )

    # Check each fly has correct number of trials
    for fly_id, (start, end) in dataset.fly_boundaries.items():
        n_trials = end - start
        assert n_trials == trials_per_fly, \
            f"Fly {fly_id} has {n_trials} trials, expected {trials_per_fly}"


def test_fly_sequence_loader(mock_trial_data):
    """Test FlySequenceLoader yields complete fly sequences."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data

    dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(tmpdir / 'trial_metadata.csv'),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
    )

    loader = FlySequenceLoader(dataset, shuffle=False)

    sequences = list(loader)

    # Should have one sequence per fly
    assert len(sequences) == n_flies

    # Each sequence should have correct length
    for seq in sequences:
        assert seq['features'].shape[0] == trials_per_fly
        assert seq['labels'].shape[0] == trials_per_fly


def test_fly_sequence_loader_shuffle(mock_trial_data):
    """Test FlySequenceLoader shuffles fly order."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data

    dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(tmpdir / 'trial_metadata.csv'),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
    )

    loader_noshuffle = FlySequenceLoader(dataset, shuffle=False, seed=42)
    loader_shuffle = FlySequenceLoader(dataset, shuffle=True, seed=43)

    # Collect fly_ids in order
    flies_noshuffle = [seq['fly_id'] for seq in loader_noshuffle]
    flies_shuffle = [seq['fly_id'] for seq in loader_shuffle]

    # Should have same flies but different order (with high probability)
    assert set(flies_noshuffle) == set(flies_shuffle)
    assert flies_noshuffle != flies_shuffle  # Different order (probabilistic)


def test_dataset_shapes(mock_trial_data):
    """Test dataset returns correct shapes."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data

    dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(tmpdir / 'trial_metadata.csv'),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
    )

    # Test single item access
    item = dataset[0]
    assert item['features'].shape == (237,)
    assert item['label'].shape == ()

    # Test fly sequence access
    fly_seq = dataset.get_fly_sequence(dataset.fly_ids[0])
    assert fly_seq['features'].shape == (trials_per_fly, 237)
    assert fly_seq['labels'].shape == (trials_per_fly,)


def test_feature_schema_validation(mock_trial_data):
    """Test that feature schema is correctly validated."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data

    # This should work without errors
    dataset = FlySequenceDataset(
        features_path=str(tmpdir / 'trial_features.pt'),
        labels_path=str(tmpdir / 'trial_labels.pt'),
        metadata_path=str(tmpdir / 'trial_metadata.csv'),
        feature_schema_path=str(tmpdir / 'feature_metadata.json'),
    )

    # Test profile indices extraction
    test_profile_indices = dataset.get_test_profile_indices()
    assert len(test_profile_indices) == 78
    assert min(test_profile_indices) >= 0
    assert max(test_profile_indices) < 237


def test_verify_no_leakage_catches_overlap():
    """Test that verify_no_leakage catches overlapping flies."""
    train_flies = ['fly_0', 'fly_1', 'fly_2']
    val_flies = ['fly_3', 'fly_4']
    test_flies = ['fly_2', 'fly_5']  # Overlap with train

    with pytest.raises(ValueError, match='Train/test overlap'):
        verify_no_leakage(train_flies, val_flies, test_flies)


def test_split_fractions_sum_to_one():
    """Test that create_fly_wise_splits validates fraction sum."""
    tmpdir = tempfile.mkdtemp()
    metadata = pd.DataFrame({
        'fly_group': ['fly_0', 'fly_1'],
        'trial_num': [1, 1],
    })
    metadata_path = Path(tmpdir) / 'metadata.csv'
    metadata.to_csv(metadata_path, index=False)

    # Should raise error for invalid fractions
    with pytest.raises(AssertionError):
        create_fly_wise_splits(
            metadata_path=str(metadata_path),
            train_frac=0.6,
            val_frac=0.2,
            test_frac=0.3,  # Sum > 1.0
        )


def test_dataset_reproducibility(mock_trial_data):
    """Test that splits are reproducible with same seed."""
    tmpdir, n_flies, trials_per_fly = mock_trial_data
    metadata_path = tmpdir / 'trial_metadata.csv'

    # Create splits twice with same seed
    splits1 = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
    )
    splits2 = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
    )

    # Should be identical
    assert splits1[0] == splits2[0]  # train
    assert splits1[1] == splits2[1]  # val
    assert splits1[2] == splits2[2]  # test


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
