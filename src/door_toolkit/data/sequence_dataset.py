"""
Sequence Dataset for Fly Trials with Proper Temporal Ordering

Decision → Evidence → Implementation
-----------------------------------
Decision: Group trials by fly_id and sort by trial index to maintain temporal order.
Evidence: Recurrent models require ordered sequences to learn trial history. Random
         shuffling would destroy temporal dependencies and violate causality.
Implementation: FlySequenceDataset yields (features, labels, fly_id, trial_idx) tuples
                sorted by fly_id and trial_idx.

Decision: Split data by fly_id, not by individual trials.
Evidence: Splitting individual trials would leak information between train/val/test
         through shared fly history. Standard practice in neuroscience (Steinmetz et al.,
         Nature 2019; Stringer et al., Science 2019).
Implementation: create_fly_wise_splits() ensures each fly appears in exactly one split.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class FlySequenceDataset(Dataset):
    """
    Dataset that yields fly sequences in temporal order.

    Each item is a single trial with metadata for sequence reconstruction.
    Use FlySequenceLoader to batch trials within each fly.
    """

    def __init__(
        self,
        features_path: str,
        labels_path: str,
        metadata_path: str,
        feature_schema_path: str,
        fly_ids: Optional[List[str]] = None,
    ):
        """
        Args:
            features_path: Path to trial_features.pt [N, 237]
            labels_path: Path to trial_labels.pt [N]
            metadata_path: Path to trial_metadata.csv
            feature_schema_path: Path to feature_metadata.json
            fly_ids: If provided, only include trials from these flies
        """
        self.features = torch.load(features_path, weights_only=False)
        self.labels = torch.load(labels_path, weights_only=False)
        self.metadata = pd.read_csv(metadata_path)

        # Load feature schema
        with open(feature_schema_path) as f:
            self.feature_schema = json.load(f)

        # Validate shapes
        assert self.features.shape[0] == len(self.labels), \
            f"Features ({self.features.shape[0]}) and labels ({len(self.labels)}) shape mismatch"
        assert self.features.shape[0] == len(self.metadata), \
            f"Features ({self.features.shape[0]}) and metadata ({len(self.metadata)}) shape mismatch"
        assert self.features.shape[1] == self.feature_schema['feature_dim'], \
            f"Features dim {self.features.shape[1]} != schema dim {self.feature_schema['feature_dim']}"

        # Filter by fly_ids if provided
        if fly_ids is not None:
            fly_ids_set = set(fly_ids)
            mask = self.metadata['fly_group'].isin(fly_ids_set)
            self.features = self.features[mask]
            self.labels = self.labels[mask]
            self.metadata = self.metadata[mask].reset_index(drop=True)

        # Sort by fly_group and trial_num for temporal ordering
        sort_idx = self.metadata.sort_values(
            ['fly_group', 'trial_num']
        ).index.values
        self.features = self.features[sort_idx]
        self.labels = self.labels[sort_idx]
        self.metadata = self.metadata.iloc[sort_idx].reset_index(drop=True)

        # Create fly boundaries for efficient sequence extraction
        self.fly_boundaries = {}
        for fly_id, group in self.metadata.groupby('fly_group', sort=False):
            start_idx = group.index[0]
            end_idx = group.index[-1] + 1
            self.fly_boundaries[fly_id] = (start_idx, end_idx)

        self.fly_ids = list(self.fly_boundaries.keys())

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return single trial with metadata for sequence reconstruction."""
        return {
            'features': self.features[idx],
            'label': self.labels[idx],
            'fly_id': self.metadata.loc[idx, 'fly_group'],
            'trial_num': self.metadata.loc[idx, 'trial_num'],
        }

    def get_fly_sequence(self, fly_id: str) -> Dict[str, torch.Tensor]:
        """
        Get all trials for a specific fly in temporal order.

        Returns:
            Dict with keys:
                - features: [seq_len, feature_dim]
                - labels: [seq_len]
                - fly_id: str
                - trial_nums: [seq_len]
        """
        if fly_id not in self.fly_boundaries:
            raise ValueError(f"Unknown fly_id: {fly_id}")

        start_idx, end_idx = self.fly_boundaries[fly_id]
        return {
            'features': self.features[start_idx:end_idx],
            'labels': self.labels[start_idx:end_idx],
            'fly_id': fly_id,
            'trial_nums': torch.tensor(
                self.metadata.loc[start_idx:end_idx-1, 'trial_num'].values
            ),
        }

    def get_test_profile_indices(self) -> List[int]:
        """Get indices for test DoOR profile (78 dimensions)."""
        return self.feature_schema['feature_groups']['test_door_profile']


def create_fly_wise_splits(
    metadata_path: str,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create train/val/test splits by fly_id to prevent leakage.

    Decision: Split by fly_id, not by trials.
    Evidence: Prevents information leakage through shared history within a fly.
    Implementation: Randomly assign each unique fly to train/val/test.

    Args:
        metadata_path: Path to trial_metadata.csv
        train_frac: Fraction of flies for training
        val_frac: Fraction of flies for validation
        test_frac: Fraction of flies for testing
        seed: Random seed for reproducibility

    Returns:
        train_fly_ids: List of fly IDs for training
        val_fly_ids: List of fly IDs for validation
        test_fly_ids: List of fly IDs for testing
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"

    metadata = pd.read_csv(metadata_path)
    unique_flies = sorted(metadata['fly_group'].unique())

    # Shuffle flies
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_flies)

    # Split
    n_flies = len(unique_flies)
    n_train = int(n_flies * train_frac)
    n_val = int(n_flies * val_frac)

    train_fly_ids = unique_flies[:n_train]
    val_fly_ids = unique_flies[n_train:n_train + n_val]
    test_fly_ids = unique_flies[n_train + n_val:]

    return train_fly_ids, val_fly_ids, test_fly_ids


class FlySequenceLoader:
    """
    Loader that yields complete fly sequences one at a time.

    Decision: Iterate one fly at a time (batch_size=1 fly) for correctness.
    Evidence: Simplifies hidden state management. Can optimize later with padded batching.
    Implementation: Iterator over flies, yielding complete sequences.
    """

    def __init__(self, dataset: FlySequenceDataset, shuffle: bool = False, seed: int = 42):
        """
        Args:
            dataset: FlySequenceDataset instance
            shuffle: Whether to shuffle fly order each epoch
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        fly_ids = self.dataset.fly_ids.copy()
        if self.shuffle:
            self.rng.shuffle(fly_ids)

        for fly_id in fly_ids:
            yield self.dataset.get_fly_sequence(fly_id)

    def __len__(self):
        return len(self.dataset.fly_ids)


def verify_no_leakage(
    train_fly_ids: List[str],
    val_fly_ids: List[str],
    test_fly_ids: List[str],
) -> bool:
    """
    Verify that fly_ids are disjoint across splits.

    Decision: Explicitly verify no overlap between splits.
    Evidence: Critical safety check for publication-grade code.
    Implementation: Check set intersections are empty.
    """
    train_set = set(train_fly_ids)
    val_set = set(val_fly_ids)
    test_set = set(test_fly_ids)

    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set

    if train_val_overlap:
        raise ValueError(f"Train/val overlap: {train_val_overlap}")
    if train_test_overlap:
        raise ValueError(f"Train/test overlap: {train_test_overlap}")
    if val_test_overlap:
        raise ValueError(f"Val/test overlap: {val_test_overlap}")

    return True
