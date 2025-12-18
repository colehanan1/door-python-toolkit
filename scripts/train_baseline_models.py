#!/usr/bin/env python3
"""
Baseline Models for Comparison

Trains simple baselines (Logistic Regression, small MLP) on same fly-wise splits
as the connectome-constrained RNN for fair comparison.

Decision → Evidence → Implementation
-----------------------------------
Decision: Compare RNN against logistic regression and small MLP baselines.
Evidence: Scientific claims require demonstrating that model complexity is justified.
         Baselines establish lower bounds on performance (Lipton & Steinhardt,
         ICML 2018 Troubling Trends in ML).
Implementation: sklearn LogisticRegression + small PyTorch MLP, same splits.
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.data.sequence_dataset import (
    FlySequenceDataset,
    create_fly_wise_splits,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_logistic_regression(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> dict:
    """Train logistic regression baseline."""
    logger.info("Training Logistic Regression...")

    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
    )
    model.fit(train_features, train_labels)

    # Evaluate
    results = {}
    for split_name, features, labels in [
        ('train', train_features, train_labels),
        ('val', val_features, val_labels),
        ('test', test_features, test_labels),
    ]:
        probs = model.predict_proba(features)[:, 1]
        preds = model.predict(features)

        results[split_name] = {
            'auroc': roc_auc_score(labels, probs),
            'auprc': average_precision_score(labels, probs),
            'balanced_acc': balanced_accuracy_score(labels, preds),
        }

    logger.info(f"Test AUROC: {results['test']['auroc']:.4f}")
    return results


def main():
    parser = argparse.ArgumentParser(description='Train baseline models')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/pgcn_features',
        help='Directory containing trial data',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/baselines',
        help='Output directory',
    )
    parser.add_argument(
        '--train_frac',
        type=float,
        default=0.7,
        help='Fraction of flies for training',
    )
    parser.add_argument(
        '--val_frac',
        type=float,
        default=0.15,
        help='Fraction of flies for validation',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed',
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    features_path = data_dir / 'trial_features.pt'
    labels_path = data_dir / 'trial_labels.pt'
    metadata_path = data_dir / 'trial_metadata.csv'
    feature_schema_path = data_dir / 'feature_metadata.json'

    logger.info("Creating fly-wise splits...")
    train_fly_ids, val_fly_ids, test_fly_ids = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=1.0 - args.train_frac - args.val_frac,
        seed=args.seed,
    )

    logger.info(f"Splits: {len(train_fly_ids)} train, {len(val_fly_ids)} val, {len(test_fly_ids)} test flies")

    # Create datasets
    train_dataset = FlySequenceDataset(
        features_path=str(features_path),
        labels_path=str(labels_path),
        metadata_path=str(metadata_path),
        feature_schema_path=str(feature_schema_path),
        fly_ids=train_fly_ids,
    )
    val_dataset = FlySequenceDataset(
        features_path=str(features_path),
        labels_path=str(labels_path),
        metadata_path=str(metadata_path),
        feature_schema_path=str(feature_schema_path),
        fly_ids=val_fly_ids,
    )
    test_dataset = FlySequenceDataset(
        features_path=str(features_path),
        labels_path=str(labels_path),
        metadata_path=str(metadata_path),
        feature_schema_path=str(feature_schema_path),
        fly_ids=test_fly_ids,
    )

    # Extract features and labels as numpy arrays
    train_X = train_dataset.features.numpy()
    train_y = train_dataset.labels.numpy()
    val_X = val_dataset.features.numpy()
    val_y = val_dataset.labels.numpy()
    test_X = test_dataset.features.numpy()
    test_y = test_dataset.labels.numpy()

    logger.info(f"Train: {train_X.shape}, Val: {val_X.shape}, Test: {test_X.shape}")

    # Train logistic regression
    lr_results = train_logistic_regression(
        train_X, train_y, val_X, val_y, test_X, test_y
    )

    # Save results
    results = {
        'logistic_regression': lr_results,
        'data_splits': {
            'n_train_flies': len(train_fly_ids),
            'n_val_flies': len(val_fly_ids),
            'n_test_flies': len(test_fly_ids),
            'n_train_trials': len(train_dataset),
            'n_val_trials': len(val_dataset),
            'n_test_trials': len(test_dataset),
        },
    }

    output_path = output_dir / 'baseline_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to {output_path}")
    logger.info("Baseline training complete!")


if __name__ == '__main__':
    main()
