"""
Training harness for Static DoOR Recurrent Circuit

Decision → Evidence → Implementation
-----------------------------------
Decision: Process sequences one fly at a time, maintaining hidden state within fly.
Evidence: Avoids batching complexity while ensuring correctness. Can optimize later.
Implementation: Loop over flies, reset hidden state at fly boundaries.

Decision: Use BCEWithLogitsLoss for binary classification.
Evidence: Numerically stable (combines sigmoid + BCE). Standard for binary tasks.
Implementation: Model outputs logits, loss applies sigmoid internally.

Decision: Track AUROC, AUPRC, balanced accuracy.
Evidence: AUROC/AUPRC handle class imbalance. Balanced accuracy is interpretable.
         Standard metrics in neuroscience (Stringer et al., Science 2019).
Implementation: sklearn metrics on held-out test set.
"""

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
)

from door_toolkit.data.sequence_dataset import (
    FlySequenceDataset,
    FlySequenceLoader,
    create_fly_wise_splits,
    verify_no_leakage,
)
from door_toolkit.neural.static_door_recurrent_circuit import (
    StaticDoORRecurrentCircuit,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def hash_file(path: str) -> str:
    """Compute SHA256 hash of file for provenance."""
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def train_epoch(
    model: StaticDoORRecurrentCircuit,
    loader: FlySequenceLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Train for one epoch over all flies.

    Decision: Reset hidden state at each fly boundary.
    Evidence: Each fly is an independent sequence. No shared history across flies.
    Implementation: Call forward_sequence with reset_hidden=True for each fly.
    """
    model.train()
    total_loss = 0.0
    total_trials = 0
    all_preds = []
    all_labels = []

    for fly_seq in loader:
        features = fly_seq['features'].to(device)  # [seq_len, feature_dim]
        labels = fly_seq['labels'].to(device)  # [seq_len]

        optimizer.zero_grad()

        # Forward through sequence (hidden state maintained within fly)
        logits, _ = model.forward_sequence(features, reset_hidden=True)
        logits = logits.squeeze(-1)  # [seq_len]

        # Compute loss
        loss = criterion(logits, labels.float())
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * len(labels)
        total_trials += len(labels)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'loss': total_loss / total_trials,
        'auroc': roc_auc_score(all_labels, all_preds),
        'auprc': average_precision_score(all_labels, all_preds),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds > 0.5),
    }

    return metrics


def eval_epoch(
    model: StaticDoORRecurrentCircuit,
    loader: FlySequenceLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Evaluate on validation or test set."""
    model.eval()
    total_loss = 0.0
    total_trials = 0
    all_preds = []
    all_labels = []
    kc_sparsity_stats = []

    with torch.no_grad():
        for fly_seq in loader:
            features = fly_seq['features'].to(device)
            labels = fly_seq['labels'].to(device)

            logits, activations_list = model.forward_sequence(features, reset_hidden=True)
            logits = logits.squeeze(-1)

            loss = criterion(logits, labels.float())
            total_loss += loss.item() * len(labels)
            total_trials += len(labels)

            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Track sparsity
            for activations in activations_list:
                stats = model.compute_sparsity_stats(activations['kcs_sparse'])
                kc_sparsity_stats.append(stats)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        'loss': total_loss / total_trials,
        'auroc': roc_auc_score(all_labels, all_preds),
        'auprc': average_precision_score(all_labels, all_preds),
        'balanced_acc': balanced_accuracy_score(all_labels, all_preds > 0.5),
        'kc_frac_active': np.mean([s['kc_frac_active'] for s in kc_sparsity_stats]),
        'kc_n_active_mean': np.mean([s['kc_n_active_mean'] for s in kc_sparsity_stats]),
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Train Static DoOR Recurrent Circuit'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/pgcn_features',
        help='Directory containing trial data',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/static_door_rnn',
        help='Directory to save model and metrics',
    )
    parser.add_argument(
        '--constraint_tier',
        type=int,
        default=0,
        choices=[0, 1, 2],
        help='Constraint tier (0=readout only, 1=+gains, 2=+recurrence)',
    )
    parser.add_argument(
        '--kc_sparsity_k',
        type=int,
        default=None,
        help='Number of active KCs (default: 5%% of 608 = 30)',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs',
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
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use',
    )

    args = parser.parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths
    data_dir = Path(args.data_dir)
    features_path = data_dir / 'trial_features.pt'
    labels_path = data_dir / 'trial_labels.pt'
    metadata_path = data_dir / 'trial_metadata.csv'
    feature_schema_path = data_dir / 'feature_metadata.json'
    orn_pn_path = data_dir / 'connectivity' / 'orn_pn_connectivity.pt'
    pn_kc_path = data_dir / 'connectivity' / 'pn_kc_connectivity.pt'
    connectivity_metadata_path = data_dir / 'connectivity' / 'connectivity_metadata.json'

    logger.info("Creating fly-wise splits...")
    train_fly_ids, val_fly_ids, test_fly_ids = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=1.0 - args.train_frac - args.val_frac,
        seed=args.seed,
    )

    # Verify no leakage
    verify_no_leakage(train_fly_ids, val_fly_ids, test_fly_ids)
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

    logger.info(f"Train: {len(train_dataset)} trials, {len(train_dataset.fly_ids)} flies")
    logger.info(f"Val: {len(val_dataset)} trials, {len(val_dataset.fly_ids)} flies")
    logger.info(f"Test: {len(test_dataset)} trials, {len(test_dataset.fly_ids)} flies")

    # Create loaders
    train_loader = FlySequenceLoader(train_dataset, shuffle=True, seed=args.seed)
    val_loader = FlySequenceLoader(val_dataset, shuffle=False)
    test_loader = FlySequenceLoader(test_dataset, shuffle=False)

    # Create model
    logger.info(f"Creating model with constraint tier {args.constraint_tier}...")
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(feature_schema_path),
        orn_pn_connectivity_path=str(orn_pn_path),
        pn_kc_connectivity_path=str(pn_kc_path),
        connectivity_metadata_path=str(connectivity_metadata_path),
        constraint_tier=args.constraint_tier,
        kc_sparsity_k=args.kc_sparsity_k,
        pn_hidden_init='zero',
        device=args.device,
    ).to(args.device)

    # Log trainable parameters
    trainable_params = model.get_trainable_parameters()
    logger.info(f"Trainable parameters ({len(trainable_params)}):")
    for name, param in trainable_params:
        logger.info(f"  {name}: {param.shape}")

    # Optimizer and loss
    optimizer = optim.Adam(
        [p for _, p in trainable_params],
        lr=args.lr,
    )
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    logger.info("Starting training...")
    metrics_history = []

    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_metrics = eval_epoch(model, val_loader, criterion, args.device)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train: loss={train_metrics['loss']:.4f}, "
            f"auroc={train_metrics['auroc']:.4f}, "
            f"auprc={train_metrics['auprc']:.4f} | "
            f"Val: loss={val_metrics['loss']:.4f}, "
            f"auroc={val_metrics['auroc']:.4f}, "
            f"auprc={val_metrics['auprc']:.4f}, "
            f"kc_active={val_metrics['kc_frac_active']:.3f}"
        )

        metrics_history.append({
            'epoch': epoch + 1,
            'train': train_metrics,
            'val': val_metrics,
        })

    # Final test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = eval_epoch(model, test_loader, criterion, args.device)
    logger.info(
        f"Test: loss={test_metrics['loss']:.4f}, "
        f"auroc={test_metrics['auroc']:.4f}, "
        f"auprc={test_metrics['auprc']:.4f}, "
        f"balanced_acc={test_metrics['balanced_acc']:.4f}"
    )

    # Save results
    logger.info("Saving results...")

    # Save model
    torch.save(model.state_dict(), output_dir / 'model.pt')

    # Save config
    config = {
        'model_config': model.get_config(),
        'training_config': vars(args),
        'data_splits': {
            'train_flies': train_fly_ids,
            'val_flies': val_fly_ids,
            'test_flies': test_fly_ids,
            'n_train_trials': len(train_dataset),
            'n_val_trials': len(val_dataset),
            'n_test_trials': len(test_dataset),
        },
        'connectivity_hashes': {
            'orn_pn': hash_file(orn_pn_path),
            'pn_kc': hash_file(pn_kc_path),
        },
        'trainable_parameters': [name for name, _ in trainable_params],
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'history': metrics_history,
            'test': test_metrics,
        }, f, indent=2)

    logger.info(f"Results saved to {output_dir}")
    logger.info("Training complete!")


if __name__ == '__main__':
    main()
