#!/usr/bin/env python3
"""
Preflight Training Checks - Sanity Tests Before Full Training

This script runs two critical sanity tests:
1. Overfit Test: Train on tiny subset, expect near-perfect performance
2. Label Shuffle Test: Shuffle labels, expect chance-level performance

If either test fails, there's likely a bug in the model/data pipeline.

Decision → Evidence → Implementation
-----------------------------------
Decision: Run overfit and label shuffle tests before full-scale training.
Evidence: These are standard debugging tools in ML (Karpathy, Recipe for Training
         Neural Networks 2019). Overfit failure indicates model/data bugs.
         Shuffle failure indicates leakage or label-independent features.
Implementation: Tiny-data overfit (3-5 flies, high epochs) + label permutation.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.data.sequence_dataset import (
    FlySequenceDataset,
    FlySequenceLoader,
    create_fly_wise_splits,
)
from door_toolkit.data.training_receptor_validator import (
    validate_receptor_ordering,
    print_validation_report,
)
from door_toolkit.neural.static_door_recurrent_circuit import (
    StaticDoORRecurrentCircuit,
)
from door_toolkit.preflight import (
    build_overfit_diagnostics,
    optimal_balanced_accuracy_threshold,
    select_tiny_overfit_flies,
)

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_overfit_test(
    data_dir: Path,
    constraint_tier: int,
    n_flies: int = 3,
    min_pos: int = 5,
    min_neg: int = 5,
    max_flies: int = 8,
    epochs: int = 100,
    lr: float = 1e-3,
    loss_eps: float = 0.05,
    balacc_eps: float = 0.95,
    seed: int = 42,
    device: str = 'cpu',
) -> dict:
    """
    Overfit Test: Train on tiny subset, expect near-perfect performance.

    Success criteria (memorization sanity check):
    (a) Loss-based memorization:
        - train_loss < loss_eps (default 0.05)
    (b) Threshold-optimized balanced accuracy:
        - choose threshold that maximizes balanced accuracy on the tiny *training* set
        - optimized_train_balacc >= balacc_eps (default 0.95)

    Also logs fixed-threshold (0.5) balanced accuracy, AUROC, and AUPRC for diagnosis.

    If test fails, there's likely a bug in model architecture or data loading.
    """
    logger.info("=" * 70)
    logger.info("OVERFIT TEST")
    logger.info("=" * 70)
    logger.info(f"Training on {n_flies} flies for {epochs} epochs")
    logger.info(f"Constraint tier: {constraint_tier}")
    logger.info("")

    # Define paths
    features_path = data_dir / 'trial_features.pt'
    labels_path = data_dir / 'trial_labels.pt'
    metadata_path = data_dir / 'trial_metadata.csv'
    feature_schema_path = data_dir / 'feature_metadata.json'
    orn_pn_path = data_dir / 'connectivity' / 'orn_pn_connectivity.pt'
    pn_kc_path = data_dir / 'connectivity' / 'pn_kc_connectivity.pt'
    connectivity_metadata_path = data_dir / 'connectivity' / 'connectivity_metadata.json'

    # Deterministic tiny subset selection (stratified by class counts across flies)
    metadata_df = pd.read_csv(metadata_path)
    labels_all = torch.load(labels_path, weights_only=False).detach().cpu().numpy()
    if len(metadata_df) != len(labels_all):
        raise ValueError(
            f"Metadata/labels mismatch: {len(metadata_df)} rows vs {len(labels_all)} labels"
        )

    per_fly_class_counts = {}
    for fly_id, idxs in metadata_df.groupby("fly_group").indices.items():
        y = labels_all[np.asarray(list(idxs))]
        per_fly_class_counts[str(fly_id)] = {
            "pos": int(np.sum(y == 1)),
            "neg": int(np.sum(y == 0)),
        }

    legacy_fly_ids, _, _ = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=1.0,
        val_frac=0.0,
        test_frac=0.0,
        seed=seed,
    )
    legacy_fly_ids = legacy_fly_ids[:n_flies]
    legacy_per_fly = {fly_id: per_fly_class_counts.get(fly_id, {"pos": 0, "neg": 0}) for fly_id in legacy_fly_ids}
    legacy_pos = int(sum(int(c.get("pos", 0)) for c in legacy_per_fly.values()))
    legacy_neg = int(sum(int(c.get("neg", 0)) for c in legacy_per_fly.values()))
    legacy_single_class_flies = sorted(
        [fly_id for fly_id, c in legacy_per_fly.items() if int(c.get("pos", 0)) == 0 or int(c.get("neg", 0)) == 0]
    )

    train_fly_ids, final_k = select_tiny_overfit_flies(
        per_fly_class_counts,
        initial_k=n_flies,
        min_pos=min_pos,
        min_neg=min_neg,
        max_k=max_flies,
    )

    if final_k != n_flies:
        logger.info(
            f"Tiny overfit fly count auto-adjusted: requested {n_flies} → using {final_k} "
            f"(min_pos={min_pos}, min_neg={min_neg}, max_flies={max_flies})"
        )
    logger.info(f"Tiny subset flies: {train_fly_ids}")

    # Create dataset
    train_dataset = FlySequenceDataset(
        features_path=str(features_path),
        labels_path=str(labels_path),
        metadata_path=str(metadata_path),
        feature_schema_path=str(feature_schema_path),
        fly_ids=train_fly_ids,
    )

    train_loader = FlySequenceLoader(train_dataset, shuffle=True, seed=seed)
    eval_loader = FlySequenceLoader(train_dataset, shuffle=False)

    logger.info(f"Dataset: {len(train_dataset)} trials from {len(train_fly_ids)} flies")

    # Class balance diagnostics
    per_fly_selected = {fly_id: per_fly_class_counts.get(fly_id, {"pos": 0, "neg": 0}) for fly_id in train_fly_ids}
    flies_with_single_class = sorted([fly_id for fly_id, c in per_fly_selected.items() if int(c.get("pos", 0)) == 0 or int(c.get("neg", 0)) == 0])
    if flies_with_single_class:
        logger.warning(
            f"Tiny subset contains flies with a single class: {flies_with_single_class}"
        )

    # Create model
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(feature_schema_path),
        orn_pn_connectivity_path=str(orn_pn_path),
        pn_kc_connectivity_path=str(pn_kc_path),
        connectivity_metadata_path=str(connectivity_metadata_path),
        constraint_tier=constraint_tier,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    def _evaluate_train() -> dict:
        model.eval()
        total_loss = 0.0
        all_probs: list[float] = []
        all_labels: list[int] = []

        with torch.no_grad():
            for fly_seq in eval_loader:
                features = fly_seq["features"].to(device)
                labels = fly_seq["labels"].to(device)
                logits, _ = model.forward_sequence(features, reset_hidden=True)
                logits = logits.squeeze(-1)
                loss = criterion(logits, labels.float())
                total_loss += float(loss.item()) * len(labels)

                probs = torch.sigmoid(logits)
                all_probs.extend(probs.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().astype(int).tolist())

        avg_loss = total_loss / len(train_dataset)
        y_true = np.asarray(all_labels, dtype=int)
        y_prob = np.asarray(all_probs, dtype=float)

        if np.unique(y_true).size < 2:
            auroc = 0.5
            auprc = float(np.mean(y_true))
        else:
            auroc = float(roc_auc_score(y_true, y_prob))
            auprc = float(average_precision_score(y_true, y_prob))

        fixed_threshold = 0.5
        if np.unique(y_true).size < 2:
            fixed_balacc = 0.5
        else:
            fixed_balacc = float(balanced_accuracy_score(y_true, y_prob >= fixed_threshold))

        opt_threshold, opt_balacc = optimal_balanced_accuracy_threshold(y_true, y_prob)

        return {
            "loss": float(avg_loss),
            "auroc": float(auroc),
            "auprc": float(auprc),
            "fixed_threshold": float(fixed_threshold),
            "fixed_balanced_acc": float(fixed_balacc),
            "optimized_threshold": float(opt_threshold),
            "optimized_balanced_acc": float(opt_balacc),
        }

    # Training loop
    best_loss = float("inf")
    best_metrics: dict = {}

    for epoch in range(epochs):
        model.train()

        for fly_seq in train_loader:
            features = fly_seq['features'].to(device)
            labels = fly_seq['labels'].to(device)

            optimizer.zero_grad()
            logits, _ = model.forward_sequence(features, reset_hidden=True)
            logits = logits.squeeze(-1)

            loss = criterion(logits, labels.float())
            loss.backward()
            optimizer.step()

        metrics = _evaluate_train()

        if epoch % 20 == 0 or epoch == epochs - 1:
            logger.info(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {metrics['loss']:.4f} | "
                f"AUROC: {metrics['auroc']:.4f} | "
                f"AUPRC: {metrics['auprc']:.4f} | "
                f"BalAcc@0.5: {metrics['fixed_balanced_acc']:.4f} | "
                f"BalAcc@opt: {metrics['optimized_balanced_acc']:.4f}"
            )

        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            best_metrics = {
                "epoch": int(epoch + 1),
                **metrics,
            }

    loss_gate_passed = bool(best_metrics["loss"] < loss_eps)
    optimized_balacc_gate_passed = bool(best_metrics["optimized_balanced_acc"] >= balacc_eps)
    success = bool(loss_gate_passed and optimized_balacc_gate_passed)

    logger.info("")
    logger.info(
        "Best metrics: "
        f"Loss={best_metrics['loss']:.4f}, "
        f"AUROC={best_metrics['auroc']:.4f}, "
        f"AUPRC={best_metrics['auprc']:.4f}, "
        f"BalAcc@0.5={best_metrics['fixed_balanced_acc']:.4f}, "
        f"BalAcc@opt={best_metrics['optimized_balanced_acc']:.4f} "
        f"(opt_thr={best_metrics['optimized_threshold']:.4f})"
    )
    logger.info(
        "Overfit gates: "
        f"loss<{loss_eps}={loss_gate_passed}, "
        f"opt_balacc>={balacc_eps}={optimized_balacc_gate_passed} "
        f"→ final={success}"
    )
    logger.info(f"Overfit test: {'✅ PASSED' if success else '❌ FAILED'}")

    diagnostics = build_overfit_diagnostics(
        tiny_subset_flies=train_fly_ids,
        per_fly_class_counts=per_fly_selected,
        fixed_threshold=0.5,
        optimized_threshold=float(best_metrics.get("optimized_threshold", 0.5)),
        loss_gate_passed=bool(loss_gate_passed),
        optimized_balacc_gate_passed=bool(optimized_balacc_gate_passed),
        final_passed=bool(success),
    )
    diagnostics["legacy_tiny_subset"] = {
        "flies": legacy_fly_ids,
        "trials": int(legacy_pos + legacy_neg),
        "pos": int(legacy_pos),
        "neg": int(legacy_neg),
        "per_fly_class_counts": legacy_per_fly,
        "flies_with_single_class": legacy_single_class_flies,
        "selection_method": "legacy_first_k_after_shuffle(create_fly_wise_splits(seed))",
    }

    return {
        'test_name': 'overfit_test',
        'passed': bool(success),
        'n_flies_requested': int(n_flies),
        'n_flies_used': int(len(train_fly_ids)),
        'n_trials': len(train_dataset),
        'epochs': epochs,
        'best_metrics': best_metrics,
        'diagnostics': diagnostics,
        'success_criteria': {
            'loss_eps': float(loss_eps),
            'optimized_balacc_eps': float(balacc_eps),
        },
    }


def run_label_shuffle_test(
    data_dir: Path,
    constraint_tier: int,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = 'cpu',
) -> dict:
    """
    Label Shuffle Test: Shuffle labels, expect chance-level performance.

    Success criteria:
    - Final AUROC ~0.5 (chance level)
    - Final balanced accuracy ~0.5

    If test fails (high performance), there's likely data leakage or
    label-independent features in the model.
    """
    logger.info("=" * 70)
    logger.info("LABEL SHUFFLE TEST")
    logger.info("=" * 70)
    logger.info(f"Training with shuffled labels for {epochs} epochs")
    logger.info(f"Constraint tier: {constraint_tier}")
    logger.info("")

    # Define paths
    features_path = data_dir / 'trial_features.pt'
    labels_path = data_dir / 'trial_labels.pt'
    metadata_path = data_dir / 'trial_metadata.csv'
    feature_schema_path = data_dir / 'feature_metadata.json'
    orn_pn_path = data_dir / 'connectivity' / 'orn_pn_connectivity.pt'
    pn_kc_path = data_dir / 'connectivity' / 'pn_kc_connectivity.pt'
    connectivity_metadata_path = data_dir / 'connectivity' / 'connectivity_metadata.json'

    # Create splits
    train_fly_ids, val_fly_ids, _ = create_fly_wise_splits(
        metadata_path=str(metadata_path),
        train_frac=0.6,
        val_frac=0.2,
        test_frac=0.2,
        seed=42,
    )

    # Load features and labels
    features = torch.load(features_path, weights_only=False)
    labels = torch.load(labels_path, weights_only=False)

    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # SHUFFLE LABELS (within entire dataset)
    shuffled_labels = labels.clone()
    perm = torch.randperm(len(labels))
    shuffled_labels = shuffled_labels[perm]

    # Save shuffled labels to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as f:
        temp_labels_path = f.name
    torch.save(shuffled_labels, temp_labels_path)

    # Create dataset with shuffled labels
    train_dataset = FlySequenceDataset(
        features_path=str(features_path),
        labels_path=temp_labels_path,
        metadata_path=str(metadata_path),
        feature_schema_path=str(feature_schema_path),
        fly_ids=train_fly_ids,
    )

    val_dataset = FlySequenceDataset(
        features_path=str(features_path),
        labels_path=temp_labels_path,
        metadata_path=str(metadata_path),
        feature_schema_path=str(feature_schema_path),
        fly_ids=val_fly_ids,
    )

    train_loader = FlySequenceLoader(train_dataset, shuffle=True, seed=42)
    val_loader = FlySequenceLoader(val_dataset, shuffle=False)

    logger.info(f"Dataset: {len(train_dataset)} train trials, {len(val_dataset)} val trials")

    # Create model
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(feature_schema_path),
        orn_pn_connectivity_path=str(orn_pn_path),
        pn_kc_connectivity_path=str(pn_kc_path),
        connectivity_metadata_path=str(connectivity_metadata_path),
        constraint_tier=constraint_tier,
        device=device,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        for fly_seq in train_loader:
            features_batch = fly_seq['features'].to(device)
            labels_batch = fly_seq['labels'].to(device)

            optimizer.zero_grad()
            logits, _ = model.forward_sequence(features_batch, reset_hidden=True)
            logits = logits.squeeze(-1)

            loss = criterion(logits, labels_batch.float())
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for fly_seq in val_loader:
            features_batch = fly_seq['features'].to(device)
            labels_batch = fly_seq['labels'].to(device)

            logits, _ = model.forward_sequence(features_batch, reset_hidden=True)
            probs = torch.sigmoid(logits.squeeze(-1))

            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels_batch.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auroc = float(roc_auc_score(all_labels, all_preds)) if len(np.unique(all_labels)) > 1 else 0.5
    bal_acc = float(balanced_accuracy_score(all_labels, all_preds > 0.5))

    # Clean up temp file
    Path(temp_labels_path).unlink()

    # Success: performance should be near chance (0.5)
    success = bool(0.4 < auroc < 0.6 and 0.4 < bal_acc < 0.6)

    logger.info("")
    logger.info(f"Final metrics: AUROC={auroc:.4f}, BalAcc={bal_acc:.4f}")
    logger.info(f"Label shuffle test: {'✅ PASSED' if success else '❌ FAILED'}")

    if not success:
        if auroc > 0.6 or bal_acc > 0.6:
            logger.warning("WARNING: Model performs better than chance on shuffled labels!")
            logger.warning("This indicates potential data leakage or label-independent features.")

    return {
        'test_name': 'label_shuffle_test',
        'passed': bool(success),
        'epochs': epochs,
        'final_metrics': {
            'auroc': float(auroc),
            'balanced_acc': float(bal_acc),
        },
        'success_criteria': {
            'auroc_range': [0.4, 0.6],
            'balanced_acc_range': [0.4, 0.6],
        },
    }


def _json_default(obj):
    """Best-effort conversion for numpy/torch scalars in reports."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _write_preflight_diagnosis_report(
    output_dir: Path,
    run_command: str,
    results: dict,
) -> Path:
    report_path = output_dir / "preflight_diagnosis_report.md"

    overfit = next((t for t in results.get("tests", []) if t.get("test_name") == "overfit_test"), None)
    shuffle = next((t for t in results.get("tests", []) if t.get("test_name") == "label_shuffle_test"), None)

    lines: list[str] = []
    lines.append("# Preflight Diagnosis Report")
    lines.append("")
    lines.append("## Run Command")
    lines.append(f"`{run_command}`")
    lines.append("")

    if overfit and overfit.get("diagnostics"):
        diag = overfit["diagnostics"]
        lines.append("## Tiny Overfit Subset Class Balance")
        lines.append("")
        lines.append(f"- Flies: `{', '.join(diag.get('tiny_subset_flies', []))}`")
        lines.append(
            f"- Trials: {diag.get('tiny_subset_trials')} | "
            f"Pos: {diag.get('tiny_subset_pos')} | Neg: {diag.get('tiny_subset_neg')}"
        )
        lines.append("")
        lines.append("| fly_id | pos | neg | only_one_class |")
        lines.append("|---|---:|---:|:---:|")
        per_fly = diag.get("per_fly_class_counts", {}) or {}
        for fly_id in diag.get("tiny_subset_flies", []):
            counts = per_fly.get(fly_id, {})
            pos = int(counts.get("pos", 0))
            neg = int(counts.get("neg", 0))
            only_one = "YES" if (pos == 0 or neg == 0) else ""
            lines.append(f"| `{fly_id}` | {pos} | {neg} | {only_one} |")
        lines.append("")

        legacy = diag.get("legacy_tiny_subset")
        if legacy:
            lines.append("## Legacy Tiny Subset (Pre-fix Selection)")
            lines.append("")
            lines.append(
                f"- Legacy selection method: `{legacy.get('selection_method')}`"
            )
            lines.append(f"- Flies: `{', '.join(legacy.get('flies', []))}`")
            lines.append(
                f"- Trials: {legacy.get('trials')} | "
                f"Pos: {legacy.get('pos')} | Neg: {legacy.get('neg')}"
            )
            if legacy.get("flies_with_single_class"):
                lines.append(
                    f"- Flies with only one class: `{', '.join(legacy.get('flies_with_single_class'))}`"
                )
            lines.append("")
            lines.append("| fly_id | pos | neg | only_one_class |")
            lines.append("|---|---:|---:|:---:|")
            legacy_counts = legacy.get("per_fly_class_counts", {}) or {}
            for fly_id in legacy.get("flies", []):
                counts = legacy_counts.get(fly_id, {})
                pos = int(counts.get("pos", 0))
                neg = int(counts.get("neg", 0))
                only_one = "YES" if (pos == 0 or neg == 0) else ""
                lines.append(f"| `{fly_id}` | {pos} | {neg} | {only_one} |")
            lines.append("")

        lines.append("## Overfit Metrics (Best Train-Loss Epoch)")
        lines.append("")
        best = overfit.get("best_metrics", {}) or {}
        thr = diag.get("overfit_thresholds", {}) or {}
        lines.append(
            f"- Loss: {best.get('loss'):.4f} (gate: < {overfit.get('success_criteria', {}).get('loss_eps')})"
        )
        lines.append(f"- AUROC: {best.get('auroc'):.4f}")
        lines.append(f"- AUPRC: {best.get('auprc'):.4f}")
        lines.append(f"- BalAcc@0.5: {best.get('fixed_balanced_acc'):.4f} (fixed_threshold={thr.get('fixed_threshold')})")
        lines.append(
            f"- BalAcc@opt: {best.get('optimized_balanced_acc'):.4f} "
            f"(optimized_threshold={thr.get('optimized_threshold'):.4f}; "
            f"gate: >= {overfit.get('success_criteria', {}).get('optimized_balacc_eps')})"
        )
        lines.append("")

        lines.append("## Overfit Metric Computation")
        lines.append("")
        lines.append("- Loss: `BCEWithLogitsLoss` on training set (no `pos_weight`).")
        lines.append("- Fixed balanced accuracy: threshold `0.5` applied to sigmoid probabilities.")
        lines.append("- Optimized balanced accuracy: threshold chosen to maximize balanced accuracy on the tiny training set.")
        lines.append("")

    lines.append("## Why The Original Overfit Gate Failed")
    lines.append("")
    lines.append(
        "- The old check gated on **balanced accuracy at a fixed 0.5 threshold**. "
        "On tiny, imbalanced subsets, the model can rank positives above negatives (high AUROC) "
        "while still keeping all probabilities below 0.5, yielding BalAcc≈0.5 (all-negative predictions)."
    )
    lines.append(
        "- The previous tiny-subset selection took the **first K flies after a shuffle**, which can include "
        "flies with only one class (e.g., all-negative), amplifying threshold/imbalance issues."
    )
    lines.append("")

    lines.append("## What Changed (Decision → Evidence → Implementation)")
    lines.append("")
    lines.append("- Decision: Make the overfit preflight a memorization sanity check, not a calibration check.")
    lines.append("- Evidence: Fixed-threshold BalAcc can be 0.5 even with high AUROC on imbalanced tiny-N.")
    lines.append("- Implementation:")
    lines.append("  - Added loss-based memorization gate (train_loss < 0.05 by default).")
    lines.append("  - Added optimized-threshold balanced accuracy gate (BalAcc@opt ≥ 0.95 by default).")
    lines.append("  - Made tiny subset fly selection deterministic + class-count constrained (min pos/neg).")
    lines.append("")

    lines.append("## Next Steps")
    lines.append("")
    lines.append("- [ ] Re-run preflight for tiers 0/1/2 with desired settings")
    lines.append("- [ ] If all pass, proceed to full training runs")
    lines.append("- [ ] If overfit fails, inspect diagnostic block for class balance and gate values")
    lines.append("")

    if shuffle:
        lines.append("## Label Shuffle Summary")
        lines.append("")
        fm = shuffle.get("final_metrics", {}) or {}
        lines.append(
            f"- AUROC: {fm.get('auroc'):.4f} | BalAcc@0.5: {fm.get('balanced_acc'):.4f} | "
            f"Passed: {shuffle.get('passed')}"
        )
        lines.append("")

    report_path.write_text("\n".join(lines) + "\n")
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description='Preflight training checks (overfit + label shuffle)'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data/pgcn_features',
        help='Directory containing trial data',
    )
    parser.add_argument(
        '--constraint_tier',
        type=int,
        default=2,
        choices=[0, 1, 2],
        help='Constraint tier for model',
    )
    parser.add_argument(
        '--tiny_overfit',
        action='store_true',
        help='Run overfit test on tiny dataset',
    )
    parser.add_argument(
        '--tiny_overfit_flies',
        type=int,
        default=3,
        help='Number of flies for tiny overfit (may auto-increase to meet min pos/neg)',
    )
    parser.add_argument(
        '--tiny_overfit_min_pos',
        type=int,
        default=5,
        help='Minimum positives required in tiny overfit subset',
    )
    parser.add_argument(
        '--tiny_overfit_min_neg',
        type=int,
        default=5,
        help='Minimum negatives required in tiny overfit subset',
    )
    parser.add_argument(
        '--label_shuffle',
        action='store_true',
        help='Run label shuffle test',
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='outputs/preflight',
        help='Output directory for report',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use',
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate receptor ordering first
    logger.info("=" * 70)
    logger.info("RECEPTOR ORDERING VALIDATION")
    logger.info("=" * 70)

    training_set_path = "data/mappings/training_receptor_set.json"
    connectivity_metadata_path = data_dir / "connectivity" / "connectivity_metadata.json"
    feature_schema_path = data_dir / "feature_metadata.json"

    try:
        report = validate_receptor_ordering(
            training_set_path=training_set_path,
            connectivity_metadata_path=str(connectivity_metadata_path),
            feature_schema_path=str(feature_schema_path),
            strict=False,  # Warn only for now
        )
        print_validation_report(report)

        if not report['passed']:
            logger.error("Receptor ordering validation FAILED")
            logger.error("Fix issues before running preflight checks")
            return

    except Exception as e:
        logger.error(f"Receptor validation error: {e}")
        return

    # Run tests
    results = {
        'receptor_validation': report,
        'tests': [],
    }

    if args.tiny_overfit:
        overfit_result = run_overfit_test(
            data_dir=data_dir,
            constraint_tier=args.constraint_tier,
            n_flies=args.tiny_overfit_flies,
            min_pos=args.tiny_overfit_min_pos,
            min_neg=args.tiny_overfit_min_neg,
            epochs=100,
            device=args.device,
        )
        results['tests'].append(overfit_result)

    if args.label_shuffle:
        shuffle_result = run_label_shuffle_test(
            data_dir=data_dir,
            constraint_tier=args.constraint_tier,
            epochs=10,
            device=args.device,
        )
        results['tests'].append(shuffle_result)

    run_command = " ".join(["python", *sys.argv])

    # Save report
    report_path = output_dir / 'preflight_report.json'
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2, default=_json_default)

    diagnosis_path = _write_preflight_diagnosis_report(
        output_dir=output_dir,
        run_command=run_command,
        results=results,
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info("PREFLIGHT CHECKS COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report saved to: {report_path}")
    logger.info(f"Diagnosis report saved to: {diagnosis_path}")

    all_passed = all(test['passed'] for test in results['tests'])
    if all_passed:
        logger.info("✅ All preflight checks PASSED")
    else:
        logger.error("❌ Some preflight checks FAILED")
        for test in results['tests']:
            if not test['passed']:
                logger.error(f"  - {test['test_name']}: FAILED")


if __name__ == '__main__':
    main()
