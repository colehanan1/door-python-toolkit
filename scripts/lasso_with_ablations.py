#!/usr/bin/env python3
"""
LASSO Ablation Analysis Script
==============================

Refit LASSO behavioral prediction models after ablating (zeroing) selected
receptor feature channels, measuring robustness of the identified receptor
circuits.

This script:
1. Fits a baseline LASSO model (no ablation)
2. For each ablation scenario, zeros out specified receptor columns
3. Refits LASSO using the SAME scaler as baseline (apples-to-apples comparison)
4. Saves detailed artifacts and summary statistics

Example usage:
    # Ablate multiple receptors together
    python scripts/lasso_with_ablations.py \\
        --door_cache door_cache \\
        --behavior_csv reaction_rates.csv \\
        --condition opto_hex \\
        --ablate Or42b,Or47b \\
        --ablation_set_mode all_in_one \\
        --output_dir outputs/ablation/

    # Ablate receptors one at a time
    python scripts/lasso_with_ablations.py \\
        --door_cache door_cache \\
        --behavior_csv reaction_rates.csv \\
        --condition opto_hex \\
        --ablate Or42b,Or47b,Or22a \\
        --ablation_set_mode single \\
        --output_dir outputs/ablation/

    # Ablate receptors from a file
    python scripts/lasso_with_ablations.py \\
        --door_cache door_cache \\
        --behavior_csv reaction_rates.csv \\
        --condition opto_hex \\
        --ablate_file receptors_to_ablate.txt \\
        --output_dir outputs/ablation/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

from door_toolkit.pathways.behavioral_prediction import (
    LassoBehavioralPredictor,
    apply_receptor_ablation,
    fit_lasso_with_fixed_scaler,
    resolve_receptor_names,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Results from a single ablation run."""

    ablation_name: str
    receptors_ablated: List[str]
    ablated_indices: List[int]
    cv_r2: float
    cv_mse: float
    n_receptors_selected: int
    lambda_value: float
    lasso_weights: Dict[str, float]
    delta_r2: float = 0.0
    delta_mse: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LASSO ablation analysis for receptor circuit robustness.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required arguments
    parser.add_argument(
        "--door_cache",
        type=str,
        required=True,
        help="Path to DoOR cache directory",
    )
    parser.add_argument(
        "--behavior_csv",
        type=str,
        required=True,
        help="Path to behavioral CSV (reaction_rates_summary_unordered.csv)",
    )
    parser.add_argument(
        "--condition",
        required=True,
        action="append",
        help="Condition name(s). Repeat flag or pass comma-separated list.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory for output files",
    )

    # Ablation specification (mutually exclusive)
    ablation_group = parser.add_mutually_exclusive_group(required=True)
    ablation_group.add_argument(
        "--ablate",
        type=str,
        help="Comma-separated list of receptor names to ablate",
    )
    ablation_group.add_argument(
        "--ablate_file",
        type=str,
        help="File with receptor names (one per line) to ablate",
    )

    # Ablation mode
    parser.add_argument(
        "--ablation_set_mode",
        type=str,
        choices=["single", "all_in_one"],
        default="all_in_one",
        help=(
            "How to process the ablation set: "
            "'single' = ablate each receptor individually; "
            "'all_in_one' = ablate all receptors together (default)"
        ),
    )

    # Prediction mode
    parser.add_argument(
        "--prediction_mode",
        type=str,
        choices=["test_odorant", "trained_odorant", "interaction"],
        default="test_odorant",
        help="Feature extraction mode (default: test_odorant)",
    )
    parser.add_argument(
        "--subtract_control",
        action="store_true",
        help="Fit on ΔPER (opto - control) instead of raw PER.",
    )
    parser.add_argument(
        "--control_condition",
        type=str,
        default=None,
        help="Optional control dataset override (default: infer from opto condition).",
    )
    parser.add_argument(
        "--missing_control_policy",
        type=str,
        choices=["skip", "zero", "error"],
        default="error",
        help="How to handle missing control values (default: error).",
    )
    parser.add_argument(
        "--debug_stats",
        action="store_true",
        help="Log y stats, chosen lambda, and nonzero coefficient count.",
    )

    # LASSO parameters
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)",
    )
    parser.add_argument(
        "--lambda_range",
        type=str,
        default="0.0001,0.001,0.01,0.1,1.0",
        help="Comma-separated lambda values for LASSO CV (default: 0.0001,0.001,0.01,0.1,1.0)",
    )
    parser.add_argument(
        "--lambda_value",
        type=float,
        default=None,
        help="Fixed lambda value (overrides --lambda_range)",
    )

    # Receptor resolution
    parser.add_argument(
        "--missing_receptor_policy",
        type=str,
        choices=["error", "skip"],
        default="error",
        help="Policy for unresolved receptor names: 'error' or 'skip' (default: error)",
    )

    # Scaling
    parser.add_argument(
        "--scale_features",
        action="store_true",
        default=True,
        help="Standardize receptor features (default: True)",
    )
    parser.add_argument(
        "--no_scale_features",
        action="store_false",
        dest="scale_features",
        help="Do not standardize receptor features",
    )

    return parser.parse_args()


def _parse_conditions(values: List[str]) -> List[str]:
    conditions: List[str] = []
    seen = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token or token in seen:
                continue
            conditions.append(token)
            seen.add(token)
    return conditions


def load_receptors_from_file(filepath: str) -> List[str]:
    """Load receptor names from a file (one per line)."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Receptor file not found: {filepath}")

    receptors = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                receptors.append(line)

    logger.info(f"Loaded {len(receptors)} receptors from {filepath}")
    return receptors


def save_weights_csv(
    weights: Dict[str, float], filepath: Path, condition: str, ablation_name: str
) -> None:
    """Save LASSO weights to CSV."""
    rows = []
    for receptor, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        rows.append({
            "condition": condition,
            "ablation": ablation_name,
            "receptor": receptor,
            "weight": weight,
            "abs_weight": abs(weight),
        })

    df = pd.DataFrame(rows)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)
    logger.info(f"Saved weights to {filepath}")


def _build_valid_odorants(
    predictor: LassoBehavioralPredictor,
    condition: str,
    subtract_control: bool,
    control_condition: Optional[str],
    missing_control_policy: str,
) -> Tuple[pd.Series, str, Optional[str], int]:
    resolved_condition = predictor._resolve_dataset_name(condition)
    if resolved_condition is None:
        raise ValueError(f"Condition '{condition}' not found in behavioral data")

    if not subtract_control:
        per_responses = predictor.behavioral_data.loc[resolved_condition]
        valid_odorants = per_responses.dropna()
        if len(valid_odorants) == 0:
            raise ValueError(f"No valid PER data for condition '{resolved_condition}'")
        return valid_odorants, resolved_condition, None, 0

    if missing_control_policy not in {"skip", "zero", "error"}:
        raise ValueError(
            f"Unknown missing_control_policy '{missing_control_policy}'. "
            "Expected one of: skip, zero, error."
        )

    if control_condition is not None:
        control_resolved = predictor._resolve_dataset_name(control_condition)
        if control_resolved is None:
            raise ValueError(f"Control condition '{control_condition}' not found in behavioral data")
    else:
        control_candidate = predictor._infer_control_condition(resolved_condition)
        if control_candidate is None:
            raise ValueError(
                f"No matched control mapping for '{resolved_condition}'. "
                "Provide --control_condition or disable --subtract_control."
            )
        control_resolved = predictor._resolve_dataset_name(control_candidate)
        if control_resolved is None:
            raise ValueError(f"Matched control '{control_candidate}' not found in behavioral data")

    if control_resolved == resolved_condition:
        raise ValueError(
            f"Control condition '{control_resolved}' matches opto condition '{resolved_condition}'."
        )

    per_opto = predictor.behavioral_data.loc[resolved_condition]
    per_ctrl = predictor.behavioral_data.loc[control_resolved]

    if missing_control_policy == "skip":
        valid_mask = per_opto.notna() & per_ctrl.notna()
        valid_odorants = (per_opto - per_ctrl)[valid_mask]
    elif missing_control_policy == "zero":
        valid_mask = per_opto.notna()
        per_ctrl_filled = per_ctrl.fillna(0.0)
        valid_odorants = (per_opto - per_ctrl_filled)[valid_mask]
    else:
        missing_mask = per_opto.notna() & per_ctrl.isna()
        if missing_mask.any():
            missing_odorants = [str(o) for o in per_opto.index[missing_mask]]
            preview = ", ".join(missing_odorants[:5])
            if len(missing_odorants) > 5:
                preview = f"{preview} (and {len(missing_odorants) - 5} more)"
            raise ValueError(
                f"Control condition '{control_resolved}' has missing values for odorants "
                f"present in '{resolved_condition}': {preview}"
            )
        valid_mask = per_opto.notna() & per_ctrl.notna()
        valid_odorants = (per_opto - per_ctrl)[valid_mask]

    if len(valid_odorants) == 0:
        raise ValueError(
            f"No valid opto/control pairs for '{resolved_condition}' and '{control_resolved}'."
        )

    return valid_odorants, resolved_condition, control_resolved, int(len(valid_odorants))


def _log_debug_stats(
    *,
    condition: str,
    mode: str,
    y: np.ndarray,
    n_pairs_used: int,
    lambda_value: float,
    n_nonzero: int,
) -> None:
    logger.info(
        "[debug] %s %s y_stats: n=%d mean=%.4f std=%.4f min=%.4f max=%.4f n_pairs=%d",
        condition,
        mode,
        len(y),
        float(np.mean(y)),
        float(np.std(y)),
        float(np.min(y)),
        float(np.max(y)),
        n_pairs_used,
    )
    logger.info(
        "[debug] %s %s lambda=%.6f n_nonzero=%d",
        condition,
        mode,
        float(lambda_value),
        int(n_nonzero),
    )


def save_model_json(
    result: AblationResult,
    condition: str,
    prediction_mode: str,
    n_samples: int,
    n_receptors_total: int,
    filepath: Path,
) -> None:
    """Save model metadata to JSON."""
    data = {
        "condition_name": condition,
        "ablation_name": result.ablation_name,
        "receptors_ablated": result.receptors_ablated,
        "ablated_indices": result.ablated_indices,
        "prediction_mode": prediction_mode,
        "n_samples": n_samples,
        "n_receptors_total": n_receptors_total,
        "cv_r2": result.cv_r2,
        "cv_mse": result.cv_mse,
        "delta_r2": result.delta_r2,
        "delta_mse": result.delta_mse,
        "lambda_value": result.lambda_value,
        "n_receptors_selected": result.n_receptors_selected,
        "lasso_weights": result.lasso_weights,
        "top_10_receptors": [
            {"receptor": r, "weight": w}
            for r, w in sorted(
                result.lasso_weights.items(), key=lambda x: abs(x[1]), reverse=True
            )[:10]
        ],
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved model to {filepath}")


def plot_ablation_comparison(
    baseline_result: AblationResult,
    ablation_results: List[AblationResult],
    output_dir: Path,
    condition: str,
) -> None:
    """Generate bar charts comparing baseline vs ablations."""
    if not ablation_results:
        return

    # Prepare data
    names = ["Baseline"] + [r.ablation_name for r in ablation_results]
    r2_values = [baseline_result.cv_r2] + [r.cv_r2 for r in ablation_results]
    mse_values = [baseline_result.cv_mse] + [r.cv_mse for r in ablation_results]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # R² comparison
    ax1 = axes[0]
    colors = ["green"] + ["coral"] * len(ablation_results)
    bars1 = ax1.bar(range(len(names)), r2_values, color=colors, alpha=0.7, edgecolor="k")
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=45, ha="right")
    ax1.set_ylabel("Cross-validated R²", fontsize=12)
    ax1.set_title(f"{condition} - R² Comparison", fontsize=14)
    ax1.axhline(baseline_result.cv_r2, color="green", linestyle="--", alpha=0.5, label="Baseline")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars1, r2_values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # MSE comparison
    ax2 = axes[1]
    bars2 = ax2.bar(range(len(names)), mse_values, color=colors, alpha=0.7, edgecolor="k")
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha="right")
    ax2.set_ylabel("Cross-validated MSE", fontsize=12)
    ax2.set_title(f"{condition} - MSE Comparison", fontsize=14)
    ax2.axhline(baseline_result.cv_mse, color="green", linestyle="--", alpha=0.5, label="Baseline")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    # Add value labels
    for bar, val in zip(bars2, mse_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    # Save
    plot_path = output_dir / "ablation_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved ablation comparison plot to {plot_path}")


def run_baseline(
    predictor: LassoBehavioralPredictor,
    condition: str,
    prediction_mode: str,
    lambda_range: np.ndarray,
    cv_folds: int,
    scale_features: bool,
    subtract_control: bool = False,
    control_condition: Optional[str] = None,
    missing_control_policy: str = "skip",
    debug_stats: bool = False,
) -> Tuple[AblationResult, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Run baseline LASSO fit (no ablation).

    Returns:
        Tuple of (baseline_result, X, y, receptor_names, scaler)
    """
    logger.info(f"Fitting baseline model for condition: {condition}")

    valid_odorants, condition_resolved, control_resolved, n_pairs_used = _build_valid_odorants(
        predictor,
        condition=condition,
        subtract_control=subtract_control,
        control_condition=control_condition,
        missing_control_policy=missing_control_policy,
    )
    if control_resolved:
        logger.info(
            "Using control condition '%s' with policy '%s': %d odorants after alignment",
            control_resolved,
            missing_control_policy,
            n_pairs_used,
        )

    # Extract features based on prediction mode
    if prediction_mode == "test_odorant":
        X, test_odorants, y = predictor._extract_test_odorant_features(valid_odorants)
    elif prediction_mode == "trained_odorant":
        trained_odorant = predictor.CONDITION_ODORANT_MAPPING.get(condition_resolved)
        if not trained_odorant:
            raise ValueError(f"Could not determine trained odorant for {condition_resolved}")
        X, test_odorants, y = predictor._extract_trained_odorant_features(
            trained_odorant, valid_odorants
        )
    elif prediction_mode == "interaction":
        trained_odorant = predictor.CONDITION_ODORANT_MAPPING.get(condition_resolved)
        if not trained_odorant:
            raise ValueError(f"Could not determine trained odorant for {condition_resolved}")
        X, test_odorants, y = predictor._extract_interaction_features(
            trained_odorant, valid_odorants
        )
    else:
        raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

    if X.shape[0] < 3:
        raise ValueError(f"Insufficient data: only {X.shape[0]} samples")

    # Get receptor names
    receptor_names = list(predictor.masked_receptor_names)

    # Fit scaler on baseline X (if scaling enabled)
    if scale_features:
        scaler = StandardScaler()
        scaler.fit(X)
    else:
        scaler = None

    # Fit baseline model
    weights, cv_r2, cv_mse, best_lambda, y_pred = fit_lasso_with_fixed_scaler(
        X=X,
        y=y,
        receptor_names=receptor_names,
        scaler=scaler,
        lambda_range=lambda_range,
        cv_folds=cv_folds,
    )

    baseline_result = AblationResult(
        ablation_name="baseline",
        receptors_ablated=[],
        ablated_indices=[],
        cv_r2=cv_r2,
        cv_mse=cv_mse,
        n_receptors_selected=len(weights),
        lambda_value=best_lambda,
        lasso_weights=weights,
        delta_r2=0.0,
        delta_mse=0.0,
    )

    logger.info(
        f"Baseline: R² = {cv_r2:.4f}, MSE = {cv_mse:.4f}, "
        f"λ = {best_lambda:.6f}, {len(weights)} receptors selected"
    )
    if debug_stats:
        _log_debug_stats(
            condition=condition_resolved,
            mode="baseline",
            y=y,
            n_pairs_used=n_pairs_used,
            lambda_value=best_lambda,
            n_nonzero=len(weights),
        )

    return baseline_result, X, y, receptor_names, scaler


def run_ablation(
    X: np.ndarray,
    y: np.ndarray,
    receptor_names: List[str],
    receptors_to_ablate: List[str],
    scaler: Optional[StandardScaler],
    lambda_range: np.ndarray,
    cv_folds: int,
    baseline_r2: float,
    baseline_mse: float,
    ablation_name: str,
    debug_stats: bool = False,
) -> AblationResult:
    """Run LASSO with specified receptors ablated."""
    logger.info(f"Running ablation: {ablation_name} ({receptors_to_ablate})")

    # Apply ablation
    X_ablated, ablated_indices = apply_receptor_ablation(
        X=X,
        receptor_names=receptor_names,
        receptors_to_ablate=receptors_to_ablate,
    )

    # Fit model with same scaler
    weights, cv_r2, cv_mse, best_lambda, y_pred = fit_lasso_with_fixed_scaler(
        X=X_ablated,
        y=y,
        receptor_names=receptor_names,
        scaler=scaler,
        lambda_range=lambda_range,
        cv_folds=cv_folds,
    )

    result = AblationResult(
        ablation_name=ablation_name,
        receptors_ablated=receptors_to_ablate,
        ablated_indices=ablated_indices,
        cv_r2=cv_r2,
        cv_mse=cv_mse,
        n_receptors_selected=len(weights),
        lambda_value=best_lambda,
        lasso_weights=weights,
        delta_r2=cv_r2 - baseline_r2,
        delta_mse=cv_mse - baseline_mse,
    )

    logger.info(
        f"Ablation '{ablation_name}': R² = {cv_r2:.4f} (Δ = {result.delta_r2:+.4f}), "
        f"MSE = {cv_mse:.4f} (Δ = {result.delta_mse:+.4f})"
    )
    if debug_stats:
        _log_debug_stats(
            condition=ablation_name,
            mode="ablation",
            y=y,
            n_pairs_used=0,
            lambda_value=best_lambda,
            n_nonzero=len(weights),
        )

    return result


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Parse lambda range
    if args.lambda_value is not None:
        lambda_range = np.array([args.lambda_value])
    else:
        lambda_range = np.array([float(x.strip()) for x in args.lambda_range.split(",")])

    logger.info(f"Lambda range: {lambda_range}")

    # Load receptors to ablate
    if args.ablate:
        receptors_to_ablate = [r.strip() for r in args.ablate.split(",") if r.strip()]
    else:
        receptors_to_ablate = load_receptors_from_file(args.ablate_file)

    if not receptors_to_ablate:
        logger.error("No receptors specified for ablation")
        return 1

    logger.info(f"Receptors to ablate: {receptors_to_ablate}")

    # Initialize predictor
    predictor = LassoBehavioralPredictor(
        doorcache_path=args.door_cache,
        behavior_csv_path=args.behavior_csv,
        scale_features=False,  # We handle scaling manually for ablation
        scale_targets=False,
    )

    # Resolve receptor names
    available_receptors = list(predictor.masked_receptor_names)
    strict_mode = args.missing_receptor_policy == "error"

    matched_receptors, unmatched = resolve_receptor_names(
        receptors_to_ablate, available_receptors, strict=strict_mode
    )

    if unmatched:
        if strict_mode:
            # Error already raised by resolve_receptor_names
            pass
        else:
            logger.warning(f"Skipping unmatched receptors: {unmatched}")

    if not matched_receptors:
        logger.error("No valid receptors to ablate after resolution")
        return 1

    logger.info(f"Resolved receptors: {matched_receptors}")

    conditions = _parse_conditions(args.condition)
    if not conditions:
        logger.error("No valid conditions provided.")
        return 1

    exit_code = 0
    for condition in conditions:
        try:
            resolved_condition = predictor._resolve_dataset_name(condition)
        except ValueError as exc:
            logger.error(str(exc))
            exit_code = 1
            continue

        if resolved_condition is None:
            logger.error(
                "Condition '%s' not found. Available: %s",
                condition,
                list(predictor.behavioral_data.index),
            )
            exit_code = 1
            continue

        if (
            args.subtract_control
            and args.missing_control_policy == "skip"
            and args.control_condition is None
        ):
            control_candidate = predictor._infer_control_condition(resolved_condition)
            if control_candidate is None:
                logger.warning(
                    "No matched control mapping for '%s'; skipping ΔPER run "
                    "(missing_control_policy=skip).",
                    resolved_condition,
                )
                continue
            control_resolved = predictor._resolve_dataset_name(control_candidate)
            if control_resolved is None:
                logger.warning(
                    "Matched control '%s' not found for '%s'; skipping ΔPER run "
                    "(missing_control_policy=skip).",
                    control_candidate,
                    resolved_condition,
                )
                continue

        # Create output directory
        output_dir = Path(args.output_dir) / resolved_condition
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run baseline
        try:
            baseline_result, X, y, receptor_names, scaler = run_baseline(
                predictor=predictor,
                condition=resolved_condition,
                prediction_mode=args.prediction_mode,
                lambda_range=lambda_range,
                cv_folds=args.cv_folds,
                scale_features=args.scale_features,
                subtract_control=args.subtract_control,
                control_condition=args.control_condition,
                missing_control_policy=args.missing_control_policy,
                debug_stats=args.debug_stats,
            )
        except Exception as e:
            logger.error("Baseline fit failed for %s: %s", resolved_condition, e)
            exit_code = 1
            continue

        # Save baseline artifacts
        save_model_json(
            result=baseline_result,
            condition=resolved_condition,
            prediction_mode=args.prediction_mode,
            n_samples=X.shape[0],
            n_receptors_total=X.shape[1],
            filepath=output_dir / "baseline_model.json",
        )
        save_weights_csv(
            weights=baseline_result.lasso_weights,
            filepath=output_dir / "baseline_weights.csv",
            condition=resolved_condition,
            ablation_name="baseline",
        )

        # Run ablations
        ablation_results: List[AblationResult] = []

        if args.ablation_set_mode == "single":
            # Ablate each receptor individually
            for receptor in matched_receptors:
                try:
                    ablation_name = f"ablate_{receptor}"
                    result = run_ablation(
                        X=X,
                        y=y,
                        receptor_names=receptor_names,
                        receptors_to_ablate=[receptor],
                        scaler=scaler,
                        lambda_range=lambda_range,
                        cv_folds=args.cv_folds,
                        baseline_r2=baseline_result.cv_r2,
                        baseline_mse=baseline_result.cv_mse,
                        ablation_name=ablation_name,
                        debug_stats=args.debug_stats,
                    )
                    ablation_results.append(result)

                    # Save individual ablation artifacts
                    ablation_dir = output_dir / ablation_name
                    save_model_json(
                        result=result,
                        condition=resolved_condition,
                        prediction_mode=args.prediction_mode,
                        n_samples=X.shape[0],
                        n_receptors_total=X.shape[1],
                        filepath=ablation_dir / "model.json",
                    )
                    save_weights_csv(
                        weights=result.lasso_weights,
                        filepath=ablation_dir / "weights.csv",
                        condition=resolved_condition,
                        ablation_name=ablation_name,
                    )

                except Exception as e:
                    logger.error(
                        "Ablation '%s' failed for %s: %s", receptor, resolved_condition, e
                    )
                    exit_code = 1
                    continue

        else:  # all_in_one
            # Ablate all receptors together
            ablation_name = "ablate_" + "_".join(matched_receptors)
            # Truncate if too long
            if len(ablation_name) > 100:
                ablation_name = f"ablate_{len(matched_receptors)}_receptors"

            try:
                result = run_ablation(
                    X=X,
                    y=y,
                    receptor_names=receptor_names,
                    receptors_to_ablate=matched_receptors,
                    scaler=scaler,
                    lambda_range=lambda_range,
                    cv_folds=args.cv_folds,
                    baseline_r2=baseline_result.cv_r2,
                    baseline_mse=baseline_result.cv_mse,
                    ablation_name=ablation_name,
                    debug_stats=args.debug_stats,
                )
                ablation_results.append(result)

                # Save ablation artifacts
                ablation_dir = output_dir / ablation_name
                save_model_json(
                    result=result,
                    condition=resolved_condition,
                    prediction_mode=args.prediction_mode,
                    n_samples=X.shape[0],
                    n_receptors_total=X.shape[1],
                    filepath=ablation_dir / "model.json",
                )
                save_weights_csv(
                    weights=result.lasso_weights,
                    filepath=ablation_dir / "weights.csv",
                    condition=resolved_condition,
                    ablation_name=ablation_name,
                )

            except Exception as e:
                logger.error("Ablation failed for %s: %s", resolved_condition, e)
                exit_code = 1
                continue

        # Generate summary CSV
        summary_rows = []

        # Baseline row
        summary_rows.append({
            "ablation_name": "baseline",
            "receptors_ablated": "",
            "n_ablated": 0,
            "cv_r2": baseline_result.cv_r2,
            "cv_mse": baseline_result.cv_mse,
            "n_receptors_selected": baseline_result.n_receptors_selected,
            "lambda_value": baseline_result.lambda_value,
            "delta_r2": 0.0,
            "delta_mse": 0.0,
        })

        # Ablation rows
        for result in ablation_results:
            summary_rows.append({
                "ablation_name": result.ablation_name,
                "receptors_ablated": ";".join(result.receptors_ablated),
                "n_ablated": len(result.receptors_ablated),
                "cv_r2": result.cv_r2,
                "cv_mse": result.cv_mse,
                "n_receptors_selected": result.n_receptors_selected,
                "lambda_value": result.lambda_value,
                "delta_r2": result.delta_r2,
                "delta_mse": result.delta_mse,
            })

        summary_df = pd.DataFrame(summary_rows)

        # Create ablations subfolder for summary files
        ablations_dir = output_dir / "ablations"
        ablations_dir.mkdir(parents=True, exist_ok=True)

        summary_path = ablations_dir / "ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        logger.info("Saved summary to %s", summary_path)

        # Generate comparison plot
        plot_ablation_comparison(
            baseline_result=baseline_result,
            ablation_results=ablation_results,
            output_dir=ablations_dir,
            condition=resolved_condition,
        )

        # Print summary
        print("\n" + "=" * 80)
        print(f"LASSO Ablation Analysis Complete: {resolved_condition}")
        print("=" * 80)
        print(
            f"\nBaseline: R² = {baseline_result.cv_r2:.4f}, MSE = {baseline_result.cv_mse:.4f}"
        )
        print(f"          {baseline_result.n_receptors_selected} receptors selected")
        print(f"\nAblation Results:")
        for result in ablation_results:
            print(
                f"  {result.ablation_name:40s}  "
                f"R² = {result.cv_r2:.4f} (Δ = {result.delta_r2:+.4f})  "
                f"MSE = {result.cv_mse:.4f} (Δ = {result.delta_mse:+.4f})"
            )
        print(f"\nOutputs saved to: {output_dir}")
        print("=" * 80)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
