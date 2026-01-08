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
        type=str,
        required=True,
        help="Optogenetic condition name (e.g., opto_hex)",
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
) -> Tuple[AblationResult, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Run baseline LASSO fit (no ablation).

    Returns:
        Tuple of (baseline_result, X, y, receptor_names, scaler)
    """
    logger.info(f"Fitting baseline model for condition: {condition}")

    # Get behavioral responses
    per_responses = predictor.behavioral_data.loc[condition]
    valid_odorants = per_responses.dropna()

    if len(valid_odorants) == 0:
        raise ValueError(f"No valid PER data for condition '{condition}'")

    # Extract features based on prediction mode
    if prediction_mode == "test_odorant":
        X, test_odorants, y = predictor._extract_test_odorant_features(valid_odorants)
    elif prediction_mode == "trained_odorant":
        trained_odorant = predictor.CONDITION_ODORANT_MAPPING.get(condition)
        if not trained_odorant:
            raise ValueError(f"Could not determine trained odorant for {condition}")
        X, test_odorants, y = predictor._extract_trained_odorant_features(
            trained_odorant, valid_odorants
        )
    elif prediction_mode == "interaction":
        trained_odorant = predictor.CONDITION_ODORANT_MAPPING.get(condition)
        if not trained_odorant:
            raise ValueError(f"Could not determine trained odorant for {condition}")
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

    # Validate condition
    if args.condition not in predictor.behavioral_data.index:
        logger.error(
            f"Condition '{args.condition}' not found. "
            f"Available: {list(predictor.behavioral_data.index)}"
        )
        return 1

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

    # Create output directory
    output_dir = Path(args.output_dir) / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run baseline
    try:
        baseline_result, X, y, receptor_names, scaler = run_baseline(
            predictor=predictor,
            condition=args.condition,
            prediction_mode=args.prediction_mode,
            lambda_range=lambda_range,
            cv_folds=args.cv_folds,
            scale_features=args.scale_features,
        )
    except Exception as e:
        logger.error(f"Baseline fit failed: {e}")
        return 1

    # Save baseline artifacts
    save_model_json(
        result=baseline_result,
        condition=args.condition,
        prediction_mode=args.prediction_mode,
        n_samples=X.shape[0],
        n_receptors_total=X.shape[1],
        filepath=output_dir / "baseline_model.json",
    )
    save_weights_csv(
        weights=baseline_result.lasso_weights,
        filepath=output_dir / "baseline_weights.csv",
        condition=args.condition,
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
                )
                ablation_results.append(result)

                # Save individual ablation artifacts
                ablation_dir = output_dir / ablation_name
                save_model_json(
                    result=result,
                    condition=args.condition,
                    prediction_mode=args.prediction_mode,
                    n_samples=X.shape[0],
                    n_receptors_total=X.shape[1],
                    filepath=ablation_dir / "model.json",
                )
                save_weights_csv(
                    weights=result.lasso_weights,
                    filepath=ablation_dir / "weights.csv",
                    condition=args.condition,
                    ablation_name=ablation_name,
                )

            except Exception as e:
                logger.error(f"Ablation '{receptor}' failed: {e}")
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
            )
            ablation_results.append(result)

            # Save ablation artifacts
            ablation_dir = output_dir / ablation_name
            save_model_json(
                result=result,
                condition=args.condition,
                prediction_mode=args.prediction_mode,
                n_samples=X.shape[0],
                n_receptors_total=X.shape[1],
                filepath=ablation_dir / "model.json",
            )
            save_weights_csv(
                weights=result.lasso_weights,
                filepath=ablation_dir / "weights.csv",
                condition=args.condition,
                ablation_name=ablation_name,
            )

        except Exception as e:
            logger.error(f"Ablation failed: {e}")

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
    summary_path = output_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved summary to {summary_path}")

    # Generate comparison plot
    plot_ablation_comparison(
        baseline_result=baseline_result,
        ablation_results=ablation_results,
        output_dir=output_dir,
        condition=args.condition,
    )

    # Print summary
    print("\n" + "=" * 80)
    print(f"LASSO Ablation Analysis Complete: {args.condition}")
    print("=" * 80)
    print(f"\nBaseline: R² = {baseline_result.cv_r2:.4f}, MSE = {baseline_result.cv_mse:.4f}")
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

    return 0


if __name__ == "__main__":
    sys.exit(main())
