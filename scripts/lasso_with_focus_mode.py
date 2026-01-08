#!/usr/bin/env python3
"""
LASSO Focus Mode Analysis Script
=================================

Refit LASSO behavioral prediction models using only a restricted set of
receptors (top-N from baseline by absolute weight), generating MSE vs N
curves to assess how many receptors are needed for accurate prediction.

This script:
1. Fits a baseline LASSO model (full receptor set)
2. Ranks receptors by absolute weight
3. For each N in topn_list, restricts to top-N receptors and refits
4. Generates focus_curve.csv and focus_curve.png

Example usage:
    # Sweep through different receptor counts
    python scripts/lasso_with_focus_mode.py \\
        --door_cache door_cache \\
        --behavior_csv reaction_rates.csv \\
        --condition opto_hex \\
        --topn_list 1,2,3,5,10,15,20 \\
        --output_dir outputs/focus/

    # Focus on a specific receptor set
    python scripts/lasso_with_focus_mode.py \\
        --door_cache door_cache \\
        --behavior_csv reaction_rates.csv \\
        --condition opto_hex \\
        --focus_receptors Or42b,Or47b,Or22a \\
        --output_dir outputs/focus/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
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
    fit_lasso_with_fixed_scaler,
    restrict_to_receptors,
    get_top_receptors_by_weight,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class FocusResult:
    """Results from a single focus mode run."""

    n_receptors: int
    receptors_used: List[str]
    cv_r2: float
    cv_mse: float
    lambda_value: float
    n_receptors_selected: int
    lasso_weights: Dict[str, float]
    delta_r2: float = 0.0
    delta_mse: float = 0.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="LASSO focus mode analysis for receptor circuit sufficiency.",
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

    # Focus mode specification
    parser.add_argument(
        "--topn_list",
        type=str,
        default="1,2,3,5,10,15,20,30",
        help="Comma-separated list of N values for top-N focus (default: 1,2,3,5,10,15,20,30)",
    )
    parser.add_argument(
        "--focus_receptors",
        type=str,
        default=None,
        help="Comma-separated list of receptors to focus on (overrides --topn_list)",
    )
    parser.add_argument(
        "--baseline_select_by",
        type=str,
        choices=["abs_weight", "stability"],
        default="abs_weight",
        help="How to rank receptors for top-N selection (default: abs_weight)",
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


def save_baseline_json(
    baseline_weights: Dict[str, float],
    baseline_r2: float,
    baseline_mse: float,
    baseline_lambda: float,
    condition: str,
    prediction_mode: str,
    n_samples: int,
    n_receptors_total: int,
    filepath: Path,
) -> None:
    """Save baseline model metadata to JSON."""
    data = {
        "condition_name": condition,
        "prediction_mode": prediction_mode,
        "n_samples": n_samples,
        "n_receptors_total": n_receptors_total,
        "cv_r2": baseline_r2,
        "cv_mse": baseline_mse,
        "lambda_value": baseline_lambda,
        "n_receptors_selected": len(baseline_weights),
        "lasso_weights": baseline_weights,
        "receptor_ranking": [
            {"receptor": r, "weight": w, "abs_weight": abs(w)}
            for r, w in sorted(
                baseline_weights.items(), key=lambda x: abs(x[1]), reverse=True
            )
        ],
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved baseline model to {filepath}")


def save_focus_json(
    result: FocusResult,
    condition: str,
    prediction_mode: str,
    n_samples: int,
    filepath: Path,
) -> None:
    """Save focus mode result to JSON."""
    data = {
        "condition_name": condition,
        "prediction_mode": prediction_mode,
        "n_samples": n_samples,
        "n_receptors_focused": result.n_receptors,
        "receptors_used": result.receptors_used,
        "cv_r2": result.cv_r2,
        "cv_mse": result.cv_mse,
        "delta_r2": result.delta_r2,
        "delta_mse": result.delta_mse,
        "lambda_value": result.lambda_value,
        "n_receptors_selected": result.n_receptors_selected,
        "lasso_weights": result.lasso_weights,
    }

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved focus result to {filepath}")


def plot_focus_curve(
    focus_results: List[FocusResult],
    baseline_r2: float,
    baseline_mse: float,
    output_dir: Path,
    condition: str,
) -> None:
    """Generate focus curve plots (MSE vs N and R² vs N)."""
    if not focus_results:
        return

    # Sort by n_receptors
    results_sorted = sorted(focus_results, key=lambda r: r.n_receptors)

    n_values = [r.n_receptors for r in results_sorted]
    mse_values = [r.cv_mse for r in results_sorted]
    r2_values = [r.cv_r2 for r in results_sorted]

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # MSE vs N
    ax1 = axes[0]
    ax1.plot(n_values, mse_values, "o-", color="coral", linewidth=2, markersize=8)
    ax1.axhline(
        baseline_mse,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Baseline MSE ({baseline_mse:.4f})",
    )
    ax1.set_xlabel("Number of Receptors (N)", fontsize=12)
    ax1.set_ylabel("Cross-validated MSE", fontsize=12)
    ax1.set_title(f"{condition} - MSE vs Number of Receptors", fontsize=14)
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Add value labels
    for n, mse in zip(n_values, mse_values):
        ax1.annotate(
            f"{mse:.4f}",
            (n, mse),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    # R² vs N
    ax2 = axes[1]
    ax2.plot(n_values, r2_values, "o-", color="steelblue", linewidth=2, markersize=8)
    ax2.axhline(
        baseline_r2,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Baseline R² ({baseline_r2:.4f})",
    )
    ax2.set_xlabel("Number of Receptors (N)", fontsize=12)
    ax2.set_ylabel("Cross-validated R²", fontsize=12)
    ax2.set_title(f"{condition} - R² vs Number of Receptors", fontsize=14)
    ax2.legend()
    ax2.grid(alpha=0.3)

    # Add value labels
    for n, r2 in zip(n_values, r2_values):
        ax2.annotate(
            f"{r2:.4f}",
            (n, r2),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()

    # Save
    plot_path = output_dir / "focus_curve.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved focus curve plot to {plot_path}")


def run_baseline(
    predictor: LassoBehavioralPredictor,
    condition: str,
    prediction_mode: str,
    lambda_range: np.ndarray,
    cv_folds: int,
    scale_features: bool,
) -> Tuple[Dict[str, float], float, float, float, np.ndarray, np.ndarray, List[str], Optional[StandardScaler]]:
    """Run baseline LASSO fit (full receptor set).

    Returns:
        Tuple of (baseline_weights, cv_r2, cv_mse, best_lambda, X, y, receptor_names, scaler)
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

    logger.info(
        f"Baseline: R² = {cv_r2:.4f}, MSE = {cv_mse:.4f}, "
        f"λ = {best_lambda:.6f}, {len(weights)} receptors selected"
    )

    return weights, cv_r2, cv_mse, best_lambda, X, y, receptor_names, scaler


def run_focus(
    X: np.ndarray,
    y: np.ndarray,
    receptor_names: List[str],
    receptors_to_keep: List[str],
    lambda_range: np.ndarray,
    cv_folds: int,
    scale_features: bool,
    baseline_r2: float,
    baseline_mse: float,
) -> FocusResult:
    """Run LASSO with restricted receptor set."""
    n_receptors = len(receptors_to_keep)
    logger.info(f"Running focus mode with N={n_receptors} receptors: {receptors_to_keep}")

    # Restrict X to specified receptors
    X_restricted, kept_names, kept_indices = restrict_to_receptors(
        X=X,
        receptor_names=receptor_names,
        receptors_to_keep=receptors_to_keep,
    )

    # Fit scaler on restricted X (for fair comparison within this N)
    if scale_features:
        scaler = StandardScaler()
        scaler.fit(X_restricted)
    else:
        scaler = None

    # Fit model
    weights, cv_r2, cv_mse, best_lambda, y_pred = fit_lasso_with_fixed_scaler(
        X=X_restricted,
        y=y,
        receptor_names=kept_names,
        scaler=scaler,
        lambda_range=lambda_range,
        cv_folds=cv_folds,
    )

    result = FocusResult(
        n_receptors=n_receptors,
        receptors_used=kept_names,
        cv_r2=cv_r2,
        cv_mse=cv_mse,
        lambda_value=best_lambda,
        n_receptors_selected=len(weights),
        lasso_weights=weights,
        delta_r2=cv_r2 - baseline_r2,
        delta_mse=cv_mse - baseline_mse,
    )

    logger.info(
        f"Focus N={n_receptors}: R² = {cv_r2:.4f} (Δ = {result.delta_r2:+.4f}), "
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

    # Initialize predictor
    predictor = LassoBehavioralPredictor(
        doorcache_path=args.door_cache,
        behavior_csv_path=args.behavior_csv,
        scale_features=False,  # We handle scaling manually
        scale_targets=False,
    )

    # Validate condition
    if args.condition not in predictor.behavioral_data.index:
        logger.error(
            f"Condition '{args.condition}' not found. "
            f"Available: {list(predictor.behavioral_data.index)}"
        )
        return 1

    # Create output directory
    output_dir = Path(args.output_dir) / args.condition
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run baseline
    try:
        (
            baseline_weights,
            baseline_r2,
            baseline_mse,
            baseline_lambda,
            X,
            y,
            receptor_names,
            scaler,
        ) = run_baseline(
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

    # Save baseline
    save_baseline_json(
        baseline_weights=baseline_weights,
        baseline_r2=baseline_r2,
        baseline_mse=baseline_mse,
        baseline_lambda=baseline_lambda,
        condition=args.condition,
        prediction_mode=args.prediction_mode,
        n_samples=X.shape[0],
        n_receptors_total=X.shape[1],
        filepath=output_dir / "baseline_model.json",
    )

    # Determine focus runs
    focus_results: List[FocusResult] = []

    if args.focus_receptors:
        # Use explicit receptor list
        focus_receptors = [r.strip() for r in args.focus_receptors.split(",") if r.strip()]
        logger.info(f"Using explicit focus receptors: {focus_receptors}")

        try:
            result = run_focus(
                X=X,
                y=y,
                receptor_names=receptor_names,
                receptors_to_keep=focus_receptors,
                lambda_range=lambda_range,
                cv_folds=args.cv_folds,
                scale_features=args.scale_features,
                baseline_r2=baseline_r2,
                baseline_mse=baseline_mse,
            )
            focus_results.append(result)

            # Save individual focus result
            focus_dir = output_dir / f"focus_n{result.n_receptors}"
            save_focus_json(
                result=result,
                condition=args.condition,
                prediction_mode=args.prediction_mode,
                n_samples=X.shape[0],
                filepath=focus_dir / "model.json",
            )

        except Exception as e:
            logger.error(f"Focus run failed: {e}")

    else:
        # Use topn_list
        topn_list = [int(x.strip()) for x in args.topn_list.split(",")]
        logger.info(f"Top-N values to test: {topn_list}")

        # Get ranked receptors from baseline
        if not baseline_weights:
            logger.error("Baseline model has no non-zero weights, cannot determine top-N")
            return 1

        if args.baseline_select_by == "abs_weight":
            ranked_receptors = get_top_receptors_by_weight(baseline_weights, len(baseline_weights))
        else:
            # stability mode would require multiple runs - fallback to abs_weight
            logger.warning("stability mode not yet implemented, using abs_weight")
            ranked_receptors = get_top_receptors_by_weight(baseline_weights, len(baseline_weights))

        logger.info(f"Receptor ranking (top 10): {ranked_receptors[:10]}")

        for n in topn_list:
            if n > len(ranked_receptors):
                logger.warning(
                    f"N={n} exceeds available ranked receptors ({len(ranked_receptors)}), skipping"
                )
                continue

            if n <= 0:
                logger.warning(f"N={n} is invalid, skipping")
                continue

            top_n_receptors = ranked_receptors[:n]

            try:
                result = run_focus(
                    X=X,
                    y=y,
                    receptor_names=receptor_names,
                    receptors_to_keep=top_n_receptors,
                    lambda_range=lambda_range,
                    cv_folds=args.cv_folds,
                    scale_features=args.scale_features,
                    baseline_r2=baseline_r2,
                    baseline_mse=baseline_mse,
                )
                focus_results.append(result)

                # Save individual focus result
                focus_dir = output_dir / f"focus_n{n}"
                save_focus_json(
                    result=result,
                    condition=args.condition,
                    prediction_mode=args.prediction_mode,
                    n_samples=X.shape[0],
                    filepath=focus_dir / "model.json",
                )

            except Exception as e:
                logger.error(f"Focus N={n} failed: {e}")
                continue

    # Generate focus_curve.csv
    curve_rows = []

    # Add baseline row
    curve_rows.append({
        "n_receptors": X.shape[1],
        "receptors_used": ";".join(sorted(baseline_weights.keys())),
        "cv_r2": baseline_r2,
        "cv_mse": baseline_mse,
        "lambda_value": baseline_lambda,
        "n_receptors_selected": len(baseline_weights),
        "delta_r2": 0.0,
        "delta_mse": 0.0,
        "is_baseline": True,
    })

    # Add focus rows
    for result in focus_results:
        curve_rows.append({
            "n_receptors": result.n_receptors,
            "receptors_used": ";".join(result.receptors_used),
            "cv_r2": result.cv_r2,
            "cv_mse": result.cv_mse,
            "lambda_value": result.lambda_value,
            "n_receptors_selected": result.n_receptors_selected,
            "delta_r2": result.delta_r2,
            "delta_mse": result.delta_mse,
            "is_baseline": False,
        })

    curve_df = pd.DataFrame(curve_rows)
    curve_df = curve_df.sort_values("n_receptors")
    curve_path = output_dir / "focus_curve.csv"
    curve_df.to_csv(curve_path, index=False)
    logger.info(f"Saved focus curve to {curve_path}")

    # Generate plots
    plot_focus_curve(
        focus_results=focus_results,
        baseline_r2=baseline_r2,
        baseline_mse=baseline_mse,
        output_dir=output_dir,
        condition=args.condition,
    )

    # Print summary
    print("\n" + "=" * 80)
    print(f"LASSO Focus Mode Analysis Complete: {args.condition}")
    print("=" * 80)
    print(f"\nBaseline (full): R² = {baseline_r2:.4f}, MSE = {baseline_mse:.4f}")
    print(f"                 {len(baseline_weights)} receptors selected out of {X.shape[1]}")
    print(f"\nFocus Mode Results:")
    for result in sorted(focus_results, key=lambda r: r.n_receptors):
        print(
            f"  N={result.n_receptors:3d}:  "
            f"R² = {result.cv_r2:.4f} (Δ = {result.delta_r2:+.4f})  "
            f"MSE = {result.cv_mse:.4f} (Δ = {result.delta_mse:+.4f})"
        )
    print(f"\nOutputs saved to: {output_dir}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
