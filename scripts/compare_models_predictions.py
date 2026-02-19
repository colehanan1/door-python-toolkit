#!/usr/bin/env python3
"""
Compare 2 regression models (elasticnet, lasso) with LOOCV + predictions.

For each model:
  1. Run multicond LOOCV
  2. Make predictions from averaged weights
  3. Generate plots

Then combine all plots into a comparison figure.

Usage:
    python scripts/compare_models_predictions.py \
        --csv /tmp/reaction_rates_no_citral.csv \
        --control-row opto_AIR \
        --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct \
        --feature-set intersection \
        --activation-threshold 0.05 \
        --outdir out/model_comparison
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure src/ is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from door_toolkit.encoder import DoOREncoder
from door_toolkit.glomerulus_features import (
    build_design_matrix,
    load_receptor_to_glomerulus_mapping,
)

logger = logging.getLogger(__name__)

# CSV odor -> DoOR name mapping
CSV_ODOR_TO_DOOR = {
    "3-Octonol": "3-octanol",
    "Apple_Cider_Vinegar": "acetic acid",
    "Benzaldehyde": "benzaldehyde",
    "Citral": "citral",
    "Ethyl_Butyrate": "ethyl butyrate",
    "Hexanol": "1-hexanol",
    "Linalool": "linalool",
}


def run_loocv_for_model(
    model: str,
    csv_path: str,
    control_row: str,
    conditions: List[str],
    feature_set: str,
    activation_threshold: float,
    l1_ratio: float,
    outdir: str,
    seed: int = 0,
) -> str:
    """Run multicond_loocv for a specific model. Returns output directory path."""
    outdir_path = Path(outdir) / f"loocv_{model}"
    outdir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/run_multicond_loocv.py",
        "--csv", csv_path,
        "--control-row", control_row,
        "--conditions", ",".join(conditions),
        "--model", model,
        "--feature-set", feature_set,
        "--activation-threshold", str(activation_threshold),
        "--outdir", str(outdir_path),
        "--seed", str(seed),
    ]

    if model == "elasticnet":
        cmd.extend(["--l1-ratio", str(l1_ratio)])

    print(f"\n{'='*80}")
    print(f"Running LOOCV: {model}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=_repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"LOOCV failed for model {model}")

    return str(outdir_path)


def run_predictions_for_model(
    model: str,
    loocv_dir: str,
    csv_path: str,
    control_row: str,
    conditions: List[str],
    feature_set: str,
    activation_threshold: float,
    outdir: str,
    plot_top_n: int = None,
) -> Tuple[str, Dict]:
    """Run predictions for a specific model. Returns output dir and prediction data."""
    outdir_path = Path(outdir) / f"predictions_{model}"
    outdir_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        "python", "scripts/predict_with_avg_weights.py",
        "--loocv-dir", loocv_dir,
        "--csv", csv_path,
        "--control-row", control_row,
        "--conditions", ",".join(conditions),
        "--feature-set", feature_set,
        "--activation-threshold", str(activation_threshold),
        "--outdir", str(outdir_path),
    ]

    if plot_top_n is not None:
        cmd.extend(["--plot-top-n", str(plot_top_n)])

    print(f"\n{'='*80}")
    print(f"Running Predictions: {model}")
    print(f"{'='*80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=_repo_root)
    if result.returncode != 0:
        raise RuntimeError(f"Predictions failed for model {model}")

    # Load prediction CSV
    pred_csv = outdir_path / "predictions.csv"
    predictions_df = pd.read_csv(pred_csv)

    # Load conditions overview
    cond_overview = Path(loocv_dir) / "conditions_overview.csv"
    overview_df = pd.read_csv(cond_overview) if cond_overview.exists() else None

    return str(outdir_path), {
        "predictions": predictions_df,
        "overview": overview_df,
    }


def load_predictions_csv(pred_csv_path: str) -> pd.DataFrame:
    """Load predictions CSV."""
    return pd.read_csv(pred_csv_path)


def compute_overall_r2(predictions_df: pd.DataFrame) -> float:
    """Compute overall R² from predictions DataFrame."""
    y_true = predictions_df["true_centered"].values
    y_pred = predictions_df["predicted_centered"].values
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum(y_true ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def plot_model_comparison(
    out_dir: Path,
    model_data: Dict[str, Dict],
    odor_labels: List[str],
    conditions: List[str],
) -> List[str]:
    """Create combined comparison plots across all 3 models."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    models = list(model_data.keys())
    n_models = len(models)
    colors_models = ["#1f77b4", "#ff7f0e"]  # blue, orange

    # --- Plot 1: Overall R² comparison across models ---
    fig, ax = plt.subplots(figsize=(8, 6))
    r2_values = []
    for model in models:
        r2 = compute_overall_r2(model_data[model]["predictions"])
        r2_values.append(r2)

    bars = ax.bar(models, r2_values, color=colors_models, alpha=0.8, edgecolor="black", linewidth=2)
    ax.set_ylabel("Overall R²", fontsize=12)
    ax.set_title("Model Comparison: Overall Prediction R²", fontsize=14, fontweight="bold")
    ax.set_ylim(0, max(r2_values) * 1.2)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, r2 in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, height,
            f"{r2:.3f}",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.tight_layout()
    p = out_dir / "01_overall_r2_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    # --- Plot 2: Per-condition R² comparison ---
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(conditions))
    width = 0.25

    for idx, model in enumerate(models):
        cond_r2s = []
        for cond in conditions:
            pred_subset = model_data[model]["predictions"]
            pred_subset = pred_subset[pred_subset["condition"] == cond]
            if len(pred_subset) > 0:
                r2 = compute_overall_r2(pred_subset)
                cond_r2s.append(r2)
            else:
                cond_r2s.append(0.0)

        ax.bar(
            x + idx * width, cond_r2s, width,
            label=model, color=colors_models[idx], alpha=0.8
        )

    ax.set_xlabel("Condition", fontsize=12)
    ax.set_ylabel("R²", fontsize=12)
    ax.set_title("Per-Condition R² Across Models", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = out_dir / "02_per_condition_r2.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    # --- Plot 3: Predictions scatter overlay (all models, one plot) ---
    fig, ax = plt.subplots(figsize=(10, 10))

    for idx, model in enumerate(models):
        pred_df = model_data[model]["predictions"]
        y_true = pred_df["true_centered"].values
        y_pred = pred_df["predicted_centered"].values
        ax.scatter(
            y_true, y_pred,
            label=model, s=80, alpha=0.6, color=colors_models[idx], edgecolors="black", linewidth=0.5
        )

    # Diagonal line
    lim = max(
        abs(np.concatenate([model_data[m]["predictions"]["true_centered"].values for m in models])).max(),
        abs(np.concatenate([model_data[m]["predictions"]["predicted_centered"].values for m in models])).max(),
    ) + 0.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=2, alpha=0.5, label="Perfect")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True ΔPER (centered)", fontsize=12)
    ax.set_ylabel("Predicted ΔPER (centered)", fontsize=12)
    ax.set_title("Predictions: All Models Overlay", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    p = out_dir / "03_scatter_all_models.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    # --- Plot 4: Per-model scatter plots (2 subplots) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, model in enumerate(models):
        ax = axes[idx]
        pred_df = model_data[model]["predictions"]
        y_true = pred_df["true_centered"].values
        y_pred = pred_df["predicted_centered"].values
        r2 = compute_overall_r2(pred_df)

        ax.scatter(y_true, y_pred, s=100, alpha=0.6, color=colors_models[idx], edgecolors="black", linewidth=1)

        lim = max(abs(y_true).max(), abs(y_pred).max()) + 0.05
        ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=2, alpha=0.5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True ΔPER (centered)", fontsize=11)
        ax.set_ylabel("Predicted ΔPER (centered)", fontsize=11)
        ax.set_title(f"{model}\nR² = {r2:.3f}", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    p = out_dir / "04_scatter_individual_models.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    return plot_paths


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Compare 2 models (elasticnet, lasso) with LOOCV and predictions."
    )
    p.add_argument("--csv", required=True, help="Path to PER CSV.")
    p.add_argument("--control-row", default="opto_AIR")
    p.add_argument("--conditions", required=True, help="Comma-separated trained conditions.")
    p.add_argument("--outdir", default="out/model_comparison")

    # Feature builder settings
    p.add_argument("--door-cache", default="door_cache")
    p.add_argument(
        "--mapping-csv",
        default="data/mappings/door_to_flywire_mapping.csv",
    )
    p.add_argument(
        "--feature-set",
        choices=["all", "union", "intersection", "no_blanks"],
        default="intersection",
    )
    p.add_argument("--activation-threshold", type=float, default=0.05)
    p.add_argument("--l1-ratio", type=float, default=0.3, help="L1 ratio for elasticnet")
    p.add_argument("--plot-top-n", type=int, default=None, help="Top N receptors in weights plot")
    p.add_argument("--seed", type=int, default=0)

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",")]

    print("\n" + "=" * 80)
    print("MODEL COMPARISON PIPELINE")
    print("=" * 80)
    print(f"CSV: {args.csv}")
    print(f"Conditions: {conditions}")
    print(f"Feature set: {args.feature_set}")
    print(f"Activation threshold: {args.activation_threshold}")
    print(f"Output directory: {outdir}")

    # Load PER data to get odor columns
    df = pd.read_csv(args.csv)
    non_odor = {"dataset", "air"}
    odor_cols = []
    for col in df.columns:
        if col.lower() in non_odor:
            continue
        n_missing = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if len(df) > 0 and n_missing / len(df) > 0.5:
            continue
        odor_cols.append(col)

    print(f"Odor columns: {odor_cols}")

    # Run all available models (only lasso and elasticnet are supported in run_multicond_loocv.py)
    model_data = {}
    models = ["elasticnet", "lasso"]

    for model in models:
        print(f"\n{'#'*80}")
        print(f"# MODEL: {model.upper()}")
        print(f"{'#'*80}")

        # Run LOOCV
        loocv_dir = run_loocv_for_model(
            model=model,
            csv_path=args.csv,
            control_row=args.control_row,
            conditions=conditions,
            feature_set=args.feature_set,
            activation_threshold=args.activation_threshold,
            l1_ratio=args.l1_ratio if model == "elasticnet" else 0.5,
            outdir=str(outdir),
            seed=args.seed,
        )

        # Run predictions
        pred_dir, pred_data = run_predictions_for_model(
            model=model,
            loocv_dir=loocv_dir,
            csv_path=args.csv,
            control_row=args.control_row,
            conditions=conditions,
            feature_set=args.feature_set,
            activation_threshold=args.activation_threshold,
            outdir=str(outdir),
            plot_top_n=args.plot_top_n,
        )

        model_data[model] = pred_data

    # Create comparison plots
    print(f"\n{'='*80}")
    print("CREATING COMPARISON PLOTS")
    print(f"{'='*80}\n")

    comparison_plots = plot_model_comparison(
        outdir / "comparison_plots",
        model_data,
        odor_cols,
        conditions,
    )

    print(f"\n{'='*80}")
    print("COMPARISON PLOTS CREATED:")
    for p in comparison_plots:
        print(f"  {p}")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"LOOCV outputs: {outdir}/loocv_*/")
    print(f"Predictions: {outdir}/predictions_*/")
    print(f"Comparison plots: {outdir}/comparison_plots/")
    print()

    for model in models:
        r2 = compute_overall_r2(model_data[model]["predictions"])
        print(f"  {model:15s}: Overall R² = {r2:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
