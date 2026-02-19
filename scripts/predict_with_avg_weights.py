#!/usr/bin/env python3
"""
Predict ΔPER using averaged LOOCV weights.

Uses the mean weights from a multicond_loocv run to predict delta PER
for each condition and generates comparison plots (predicted vs true).

Usage:
    python scripts/predict_with_avg_weights.py \
        --loocv-dir out/multicond_loocv_no_citral \
        --csv /tmp/reaction_rates_no_citral.csv \
        --control-row opto_AIR \
        --conditions opto_EB,opto_hex,opto_benz_1,opto_3-oct \
        --outdir out/prediction_plots_no_citral
"""

import argparse
import logging
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


def load_avg_weights(loocv_dir: str, condition: str) -> pd.DataFrame:
    """Load mean weights CSV for a condition."""
    # Try different filename patterns
    slug = condition.replace("-", "_")
    for pattern in [
        f"weights_mean_{slug}.csv",
        f"weights_mean_{condition}.csv",
    ]:
        path = Path(loocv_dir) / pattern
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"No weights file found for '{condition}' in {loocv_dir}"
    )


def predict_delta_per(
    X: np.ndarray,
    feature_names: List[str],
    weights_df: pd.DataFrame,
) -> np.ndarray:
    """Predict ΔPER using averaged weights: y_pred = X @ w."""
    weights_dict = dict(zip(weights_df["feature"], weights_df["mean_w"]))
    w = np.array([weights_dict.get(feat, 0.0) for feat in feature_names])
    return X @ w


def get_true_delta_per(
    df: pd.DataFrame,
    odor_cols: List[str],
    control_row: str,
    condition: str,
) -> np.ndarray:
    """Compute true ΔPER = condition_PER - control_PER."""
    control_idx = df[df["dataset"] == control_row].index[0]
    cond_idx = df[df["dataset"] == condition].index[0]

    control_per = np.array([
        df.loc[control_idx, col] if pd.notna(df.loc[control_idx, col]) else 0.0
        for col in odor_cols
    ])
    cond_per = np.array([
        df.loc[cond_idx, col] if pd.notna(df.loc[cond_idx, col]) else 0.0
        for col in odor_cols
    ])
    return cond_per - control_per


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R² between centered vectors."""
    y_t = y_true - np.mean(y_true)
    y_p = y_pred - np.mean(y_pred)
    ss_res = np.sum((y_t - y_p) ** 2)
    ss_tot = np.sum(y_t ** 2)
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def plot_predictions(
    out_dir: Path,
    odor_labels: List[str],
    predictions: Dict[str, dict],
    plot_top_n: int = None,
) -> List[str]:
    """Generate all prediction comparison plots.

    Args:
        plot_top_n: If set, only show top N receptors by absolute weight in weights plot.
                   If None, show all receptors.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    n_conds = len(predictions)
    ncols = min(2, n_conds)
    nrows = (n_conds + ncols - 1) // ncols

    # --- Plot 1: Per-condition bar charts ---
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if n_conds == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    for idx, (cond, data) in enumerate(sorted(predictions.items())):
        ax = axes[idx]
        x = np.arange(len(odor_labels))
        width = 0.35

        y_true_c = data["y_true"] - np.mean(data["y_true"])
        y_pred_c = data["y_pred"] - np.mean(data["y_pred"])

        ax.bar(x - width / 2, y_true_c, width, label="True ΔPER", color="#1f77b4", alpha=0.8)
        ax.bar(x + width / 2, y_pred_c, width, label="Predicted ΔPER", color="#ff7f0e", alpha=0.8)

        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.set_ylabel("ΔPER (centered)")
        ax.set_title("{0}\nR² = {1:.3f}  |  MSE = {2:.6f}".format(
            cond, data["r2"], data["mse"]
        ))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [o.replace("_", " ") for o in odor_labels],
            fontsize=8, rotation=45, ha="right",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    # Hide empty axes
    for i in range(n_conds, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    p = out_dir / "predictions_vs_true.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    # --- Plot 2: Scatter ---
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    all_true, all_pred = [], []

    for (cond, data), color in zip(sorted(predictions.items()), colors):
        y_t = data["y_true"] - np.mean(data["y_true"])
        y_p = data["y_pred"] - np.mean(data["y_pred"])
        all_true.extend(y_t)
        all_pred.extend(y_p)
        ax.scatter(y_t, y_p, label=cond, s=100, alpha=0.7, color=color)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    overall_r2 = compute_r2(all_true, all_pred)

    lim = max(
        abs(all_true.min()), abs(all_true.max()),
        abs(all_pred.min()), abs(all_pred.max()),
    ) + 0.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.5, alpha=0.5, label="Perfect")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True ΔPER (centered)", fontsize=12)
    ax.set_ylabel("Predicted ΔPER (centered)", fontsize=12)
    ax.set_title("Predicted vs True ΔPER\nOverall R² = {0:.3f}".format(overall_r2))
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    p = out_dir / "predictions_scatter.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    # --- Plot 3: Weights comparison ---
    feature_names = predictions[list(predictions.keys())[0]]["feature_names"]

    # Select top N receptors if requested
    if plot_top_n is not None and plot_top_n > 0:
        # Find top N receptors by max absolute weight across all conditions
        max_abs_weights = np.zeros(len(feature_names))
        for cond, data in predictions.items():
            max_abs_weights = np.maximum(max_abs_weights, np.abs(data["weights"]))
        top_indices = np.argsort(max_abs_weights)[-plot_top_n:]
        top_indices = np.sort(top_indices)  # Keep original order

        # Filter data
        display_features = [feature_names[i] for i in top_indices]
        for cond in predictions:
            predictions[cond]["weights"] = predictions[cond]["weights"][top_indices]
    else:
        display_features = feature_names

    fig, ax = plt.subplots(figsize=(max(10, len(display_features) * 0.6), 6))

    x = np.arange(len(display_features))
    width = 0.75 / n_conds

    for idx, (cond, data) in enumerate(sorted(predictions.items())):
        ax.bar(
            x + idx * width, data["weights"], width,
            label=cond, color=colors[idx % len(colors)], alpha=0.8,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Receptor")
    ax.set_ylabel("Mean Weight")
    ax.set_title("Averaged Weights ({0} receptors)".format(len(display_features)))
    ax.set_xticks(x + width * (n_conds - 1) / 2)
    ax.set_xticklabels(display_features, rotation=45, ha="right", fontsize=10)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p = out_dir / "weights_comparison.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    return plot_paths


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Predict ΔPER using averaged LOOCV weights."
    )
    p.add_argument(
        "--loocv-dir", required=True,
        help="Directory containing multicond_loocv output (weights_mean_*.csv).",
    )
    p.add_argument(
        "--csv", required=True,
        help="Path to PER CSV.",
    )
    p.add_argument(
        "--control-row", default="opto_AIR",
    )
    p.add_argument(
        "--conditions", required=True,
        help="Comma-separated trained conditions.",
    )
    p.add_argument("--outdir", default="out/prediction_plots")
    p.add_argument(
        "--plot-top-n", type=int, default=None,
        help="Show only top N receptors in weights comparison plot (by max absolute weight). "
             "If None, show all.",
    )

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

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    conditions = [c.strip() for c in args.conditions.split(",")]

    # Load PER data
    df = pd.read_csv(args.csv)

    # Detect odor columns (exclude dataset, AIR, etc.)
    non_odor = {"dataset", "air"}
    odor_cols = []
    for col in df.columns:
        if col.lower() in non_odor:
            continue
        n_missing = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if len(df) > 0 and n_missing / len(df) > 0.5:
            continue
        odor_cols.append(col)

    print("Odor columns: {0}".format(odor_cols))

    # Build feature matrix
    door_odors = [CSV_ODOR_TO_DOOR.get(o, o.lower()) for o in odor_cols]
    print("DoOR odors: {0}".format(door_odors))

    encoder = DoOREncoder(cache_path=args.door_cache, use_torch=False)
    mapping, _ = load_receptor_to_glomerulus_mapping(args.mapping_csv)

    X, feature_names, meta = build_design_matrix(
        door_odors, encoder, mapping,
        feature_set=args.feature_set,
        activation_threshold=args.activation_threshold,
    )
    print("Feature matrix: {0} (odors × receptors)".format(X.shape))
    print("Receptors: {0}".format(feature_names))

    # Predict for each condition
    predictions = {}
    for cond in conditions:
        try:
            weights_df = load_avg_weights(args.loocv_dir, cond)
        except FileNotFoundError as e:
            print("WARNING: {0}".format(e))
            continue

        weights_dict = dict(zip(weights_df["feature"], weights_df["mean_w"]))
        w = np.array([weights_dict.get(feat, 0.0) for feat in feature_names])

        y_pred = X @ w
        y_true = get_true_delta_per(df, odor_cols, args.control_row, cond)
        r2 = compute_r2(y_true, y_pred)
        mse = np.mean((
            (y_true - np.mean(y_true)) - (y_pred - np.mean(y_pred))
        ) ** 2)

        print("\n--- {0} ---".format(cond))
        print("  Non-zero weights: {0}/{1}".format(np.sum(w != 0), len(w)))
        print("  Predicted ΔPER: {0}".format(y_pred))
        print("  True ΔPER:      {0}".format(y_true))
        print("  R²: {0:.4f}".format(r2))
        print("  MSE: {0:.6f}".format(mse))

        predictions[cond] = {
            "weights": w,
            "feature_names": feature_names,
            "y_pred": y_pred,
            "y_true": y_true,
            "r2": r2,
            "mse": mse,
        }

    if not predictions:
        print("No predictions generated!")
        return 1

    # Generate plots
    out_dir = Path(args.outdir)
    plots = plot_predictions(out_dir, odor_cols, predictions, plot_top_n=args.plot_top_n)

    print("\n" + "=" * 70)
    print("PLOTS WRITTEN:")
    for p in plots:
        print("  {0}".format(p))

    # Save predictions CSV
    rows = []
    for cond, data in sorted(predictions.items()):
        for i, odor in enumerate(odor_cols):
            rows.append({
                "condition": cond,
                "odor": odor,
                "true_delta_per": data["y_true"][i],
                "predicted_delta_per": data["y_pred"][i],
                "true_centered": data["y_true"][i] - np.mean(data["y_true"]),
                "predicted_centered": data["y_pred"][i] - np.mean(data["y_pred"]),
            })
    pred_csv = out_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(pred_csv, index=False)
    print("  {0}".format(pred_csv))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
