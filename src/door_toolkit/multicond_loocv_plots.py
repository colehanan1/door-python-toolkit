"""
Plotting for multi-condition LOOCV results.

Generate comparison plots: baseline vs. delta weights with ΔPER overlays.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def plot_weights_and_deltaperby_odor(
    outdir: str,
    *,
    odors: List[str],
    feature_names: List[str],
    condition_data: Dict[str, dict],
    baseline_weights: Optional[pd.DataFrame] = None,
    top_n: int = 10,
    control_row: str = "opto_AIR",
) -> List[str]:
    """Plot baseline vs. delta weights + ΔPER per odor.

    For each odor:
    - Top subplot: Baseline (if provided) + delta weights for each trained condition
    - Bottom subplot: Mean-centered ΔPER for each trained condition

    Args:
        outdir: Output directory for plots.
        odors: List of odor names (column order).
        feature_names: List of feature (receptor) names.
        condition_data: Dict[condition] -> {y_raw, y_centered, centering_mean, loocv_result}.
        baseline_weights: Optional DataFrame with columns [feature, baseline_w].
        top_n: Number of top features to show per plot.
        control_row: Control condition name.

    Returns:
        List of plot file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Collect delta weights for all trained conditions
    trained_conditions = [
        c for c in condition_data.keys() if c != control_row
    ]
    if not trained_conditions:
        logger.warning("No trained conditions found (only control).")
        return []

    # Build weight matrix: (n_features, n_conditions)
    delta_weights = {}
    for cond in trained_conditions:
        loocv = condition_data[cond]["loocv_result"]
        # Average across folds
        mean_w = np.nanmean(loocv["fold_weights"], axis=0)
        delta_weights[cond] = mean_w

    delta_weight_df = pd.DataFrame(delta_weights, index=feature_names)

    # If baseline provided, align features
    if baseline_weights is not None:
        baseline_weights = baseline_weights.copy()
        if "feature" in baseline_weights.columns:
            baseline_weights = baseline_weights.set_index("feature")
        elif "receptor" in baseline_weights.columns:
            baseline_weights = baseline_weights.set_index("receptor")

        if "baseline_w" not in baseline_weights.columns and "weight" in baseline_weights.columns:
            baseline_weights = baseline_weights.rename(columns={"weight": "baseline_w"})
        if "baseline_w" not in baseline_weights.columns:
            raise ValueError("baseline_weights must include 'baseline_w' (or 'weight') column.")

        # Reindex to match; fill missing with 0 and warn if mismatch.
        missing = [f for f in feature_names if f not in baseline_weights.index]
        extra = [f for f in baseline_weights.index if f not in feature_names]
        if missing:
            logger.warning(
                "Baseline weights missing %d features; filling with 0. Example: %s",
                len(missing),
                missing[:5],
            )
        if extra:
            logger.warning(
                "Baseline weights have %d extra features not in current feature set. Example: %s",
                len(extra),
                extra[:5],
            )
        baseline_weights = baseline_weights.reindex(feature_names).fillna(0.0)
    else:
        baseline_weights = None

    plot_paths = []

    # Plot for each odor
    for odor_idx, odor in enumerate(odors):
        # Get ΔPER (centered) for each trained condition at this odor
        deltapers = {}
        for cond in trained_conditions:
            y_centered = condition_data[cond]["y_centered"]
            deltapers[cond] = y_centered[odor_idx]

        # Find top features by max abs delta weight across conditions
        max_abs_delta = np.abs(delta_weight_df).max(axis=1)
        top_idx = np.argsort(max_abs_delta.values)[::-1][:top_n]
        top_features = delta_weight_df.index[top_idx].tolist()

        # Extract weights and ΔPER for top features
        delta_w_subset = delta_weight_df.loc[top_features, trained_conditions]
        deltapers_subset = {
            cond: deltapers[cond] for cond in trained_conditions
        }

        if baseline_weights is not None:
            baseline_w_subset = baseline_weights.loc[top_features, "baseline_w"]
        else:
            baseline_w_subset = None

        # Create figure with 2 subplots
        fig, axes = plt.subplots(
            nrows=2,
            figsize=(max(10, len(top_features) * 0.7), 8),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_w = axes[0]
        ax_p = axes[1]

        x = np.arange(len(top_features))
        n_series = (1 if baseline_w_subset is not None else 0) + len(
            trained_conditions
        )
        width = 0.75 / max(n_series, 1)

        # Plot baseline weights (if available)
        offset_idx = 0
        if baseline_w_subset is not None:
            bars = ax_w.bar(
                x - (n_series - 1) * width / 2 + offset_idx * width,
                baseline_w_subset.values,
                width,
                label="baseline",
                color="#6A5ACD",
                alpha=0.8,
            )
            for rect in bars:
                height = rect.get_height()
                ax_w.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    "{0:.3f}".format(height),
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                    rotation=90,
                )
            offset_idx += 1

        # Plot delta weights for each trained condition
        colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"
        ]
        for i, cond in enumerate(trained_conditions):
            col_idx = (
                i if baseline_w_subset is None else i + 1
            )
            bars = ax_w.bar(
                x - (n_series - 1) * width / 2 + col_idx * width,
                delta_w_subset[cond].values,
                width,
                label=cond,
                color=colors[i % len(colors)],
                alpha=0.8,
            )
            for rect in bars:
                height = rect.get_height()
                ax_w.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    "{0:.3f}".format(height),
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                    rotation=90,
                )

        ax_w.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax_w.set_xticks(x)
        ax_w.set_xticklabels(top_features, rotation=45, ha="right", fontsize=9)
        ax_w.set_ylabel("Weight")
        ax_w.set_title(
            "Top {0} features: Baseline vs. Δweights ({1})".format(
                len(top_features), odor
            )
        )
        ax_w.legend(fontsize=8, loc="upper right")
        ax_w.grid(axis="y", alpha=0.3)

        # Plot mean-centered ΔPER for each trained condition
        per_vals = [deltapers_subset[cond] for cond in trained_conditions]
        bars = ax_p.bar(
            np.arange(len(trained_conditions)),
            per_vals,
            color=colors[: len(trained_conditions)],
            alpha=0.8,
        )
        ax_p.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax_p.set_xticks(np.arange(len(trained_conditions)))
        ax_p.set_xticklabels(trained_conditions, rotation=45, ha="right", fontsize=9)
        ax_p.set_ylabel("ΔPER (centered)")
        ax_p.set_title("Mean-centered ΔPER for {0}".format(odor))
        for i, (rect, val) in enumerate(zip(bars, per_vals)):
            ax_p.text(
                i,
                val,
                "{0:.4f}".format(val),
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=8,
            )
        ax_p.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        plot_path = out_path / "weights_deltaper_{0}.png".format(
            odor.lower().replace(" ", "_").replace("/", "_")
        )
        fig.savefig(plot_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(plot_path))
        logger.info("Saved plot: %s", plot_path)

    return plot_paths


def plot_condition_comparison(
    outdir: str,
    *,
    conditions: List[str],
    feature_names: List[str],
    condition_data: Dict[str, dict],
    top_n: int = 10,
    control_row: str = "opto_AIR",
) -> List[str]:
    """Plot top features' mean weights across all conditions.

    Returns:
        List of plot file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Build weight matrix across all conditions
    all_weights = {}
    for cond in conditions:
        loocv = condition_data[cond]["loocv_result"]
        mean_w = np.nanmean(loocv["fold_weights"], axis=0)
        all_weights[cond] = mean_w

    weight_df = pd.DataFrame(all_weights, index=feature_names)

    # Find top features by max abs weight
    max_abs = np.abs(weight_df).max(axis=1)
    top_idx = np.argsort(max_abs.values)[::-1][:top_n]
    top_features = weight_df.index[top_idx].tolist()

    fig, ax = plt.subplots(
        figsize=(max(12, len(top_features) * 0.8), 6)
    )

    x = np.arange(len(top_features))
    width = 0.75 / len(conditions)

    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f",
    ]

    for i, cond in enumerate(conditions):
        offset = x - (len(conditions) - 1) * width / 2 + i * width
        bars = ax.bar(
            offset,
            weight_df.loc[top_features, cond].values,
            width,
            label=cond,
            color=colors[i % len(colors)],
            alpha=0.8,
        )
        for rect in bars:
            height = rect.get_height()
            if abs(height) > 0.001:
                ax.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    "{0:.2f}".format(height),
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                    rotation=90,
                )

    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Mean Weight")
    ax.set_title("Top {0} Features Across All Conditions".format(len(top_features)))
    ax.legend(fontsize=9, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plot_path = out_path / "weights_all_conditions.png"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    logger.info("Saved comparison plot: %s", plot_path)
    return [str(plot_path)]
