#!/usr/bin/env python3
"""
Full-fit regression (no cross-validation).

Trains on ALL odors at once for each condition. No leave-one-out.
Uses inner LOOCV only for alpha selection (regularization strength),
then fits the final model on all data.

Outputs per condition:
  - weights CSV
  - predictions (in-sample) CSV
  - comparison plots (predicted vs true ΔPER)
  - weights bar chart

Usage:
    python scripts/run_full_fit.py \
        --csv /tmp/reaction_rates_no_citral.csv \
        --control-row opto_AIR \
        --conditions opto_AIR,opto_EB,opto_hex,opto_benz_1,opto_3-oct \
        --model lasso \
        --feature-set intersection \
        --activation-threshold 0.05 \
        --plot --plot-top-n 13 \
        --outdir out/full_fit
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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
from door_toolkit.multicond_loocv import (
    load_per_csv,
    detect_odor_columns,
    impute_control_missing,
    make_targets,
    _fit_one,
    _select_alpha,
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


def _resolve_door_name(csv_col: str) -> str:
    if csv_col in CSV_ODOR_TO_DOOR:
        return CSV_ODOR_TO_DOOR[csv_col]
    low = csv_col.lower().replace(" ", "_")
    if low in CSV_ODOR_TO_DOOR:
        return CSV_ODOR_TO_DOOR[low]
    return csv_col


def full_fit_condition(
    X: np.ndarray,
    y_centered: np.ndarray,
    *,
    model: str = "lasso",
    alpha: Optional[float] = None,
    alpha_grid: Optional[Sequence[float]] = None,
    l1_ratio: float = 0.5,
    seed: int = 0,
    standardize: bool = True,
    zero_eps: float = 1e-6,
    min_nonzero: int = 1,
) -> Dict:
    """Fit regression on ALL data (no leave-one-out).

    If alpha is given, use it directly. Otherwise pick alpha via inner LOOCV.

    Returns dict with: coef, intercept, y_pred, alpha, metrics
    """
    if alpha_grid is None:
        alpha_grid = list(np.logspace(-4, 1, 60))

    valid_mask = np.isfinite(y_centered)
    X_valid = X[valid_mask]
    y_valid = y_centered[valid_mask]
    n_valid = len(y_valid)

    if n_valid < 3:
        raise ValueError(f"Need >= 3 valid samples, got {n_valid}")

    if alpha is not None:
        # Use fixed alpha directly (no inner LOOCV)
        best_alpha = alpha
        inner_mse = float("nan")
    else:
        # Select best alpha via inner LOOCV
        best_alpha, inner_mse, score_df = _select_alpha(
            X_valid, y_valid,
            model=model, alpha_grid=alpha_grid, l1_ratio=l1_ratio,
            random_state=seed, standardize=standardize,
            zero_eps=zero_eps, min_nonzero=min_nonzero,
        )

    # Fit final model on ALL valid data
    coef, intercept, y_pred_valid = _fit_one(
        X_valid, y_valid,
        model=model, alpha=best_alpha, random_state=seed,
        l1_ratio=l1_ratio, standardize=standardize,
    )

    # Full predictions (including NaN positions)
    y_pred = np.full(len(y_centered), np.nan)
    y_pred[valid_mask] = y_pred_valid

    # Metrics (in-sample)
    ss_res = float(np.sum((y_valid - y_pred_valid) ** 2))
    ss_tot = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    mse = float(np.mean((y_valid - y_pred_valid) ** 2))

    if n_valid >= 3:
        r_val, r_pval = pearsonr(y_valid, y_pred_valid)
    else:
        r_val, r_pval = float("nan"), float("nan")

    n_nonzero = int(np.sum(np.abs(coef) > zero_eps))

    return {
        "coef": coef,
        "intercept": intercept,
        "y_pred": y_pred,
        "best_alpha": best_alpha,
        "inner_loo_mse": inner_mse,
        "metrics": {
            "train_mse": mse,
            "train_r": float(r_val),
            "train_r2": r2,
            "n_samples": n_valid,
            "n_nonzero": n_nonzero,
        },
    }


def plot_full_fit(
    out_dir: Path,
    odor_labels: List[str],
    feature_names: List[str],
    condition_results: Dict[str, Dict],
    control_row: str,
    plot_top_n: int = None,
) -> List[str]:
    """Generate plots for full-fit results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []

    conditions = list(condition_results.keys())
    # Separate control from trained
    trained_conds = [c for c in conditions if c != control_row]
    n_trained = len(trained_conds)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # --- Plot 1: Predicted vs True ΔPER per condition ---
    ncols = min(2, n_trained)
    nrows = (n_trained + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if n_trained == 1:
        axes = [axes]
    else:
        axes = np.array(axes).flatten()

    for idx, cond in enumerate(sorted(trained_conds)):
        ax = axes[idx]
        data = condition_results[cond]
        y_true = data["y_centered"]
        y_pred = data["fit_result"]["y_pred"]

        # Center for plotting
        valid = np.isfinite(y_true) & np.isfinite(y_pred)
        x = np.arange(len(odor_labels))
        width = 0.35

        true_vals = np.where(valid, y_true, 0)
        pred_vals = np.where(valid, y_pred, 0)

        ax.bar(x - width / 2, true_vals, width, label="True ΔPER", color="#1f77b4", alpha=0.8)
        ax.bar(x + width / 2, pred_vals, width, label="Predicted ΔPER", color="#ff7f0e", alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

        metrics = data["fit_result"]["metrics"]
        ax.set_ylabel("ΔPER (centered)")
        ax.set_title(f"{cond}\nR² = {metrics['train_r2']:.3f}  |  MSE = {metrics['train_mse']:.6f}")
        ax.set_xticks(x)
        ax.set_xticklabels(
            [o.replace("_", " ") for o in odor_labels],
            fontsize=8, rotation=45, ha="right",
        )
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    for i in range(n_trained, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    p = out_dir / "full_fit_predictions_vs_true.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    # --- Plot 2: Scatter (all conditions) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    all_true, all_pred = [], []

    for (cond, color) in zip(sorted(trained_conds), colors):
        data = condition_results[cond]
        y_t = data["y_centered"]
        y_p = data["fit_result"]["y_pred"]
        valid = np.isfinite(y_t) & np.isfinite(y_p)
        all_true.extend(y_t[valid])
        all_pred.extend(y_p[valid])
        ax.scatter(y_t[valid], y_p[valid], label=cond, s=100, alpha=0.7, color=color)

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - np.mean(all_true)) ** 2)
    overall_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    lim = max(abs(all_true).max(), abs(all_pred).max()) + 0.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1.5, alpha=0.5, label="Perfect")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("True ΔPER (centered)", fontsize=12)
    ax.set_ylabel("Predicted ΔPER (centered)", fontsize=12)
    ax.set_title(f"Full Fit: Predicted vs True\nOverall R² = {overall_r2:.3f}")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    p = out_dir / "full_fit_scatter.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    # --- Plot 3: Per-condition weights vs control (only changed receptors) ---
    if control_row in condition_results:
        ctrl_coef = condition_results[control_row]["fit_result"]["coef"]
        zero_eps = 1e-6

        for idx, cond in enumerate(sorted(trained_conds)):
            cond_coef = condition_results[cond]["fit_result"]["coef"]
            delta = cond_coef - ctrl_coef
            changed_mask = np.abs(delta) > zero_eps

            if not np.any(changed_mask):
                continue

            changed_idx = np.where(changed_mask)[0]
            # Optionally limit to top-N changed receptors by |delta|
            if plot_top_n is not None and plot_top_n > 0 and plot_top_n < len(changed_idx):
                top_local = np.argsort(np.abs(delta[changed_idx]))[-plot_top_n:]
                changed_idx = changed_idx[np.sort(top_local)]

            display_features = [feature_names[i] for i in changed_idx]
            ctrl_vals = ctrl_coef[changed_idx]
            cond_vals = cond_coef[changed_idx]

            fig, ax = plt.subplots(figsize=(max(10, len(display_features) * 0.8), 6))
            x = np.arange(len(display_features))
            width = 0.35

            bars_ctrl = ax.bar(
                x - width / 2, ctrl_vals, width,
                label=control_row, color="#7f7f7f", alpha=0.8,
            )
            bars_cond = ax.bar(
                x + width / 2, cond_vals, width,
                label=cond, color=colors[idx % len(colors)], alpha=0.8,
            )

            # Label each bar with its weight value
            y_min, y_max = ax.get_ylim()
            y_pad = 0.02 * (y_max - y_min if y_max != y_min else 1.0)
            for bars, vals in [(bars_ctrl, ctrl_vals), (bars_cond, cond_vals)]:
                for bar, val in zip(bars, vals):
                    if not np.isfinite(val):
                        continue
                    y = bar.get_height()
                    va = "bottom" if y >= 0 else "top"
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        y + (y_pad if y >= 0 else -y_pad),
                        f"{val:+.3f}",
                        ha="center",
                        va=va,
                        fontsize=7,
                        rotation=90,
                    )

            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Receptor")
            ax.set_ylabel("Weight")
            ax.set_title(
                f"Weights: {control_row} vs {cond} "
                f"({len(display_features)} changed receptors)"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(display_features, rotation=45, ha="right", fontsize=10)
            ax.legend()
            ax.grid(axis="y", alpha=0.3)

            fig.tight_layout()
            slug = cond.replace("-", "_")
            p = out_dir / f"full_fit_weights_{slug}_vs_control.png"
            fig.savefig(p, dpi=200, bbox_inches="tight")
            plt.close(fig)
            plot_paths.append(str(p))

    # --- Plot 4: R² summary bar chart ---
    fig, ax = plt.subplots(figsize=(8, 5))
    r2_vals = [condition_results[c]["fit_result"]["metrics"]["train_r2"] for c in sorted(trained_conds)]
    bars = ax.bar(sorted(trained_conds), r2_vals, color=colors[:n_trained], alpha=0.8, edgecolor="black")
    for bar, r2 in zip(bars, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{r2:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_ylabel("R² (in-sample)")
    ax.set_title("Full-Fit R² Per Condition (In-Sample)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    p = out_dir / "full_fit_r2_summary.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plot_paths.append(str(p))

    return plot_paths


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Full-fit regression on all odors (no leave-one-out cross-validation)."
    )
    p.add_argument("--csv", required=True, help="Path to PER CSV.")
    p.add_argument("--control-row", default="opto_AIR")
    p.add_argument("--conditions", required=True, help="Comma-separated conditions.")
    p.add_argument(
        "--model", choices=["lasso", "elasticnet"], default="lasso",
        help="Regression model (default: lasso).",
    )
    p.add_argument("--outdir", default="out/full_fit")
    p.add_argument("--seed", type=int, default=0)

    # Feature settings
    p.add_argument("--door-cache", default="door_cache")
    p.add_argument("--mapping-csv", default="data/mappings/door_to_flywire_mapping.csv")
    p.add_argument(
        "--feature-set",
        choices=["all", "union", "intersection", "no_blanks"],
        default="intersection",
    )
    p.add_argument("--activation-threshold", type=float, default=0.0)
    p.add_argument("--agg", choices=["max", "mean", "sum"], default="max")

    # Model settings
    p.add_argument("--l1-ratio", type=float, default=0.5, help="L1 ratio for elasticnet.")
    p.add_argument(
        "--alpha", type=float, default=None,
        help="Fixed regularization alpha (skip inner LOOCV alpha selection). "
             "Lower = less regularization = more receptors kept. Try 0.001.",
    )
    p.add_argument("--no-standardize", action="store_true")
    p.add_argument(
        "--no-center-control", action="store_true",
        help="Do NOT mean-center control (opto_AIR) targets.",
    )
    p.add_argument(
        "--no-center-delta", action="store_true",
        help="Do NOT mean-center ΔPER targets for trained conditions.",
    )

    # Plot settings
    p.add_argument("--plot", action="store_true", help="Generate plots.")
    p.add_argument("--plot-top-n", type=int, default=None, help="Top N receptors in weights plot.")

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    conditions = [c.strip() for c in args.conditions.split(",")]
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load CSV
    df = load_per_csv(args.csv)
    df = impute_control_missing(df, args.control_row)
    odors = detect_odor_columns(df)

    print(f"\n{'='*70}")
    print("FULL-FIT REGRESSION (no cross-validation)")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"CSV: {args.csv}")
    print(f"Conditions: {conditions}")
    print(f"Odors ({len(odors)}): {odors}")

    # Build feature matrix
    door_odors = [_resolve_door_name(o) for o in odors]
    encoder = DoOREncoder(cache_path=args.door_cache, use_torch=False)
    mapping, _ = load_receptor_to_glomerulus_mapping(args.mapping_csv)

    X, feature_names, meta = build_design_matrix(
        door_odors, encoder, mapping,
        feature_set=args.feature_set,
        activation_threshold=args.activation_threshold,
        agg=args.agg,
    )
    print(f"Feature matrix: {X.shape} (odors × receptors)")
    print(f"Receptors ({len(feature_names)}): {feature_names}")

    # Fit each condition
    condition_results = {}

    for cond in conditions:
        y_raw, y_centered, centering_mean = make_targets(
            df, odors, args.control_row, cond,
        )

        is_control = (cond == args.control_row)
        if is_control and args.no_center_control:
            y_centered = y_raw.copy()
            centering_mean = 0.0
        if (not is_control) and args.no_center_delta:
            y_centered = y_raw.copy()
            centering_mean = 0.0
        if is_control:
            target_type = "raw PER (uncentered)" if args.no_center_control else "raw PER (centered)"
        else:
            target_type = "ΔPER (uncentered)" if args.no_center_delta else "ΔPER (centered)"

        print(f"\n--- {cond} ---")
        print(f"  Target: {target_type}")
        print(f"  y_raw:      {np.array2string(y_raw, precision=4)}")
        print(f"  y_centered: {np.array2string(y_centered, precision=4)}")

        fit_result = full_fit_condition(
            X, y_centered,
            model=args.model,
            alpha=args.alpha,
            l1_ratio=args.l1_ratio,
            seed=args.seed,
            standardize=not args.no_standardize,
        )

        m = fit_result["metrics"]
        print(f"  Alpha: {fit_result['best_alpha']:.6f}")
        print(f"  Non-zero weights: {m['n_nonzero']}/{len(feature_names)}")
        print(f"  Train R²: {m['train_r2']:.4f}")
        print(f"  Train MSE: {m['train_mse']:.6f}")
        print(f"  Predicted: {np.array2string(fit_result['y_pred'], precision=4)}")

        condition_results[cond] = {
            "y_raw": y_raw,
            "y_centered": y_centered,
            "centering_mean": centering_mean,
            "fit_result": fit_result,
        }

        # Export weights CSV
        slug = cond.replace("-", "_")
        weights_df = pd.DataFrame({
            "feature": feature_names,
            "weight": fit_result["coef"],
        })
        weights_df.to_csv(outdir / f"weights_{slug}.csv", index=False)

        # Export predictions CSV
        pred_df = pd.DataFrame({
            "odor": odors,
            "true_centered": y_centered,
            "predicted_centered": fit_result["y_pred"],
            "true_raw": y_raw,
        })
        pred_df.to_csv(outdir / f"predictions_{slug}.csv", index=False)

        # Export summary JSON
        summary = {
            "condition": cond,
            "model": args.model,
            "target_type": target_type,
            "best_alpha": fit_result["best_alpha"],
            "metrics": m,
            "feature_count": len(feature_names),
            "odors": odors,
            "centering_mean": centering_mean,
        }
        with open(outdir / f"summary_{slug}.json", "w") as f:
            json.dump(summary, f, indent=2)

    # Overview CSV
    overview_rows = []
    for cond in conditions:
        m = condition_results[cond]["fit_result"]["metrics"]
        overview_rows.append({
            "condition": cond,
            "model": args.model,
            "train_r2": m["train_r2"],
            "train_mse": m["train_mse"],
            "train_r": m["train_r"],
            "n_nonzero": m["n_nonzero"],
            "n_samples": m["n_samples"],
            "alpha": condition_results[cond]["fit_result"]["best_alpha"],
        })
    overview_df = pd.DataFrame(overview_rows)
    overview_df.to_csv(outdir / "overview.csv", index=False)

    # Plots
    if args.plot:
        print(f"\n{'='*70}")
        print("GENERATING PLOTS")
        print(f"{'='*70}")

        plots = plot_full_fit(
            outdir, odors, feature_names,
            condition_results, args.control_row,
            plot_top_n=args.plot_top_n,
        )
        for p in plots:
            print(f"  {p}")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Model: {args.model}")
    print(f"Receptors: {len(feature_names)}")
    print(f"Odors: {len(odors)}")
    print()
    for cond in conditions:
        m = condition_results[cond]["fit_result"]["metrics"]
        nz = m["n_nonzero"]
        print(f"  {cond:20s}  R²={m['train_r2']:.4f}  MSE={m['train_mse']:.6f}  non-zero={nz}/{len(feature_names)}")

    print(f"\nOutputs: {outdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
