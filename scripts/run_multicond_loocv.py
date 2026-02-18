#!/usr/bin/env python3
"""
Run multi-condition leave-one-odor-out LOOCV regression.

Fits one regression model per condition:
  - Control (opto_AIR): raw PER, mean-centered.
  - Trained conditions: ΔPER = trained − control, mean-centered.

Features are the intersection set across all 7 odors using the DoOR
receptor feature builder.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Ensure src/ is importable when running as a standalone script.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from door_toolkit.encoder import DoOREncoder
from door_toolkit.glomerulus_features import (
    build_design_matrix,
    load_receptor_to_glomerulus_mapping,
)
from door_toolkit.multicond_loocv import run_multicond_loocv
from door_toolkit.multicond_loocv_plots import (
    plot_weights_and_deltaperby_odor,
    plot_condition_comparison,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Odor-name mapping: CSV column names -> DoOR names
# ---------------------------------------------------------------------------
# The PER CSV uses short/lab names; DoOR expects canonical chemical names.
# We map each of the 7 CSV columns to the DoOR name used by the encoder.

CSV_ODOR_TO_DOOR = {
    "3-Octonol": "3-octanol",
    "3-octonol": "3-octanol",
    "Apple_Cider_Vinegar": "acetic acid",
    "apple_cider_vinegar": "acetic acid",
    "Benzaldehyde": "benzaldehyde",
    "benzaldehyde": "benzaldehyde",
    "Citral": "citral",
    "citral": "citral",
    "Ethyl_Butyrate": "ethyl butyrate",
    "ethyl_butyrate": "ethyl butyrate",
    "Hexanol": "1-hexanol",
    "hexanol": "1-hexanol",
    "Linalool": "linalool",
    "linalool": "linalool",
}


def _resolve_door_name(csv_col: str) -> str:
    """Map a CSV odor column name to a DoOR name."""
    if csv_col in CSV_ODOR_TO_DOOR:
        return CSV_ODOR_TO_DOOR[csv_col]
    # Fallback: try lowercase
    low = csv_col.lower().replace(" ", "_")
    if low in CSV_ODOR_TO_DOOR:
        return CSV_ODOR_TO_DOOR[low]
    # Last resort: use as-is (the encoder does its own fuzzy matching)
    return csv_col


def _build_feature_builder(
    *,
    door_cache: str,
    mapping_csv: str,
    feature_set: str,
    activation_threshold: float,
    agg: str,
):
    """Create a feature builder callable for the multicond pipeline."""
    encoder = DoOREncoder(cache_path=door_cache, use_torch=False)
    mapping, mapping_meta = load_receptor_to_glomerulus_mapping(mapping_csv)
    logger.info(
        "Loaded DoOR mapping: %d receptors (adult_only=%s)",
        mapping_meta.get("n_receptors_mapped", -1),
        mapping_meta.get("adult_only", True),
    )

    def _builder(
        csv_odors: List[str],
    ) -> Tuple[np.ndarray, List[str], dict]:
        door_odors = [_resolve_door_name(o) for o in csv_odors]
        logger.info("CSV odors -> DoOR: %s", list(zip(csv_odors, door_odors)))
        X, feature_names, meta = build_design_matrix(
            door_odors,
            encoder,
            mapping,
            feature_set=feature_set,
            activation_threshold=activation_threshold,
            agg=agg,
        )
        return X, feature_names, meta

    return _builder


def _parse_alpha_grid(text: str) -> List[float]:
    if text.strip().lower() == "default":
        return list(np.logspace(-4, 1, 60))
    return [float(v.strip()) for v in text.split(",") if v.strip()]


def _parse_conditions(text: str) -> List[str]:
    return [t.strip() for t in text.split(",") if t.strip()]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-condition leave-one-odor-out LOOCV regression."
    )
    p.add_argument(
        "--csv", required=True,
        help="Path to PER CSV (reaction_rates_summary_unordered.csv).",
    )
    p.add_argument(
        "--control-row", default="opto_AIR",
        help="Control condition row label.",
    )
    p.add_argument(
        "--conditions", required=True,
        help="Comma-separated conditions (including control if desired).",
    )
    p.add_argument(
        "--model", choices=["lasso", "elasticnet"], default="lasso",
    )
    p.add_argument("--outdir", default="out/multicond_loocv")

    # Feature builder settings
    p.add_argument("--door-cache", default="door_cache")
    p.add_argument(
        "--mapping-csv",
        default="data/mappings/door_to_flywire_mapping.csv",
    )
    p.add_argument(
        "--feature-set",
        choices=["all", "union", "intersection", "no_blanks"],
        default="no_blanks",
        help="all=60 receptors; union=54 active; intersection=1; no_blanks=57 (excludes 3 all-zero receptors)",
    )
    p.add_argument("--activation-threshold", type=float, default=0.0)
    p.add_argument("--agg", choices=["max", "mean", "sum"], default="max")

    # Sparse-fit settings
    p.add_argument(
        "--alpha-grid", default="default",
        help="Comma-separated alpha grid or 'default'.",
    )
    p.add_argument("--l1-ratio", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--no-standardize", dest="standardize", action="store_false",
        default=True,
    )
    p.add_argument("--zero-eps", type=float, default=1e-6)
    p.add_argument("--min-nonzero", type=int, default=1)

    # Plotting options
    p.add_argument(
        "--plot", action="store_true",
        help="Generate per-odor baseline vs. delta weight plots.",
    )
    p.add_argument(
        "--plot-top-n", type=int, default=10,
        help="Number of top features to plot per odor (default: 10).",
    )
    p.add_argument(
        "--plot-outdir", default=None,
        help="Output directory for plots (default: <outdir>/plots).",
    )
    p.add_argument(
        "--plot-baseline-weights", default=None,
        help="Path to baseline weights CSV (feature, baseline_w columns).",
    )
    p.add_argument(
        "--plot-comparison", action="store_true",
        help="Also plot all conditions comparison across top features.",
    )

    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s %(levelname)s: %(message)s",
    )

    conditions = _parse_conditions(args.conditions)
    alpha_grid = _parse_alpha_grid(args.alpha_grid)

    feature_builder = _build_feature_builder(
        door_cache=args.door_cache,
        mapping_csv=args.mapping_csv,
        feature_set=args.feature_set,
        activation_threshold=args.activation_threshold,
        agg=args.agg,
    )

    result = run_multicond_loocv(
        csv_path=args.csv,
        control_row=args.control_row,
        conditions=conditions,
        feature_builder=feature_builder,
        model=args.model,
        alpha_grid=alpha_grid,
        l1_ratio=args.l1_ratio,
        seed=args.seed,
        standardize=args.standardize,
        zero_eps=args.zero_eps,
        min_nonzero=args.min_nonzero,
        outdir=args.outdir,
    )

    # Plotting
    if args.plot:
        plot_outdir = args.plot_outdir or str(Path(args.outdir) / "plots")

        # Load baseline weights if provided
        baseline_df = None
        if args.plot_baseline_weights:
            baseline_df = pd.read_csv(args.plot_baseline_weights)
            if "receptor" in baseline_df.columns:
                baseline_df = baseline_df.rename(
                    columns={"receptor": "feature"}
                )
            elif "feature" not in baseline_df.columns:
                raise ValueError(
                    "Baseline weights CSV must have 'feature' or 'receptor' column"
                )

        plots = plot_weights_and_deltaperby_odor(
            plot_outdir,
            odors=result["odors"],
            feature_names=result["feature_names"],
            condition_data=result["condition_data"],
            baseline_weights=baseline_df,
            top_n=args.plot_top_n,
            control_row=args.control_row,
        )
        print("Plots written ({0}):".format(len(plots)))
        for p in plots:
            print("  {0}".format(p))

        if args.plot_comparison:
            comp_plots = plot_condition_comparison(
                plot_outdir,
                conditions=result["conditions"],
                feature_names=result["feature_names"],
                condition_data=result["condition_data"],
                top_n=args.plot_top_n,
                control_row=args.control_row,
            )
            print("Comparison plots written ({0}):".format(len(comp_plots)))
            for p in comp_plots:
                print("  {0}".format(p))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
