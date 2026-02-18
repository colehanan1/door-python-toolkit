"""
CLI entry point for glomerulus weight vector fitting.

Console script: door-fit-glomerulus-weights

This module provides the same functionality as scripts/fit_glomerulus_weights.py
but is importable as a proper package entry point.
"""

import argparse
import logging
import sys

import numpy as np
import yaml

from door_toolkit.encoder import DoOREncoder
from door_toolkit.glomerulus_features import (
    build_design_matrix,
    load_receptor_to_glomerulus_mapping,
)
from door_toolkit.glomerulus_regression import (
    export_weight_report,
    fit_glomerulus_weight_vector,
)

logger = logging.getLogger(__name__)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fit a baseline glomerulus weight vector from control PER labels.",
    )
    p.add_argument("--config", required=True, help="Path to YAML config file.")
    p.add_argument(
        "--feature-set", choices=["all", "union", "intersection"], default=None,
    )
    p.add_argument("--activation-threshold", type=float, default=None)
    p.add_argument("--agg", choices=["max", "mean", "sum"], default=None)
    p.add_argument("--model", choices=["lasso", "ridge", "elasticnet"], default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--zero-eps", type=float, default=None)
    p.add_argument("--random-state", type=int, default=None)
    p.add_argument("--target-mode", choices=["raw", "centered"], default=None,
                   help="raw: fit raw PER labels; centered: fit y - mean(y)")
    return p.parse_args(argv)


def glomerulus_main(argv=None):
    """Entry point for door-fit-glomerulus-weights console script."""
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})

    feature_set = args.feature_set or defaults.get("feature_set", "union")
    activation_threshold = (
        args.activation_threshold
        if args.activation_threshold is not None
        else defaults.get("activation_threshold", 0.05)
    )
    agg = args.agg or defaults.get("aggregation", "max")
    model = args.model or defaults.get("model", "lasso")
    outdir = args.outdir or defaults.get("outdir", "out/glomerulus_baseline")
    zero_eps = args.zero_eps if args.zero_eps is not None else defaults.get("zero_eps", 1e-6)
    random_state = (
        args.random_state if args.random_state is not None else defaults.get("random_state", 0)
    )
    target_mode = args.target_mode or defaults.get("target_mode", "centered")
    standardize = defaults.get("standardize", True)
    door_cache_path = defaults.get("door_cache_path", "door_cache")
    mapping_csv_path = defaults.get(
        "mapping_csv_path", "data/mappings/door_to_flywire_mapping.csv"
    )

    odor_entries = cfg["odors"]
    odor_names = [e["name"] for e in odor_entries]
    y = np.array([e["per_label"] for e in odor_entries], dtype=np.float64)

    logger.info("Odors: %s", odor_names)
    logger.info("PER labels: %s", y.tolist())

    # Apply target centering if requested
    y_raw = y.copy()  # preserve original for metadata
    y_mean = float(y.mean())
    if target_mode == "centered":
        y = y - y_mean
        logger.info("Target mode: centered (y_mean=%.4f, intercept should ≈ 0)", y_mean)
    else:
        logger.info("Target mode: raw (intercept should ≈ y_mean=%.4f)", y_mean)

    encoder = DoOREncoder(cache_path=door_cache_path, use_torch=False)
    mapping, mapping_meta = load_receptor_to_glomerulus_mapping(mapping_csv_path)

    X, glom_names, design_meta = build_design_matrix(
        odor_names, encoder, mapping,
        feature_set=feature_set,
        activation_threshold=activation_threshold,
        agg=agg,
    )

    logger.info("Design matrix shape: %s", X.shape)

    results = fit_glomerulus_weight_vector(
        X, y, glom_names,
        model=model, standardize=standardize,
        random_state=random_state, zero_eps=zero_eps,
        target_mode=target_mode,
        y_raw=y_raw,
        y_mean=y_mean,
    )

    logger.info(
        "Model: %s | alpha=%.6f | R²=%.4f | MSE=%.6f | +:%d −:%d 0:%d",
        results["model_name"], results["alpha"], results["r2"], results["mse"],
        int((results["weight_signs"] > 0).sum()),
        int((results["weight_signs"] < 0).sum()),
        int((results["weight_signs"] == 0).sum()),
    )

    extra_meta = {
        "config_file": str(args.config),
        "feature_set": feature_set,
        "activation_threshold": activation_threshold,
        "aggregation": agg,
        "odors": odor_names,
        "per_labels": y_raw.tolist(),  # Always export original raw labels
        "n_all_receptors": design_meta["n_all_receptors"],
        "per_odor_active_counts": design_meta["per_odor_active_counts"],
    }
    paths = export_weight_report(results, outdir, extra_meta=extra_meta)

    logger.info("Results written to: %s", outdir)
    for key, path in paths.items():
        logger.info("  %s: %s", key, path)


if __name__ == "__main__":
    glomerulus_main()
