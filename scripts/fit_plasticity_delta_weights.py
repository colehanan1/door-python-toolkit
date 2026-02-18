#!/usr/bin/env python3
"""
Fit sparse plasticity delta-weights from opto behavioral CSV.

Pipeline:
1) Load PER matrix from CSV and resolve 4 odor columns.
2) Compute delta PER = trained - control (opto_AIR by default).
3) Optionally mean-center each condition's delta vector.
4) Build DoOR feature matrix with existing glomerulus feature builder.
5) Fit one sparse model per training condition and export reports.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

# Ensure src/ is importable when running as a standalone script.
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from door_toolkit.encoder import DoOREncoder
from door_toolkit.glomerulus_features import (
    build_design_matrix,
    load_receptor_to_glomerulus_mapping,
)
from door_toolkit.plasticity_delta_fit import (
    ODOR_KEY_TO_DOOR,
    compute_delta,
    export_reports,
    export_baseline_comparison_reports,
    extract_four_odors,
    fit_delta_weights_per_condition,
    filter_training_conditions,
    load_per_matrix,
    load_baseline_weights,
    mean_center_rows,
    parse_odor_argument,
    sanitize_filename_token,
    validate_feature_alignment,
)


logger = logging.getLogger(__name__)


def _parse_alpha_grid(alpha_grid_text: str) -> List[float]:
    if alpha_grid_text.strip().lower() == "default":
        return list(np.logspace(-4, 1, 60))
    values = [token.strip() for token in alpha_grid_text.split(",") if token.strip()]
    if not values:
        raise ValueError("alpha grid cannot be empty")
    return [float(v) for v in values]


def _group_key(row_name: str) -> str:
    normalized = "".join(ch for ch in row_name.lower() if ch.isalnum())
    if "benz" in normalized:
        return "benz"
    if "hex" in normalized:
        return "hex"
    if "3oct" in normalized:
        return "3-oct"
    if "eb" in normalized:
        return "eb"
    return "other"


def _normalize_label(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


def _resolve_row_label_local(index_values: Iterable[str], requested: str) -> str:
    index_list = [str(x) for x in index_values]
    if requested in index_list:
        return requested
    target = _normalize_label(requested)
    matches = [idx for idx in index_list if _normalize_label(idx) == target]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous control row '{0}' matches multiple rows: {1}".format(requested, matches)
        )
    raise ValueError("Control row '{0}' not found in CSV index".format(requested))


def _print_schema_evidence(
    df,
    odor_colmap: Dict[str, str],
    requested_control_row: str,
    resolved_control_row: str,
    conditions: List[str],
) -> None:
    print("=== CSV SCHEMA EVIDENCE ===")
    print("Rows (index labels):")
    for i, row_name in enumerate(df.index):
        print("  {0:2d}: {1}".format(i, row_name))
    print("Columns (odor labels):")
    for i, col_name in enumerate(df.columns):
        print("  {0:2d}: {1}".format(i, col_name))
    print("Control row requested:", requested_control_row)
    print("Control row resolved :", resolved_control_row)
    print("Control row present  :", resolved_control_row in df.index)
    print("")

    print("Resolved 4-odor mapping (canonical -> CSV column -> DoOR odor):")
    for odor_key in odor_colmap:
        print(
            "  {0} -> {1} -> {2}".format(
                odor_key,
                odor_colmap[odor_key],
                ODOR_KEY_TO_DOOR[odor_key],
            )
        )
    print("")

    grouped = {"eb": [], "hex": [], "benz": [], "3-oct": [], "other": []}
    for condition in conditions:
        grouped[_group_key(condition)].append(condition)

    print("Training rows discovered (opto_* minus control):")
    for key in ["eb", "hex", "benz", "3-oct", "other"]:
        print("  {0:6s}: {1}".format(key, grouped[key]))
    print("")


def _print_delta_repro(
    df4,
    delta_raw,
    delta_centered,
    *,
    control_row: str,
) -> None:
    control_vec = df4.loc[control_row]
    print("=== DELTA REPRODUCTION ===")
    print("Control PER ({0}): {1}".format(control_row, control_vec.to_dict()))
    for condition_name in delta_raw.index:
        trained_vec = df4.loc[condition_name]
        raw_vec = delta_raw.loc[condition_name]
        centered_vec = delta_centered.loc[condition_name]
        mean_raw = float(raw_vec.mean(skipna=True))
        print("")
        print("Condition:", condition_name)
        print("  PER_trained      =", trained_vec.to_dict())
        print("  PER_control      =", control_vec.to_dict())
        print("  delta_raw        =", raw_vec.to_dict())
        print("  mean(delta_raw)  = {0:.6f}".format(mean_raw))
        print("  delta_centered   =", centered_vec.to_dict())
    print("")


def _build_feature_builder(
    *,
    door_cache: str,
    mapping_csv: str,
    feature_set: str,
    activation_threshold: float,
    agg: str,
):
    encoder = DoOREncoder(cache_path=door_cache, use_torch=False)
    mapping, mapping_meta = load_receptor_to_glomerulus_mapping(mapping_csv)
    logger.info(
        "Loaded DoOR mapping: %d receptors mapped (adult_only=%s)",
        mapping_meta.get("n_receptors_mapped", -1),
        mapping_meta.get("adult_only", True),
    )

    def _builder(odor_keys):
        door_odors = [ODOR_KEY_TO_DOOR[odor_key] for odor_key in odor_keys]
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


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fit sparse plasticity delta-weight models.")
    p.add_argument(
        "--csv",
        required=True,
        help="Path to reaction_rates_summary_unordered.csv",
    )
    p.add_argument(
        "--control-row",
        default="opto_AIR",
        help="Control row label used for delta subtraction.",
    )
    p.add_argument(
        "--odors",
        default="benzaldehyde,ethyl_butyrate,1-hexanol,3-octanol",
        help="Comma-separated 4 target odors (aliases accepted).",
    )
    p.add_argument("--center-delta", dest="center_delta", action="store_true", default=True)
    p.add_argument("--no-center-delta", dest="center_delta", action="store_false")
    p.add_argument("--model", choices=["lasso", "elasticnet"], default="lasso")
    p.add_argument("--outdir", default="out/plasticity_delta_4odors")
    p.add_argument(
        "--conditions",
        default=None,
        help="Comma-separated allowlist of training conditions (e.g., opto_EB,opto_hex).",
    )
    p.add_argument(
        "--exclude-conditions",
        default=None,
        help="Comma-separated blocklist of training conditions to skip.",
    )
    p.add_argument(
        "--baseline-outdir",
        default=None,
        help="Baseline output dir containing weights.csv + model_summary.json.",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Generate contribution plots per odor across all fitted conditions.",
    )
    p.add_argument(
        "--plot-top-n",
        type=int,
        default=12,
        help="Number of top features to plot per odor (default: 12).",
    )
    p.add_argument(
        "--plot-outdir",
        default=None,
        help="Output directory for plots (default: <outdir>/plots).",
    )
    p.add_argument(
        "--plot-weight-delta",
        action="store_true",
        help="Generate baseline/delta weight plots per odor with ΔPER bars.",
    )
    p.add_argument(
        "--plot-weight-top-n",
        type=int,
        default=None,
        help="Top-N features for weight plots (default: uses --min-nonzero).",
    )

    # Feature-builder settings aligned with existing glomerulus pipeline.
    p.add_argument("--door-cache", default="door_cache")
    p.add_argument("--mapping-csv", default="data/mappings/door_to_flywire_mapping.csv")
    p.add_argument("--feature-set", choices=["all", "union", "intersection"], default="intersection")
    p.add_argument("--activation-threshold", type=float, default=0.05)
    p.add_argument("--agg", choices=["max", "mean", "sum"], default="max")

    # Sparse-fit settings.
    p.add_argument("--alpha-grid", default="default", help="Comma-separated alpha grid or 'default'.")
    p.add_argument("--l1-ratio", type=float, default=0.5, help="ElasticNet l1_ratio.")
    p.add_argument("--random-state", type=int, default=0)
    p.add_argument("--zero-eps", type=float, default=1e-6)
    p.add_argument("--no-standardize", dest="standardize", action="store_false", default=True)
    p.add_argument(
        "--min-nonzero",
        type=int,
        default=3,
        help="Prefer solutions with at least this many non-zero weights (default: 3).",
    )
    return p.parse_args(argv)


def _parse_condition_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    tokens = [tok.strip() for tok in value.split(",") if tok.strip()]
    return tokens


def _plot_contributions(
    plot_outdir: str,
    *,
    base_outdir: str,
    odor_colmap: Dict[str, str],
    conditions: List[str],
    plot_top_n: int,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    out_path = Path(plot_outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    base_path = Path(base_outdir)

    odor_keys = list(odor_colmap.keys())
    for odor_key in odor_keys:
        odor_slug = sanitize_filename_token(odor_colmap[odor_key])
        series_map = {}
        for condition in conditions:
            cond_slug = sanitize_filename_token(condition)
            contrib_path = base_path / "contrib_{0}_{1}.csv".format(cond_slug, odor_slug)
            if not contrib_path.exists():
                raise FileNotFoundError("Missing contribution file: {0}".format(contrib_path))
            df = pd.read_csv(contrib_path)
            series_map[condition] = df.set_index("feature")["contribution"]

        # Align features, pick top-N by max abs contribution across conditions.
        all_features = sorted(set().union(*[s.index for s in series_map.values()]))
        contrib_matrix = []
        for condition in conditions:
            s = series_map[condition].reindex(all_features).fillna(0.0)
            contrib_matrix.append(s.to_numpy())
        contrib_matrix = np.stack(contrib_matrix, axis=1)  # (n_features, n_conditions)
        max_abs = np.max(np.abs(contrib_matrix), axis=1)
        top_idx = np.argsort(max_abs)[::-1][: plot_top_n or len(all_features)]
        top_features = [all_features[i] for i in top_idx]
        top_matrix = contrib_matrix[top_idx, :]

        # Plot grouped bars.
        n_features = len(top_features)
        n_conditions = len(conditions)
        x = np.arange(n_features)
        width = 0.8 / max(n_conditions, 1)

        fig, ax = plt.subplots(figsize=(max(8, n_features * 0.6), 4.5))
        for i, condition in enumerate(conditions):
            bars = ax.bar(
                x + (i - (n_conditions - 1) / 2) * width,
                top_matrix[:, i],
                width,
                label=condition,
            )
            for rect in bars:
                height = rect.get_height()
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
        ax.set_xticklabels(top_features, rotation=60, ha="right")
        ax.set_ylabel("Contribution (x * Δw)")
        ax.set_title("Top {0} contributions for {1}".format(len(top_features), odor_key))
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()

        plot_path = out_path / "contrib_{0}_all_conditions.png".format(odor_slug)
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    return [str(p) for p in out_path.glob("contrib_*_all_conditions.png")]


def _plot_weight_deltas(
    plot_outdir: str,
    *,
    odor_colmap: Dict[str, str],
    conditions: List[str],
    baseline_df,
    fit_results: dict,
    delta_centered,
    top_n: int,
    zero_eps: float,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out_path = Path(plot_outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    baseline_map = baseline_df.set_index("feature")["baseline_w"]
    feature_names = list(fit_results["feature_names"])

    # Build max abs delta per feature across conditions.
    delta_matrix = []
    for condition in conditions:
        weights = np.asarray(fit_results["condition_results"][condition]["weights"], dtype=np.float64)
        delta_matrix.append(weights)
    delta_matrix = np.stack(delta_matrix, axis=1)  # (n_features, n_conditions)
    max_abs = np.max(np.abs(delta_matrix), axis=1)
    order = np.argsort(max_abs)[::-1]
    if top_n is not None and top_n > 0:
        order = order[:top_n]
    top_features = [feature_names[i] for i in order]

    # Precompute baseline and delta weight vectors for plotting
    baseline_vals = baseline_map.reindex(top_features).fillna(0.0).to_numpy()
    delta_vals = {cond: fit_results["condition_results"][cond]["weights"][order] for cond in conditions}

    odor_keys = list(odor_colmap.keys())
    for odor_key in odor_keys:
        fig, axes = plt.subplots(
            nrows=2,
            figsize=(max(8, len(top_features) * 0.6), 7),
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax_w = axes[0]
        ax_p = axes[1]

        x = np.arange(len(top_features))
        n_series = 1 + len(conditions)
        width = 0.8 / max(n_series, 1)

        # Baseline weights
        bars = ax_w.bar(
            x - (n_series - 1) * width / 2,
            baseline_vals,
            width,
            label="baseline",
            color="#6A5ACD",
        )
        for rect in bars:
            height = rect.get_height()
            ax_w.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                "{0:.2f}".format(height),
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=7,
                rotation=90,
            )

        # Delta weights per condition
        for i, condition in enumerate(conditions):
            offset = x - (n_series - 1) * width / 2 + (i + 1) * width
            bars = ax_w.bar(
                offset,
                delta_vals[condition],
                width,
                label=condition,
            )
            for rect in bars:
                height = rect.get_height()
                ax_w.text(
                    rect.get_x() + rect.get_width() / 2.0,
                    height,
                    "{0:.2f}".format(height),
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=7,
                    rotation=90,
                )

        ax_w.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax_w.set_xticks(x)
        ax_w.set_xticklabels(top_features, rotation=60, ha="right")
        ax_w.set_ylabel("Weight")
        ax_w.set_title("Baseline vs Δweights (top {0})".format(len(top_features)))
        ax_w.legend(fontsize=8, ncol=2)

        # ΔPER (centered) bars per condition for this odor
        per_vals = [float(delta_centered.loc[cond, odor_key]) for cond in conditions]
        ax_p.bar(np.arange(len(conditions)), per_vals, color="#F58518")
        ax_p.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        ax_p.set_xticks(np.arange(len(conditions)))
        ax_p.set_xticklabels(conditions, rotation=45, ha="right")
        ax_p.set_ylabel("ΔPER (centered)")
        ax_p.set_title("ΔPER centered for {0}".format(odor_key))
        for i, val in enumerate(per_vals):
            ax_p.text(
                i,
                val,
                "{0:.3f}".format(val),
                ha="center",
                va="bottom" if val >= 0 else "top",
                fontsize=7,
                rotation=0,
            )

        fig.tight_layout()
        plot_path = out_path / "weights_delta_per_{0}.png".format(
            sanitize_filename_token(odor_colmap[odor_key])
        )
        fig.savefig(plot_path, dpi=200)
        plt.close(fig)

    return [str(p) for p in out_path.glob("weights_delta_per_*.png")]


def main(argv=None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    odor_keys = parse_odor_argument(args.odors)
    alpha_grid = _parse_alpha_grid(args.alpha_grid)

    df, odor_colmap, _ = load_per_matrix(
        args.csv,
        odors=odor_keys,
        control_row=args.control_row,
    )
    resolved_control_row = _resolve_row_label_local(df.index, args.control_row)
    allowlist = _parse_condition_list(args.conditions)
    blocklist = _parse_condition_list(args.exclude_conditions)
    conditions = filter_training_conditions(
        df,
        control_row=args.control_row,
        allowlist=allowlist if allowlist else None,
        blocklist=blocklist if blocklist else None,
    )

    _print_schema_evidence(
        df,
        odor_colmap,
        args.control_row,
        resolved_control_row,
        conditions,
    )
    print("Fitting conditions:", conditions)

    # Restrict to control + inferred trained rows only.
    keep_rows = [resolved_control_row] + [cond for cond in conditions if cond != resolved_control_row]
    df4 = extract_four_odors(df.loc[keep_rows], odor_colmap)

    delta_raw = compute_delta(df4, control_row=resolved_control_row, trained_conditions=conditions)
    delta_centered = mean_center_rows(delta_raw)
    y_for_fit = delta_centered if args.center_delta else delta_raw

    _print_delta_repro(
        df4,
        delta_raw,
        delta_centered if args.center_delta else delta_raw,
        control_row=resolved_control_row,
    )

    feature_builder = _build_feature_builder(
        door_cache=args.door_cache,
        mapping_csv=args.mapping_csv,
        feature_set=args.feature_set,
        activation_threshold=args.activation_threshold,
        agg=args.agg,
    )

    fit_results = fit_delta_weights_per_condition(
        odors=list(df4.columns),
        y_centered=y_for_fit,
        feature_builder=feature_builder,
        model=args.model,
        alpha_grid=alpha_grid,
        l1_ratio=args.l1_ratio,
        standardize=args.standardize,
        random_state=args.random_state,
        zero_eps=args.zero_eps,
        min_nonzero=args.min_nonzero,
    )

    exported = export_reports(
        args.outdir,
        df4=df4,
        control_row=resolved_control_row,
        odor_colmap=odor_colmap,
        delta_raw=delta_raw,
        delta_centered=delta_centered if args.center_delta else delta_raw,
        fit_results=fit_results,
        zero_eps=args.zero_eps,
    )

    baseline_df = None
    if args.baseline_outdir:
        baseline_df, baseline_summary = load_baseline_weights(args.baseline_outdir)
        validate_feature_alignment(baseline_df["feature"], fit_results["feature_names"])
        comparison_exports = export_baseline_comparison_reports(
            args.outdir,
            baseline_df=baseline_df,
            df4=df4,
            delta_raw=delta_raw,
            delta_centered=delta_centered if args.center_delta else delta_raw,
            fit_results=fit_results,
            control_row=resolved_control_row,
            centered=args.center_delta,
            zero_eps=args.zero_eps,
        )
        exported.update(comparison_exports)
        print("Baseline model:", baseline_summary.get("model", "unknown"))

    print("=== OUTPUT SUMMARY ===")
    print("outdir:", args.outdir)
    print("training conditions fit:", list(fit_results["condition_results"].keys()))
    print("files:")
    for key, paths in exported.items():
        print("  {0}: {1}".format(key, len(paths)))

    benz_rows = [cond for cond in fit_results["condition_results"].keys() if "benz" in cond.lower()]
    hex_odor_key = "1-hexanol"
    if benz_rows:
        benz_slug = sanitize_filename_token(benz_rows[0])
        hex_slug = sanitize_filename_token(odor_colmap.get(hex_odor_key, hex_odor_key))
        expected = Path(args.outdir) / "contrib_{0}_{1}.csv".format(benz_slug, hex_slug)
        print("benz->hex contribution file:", expected)
        print("benz->hex exists:", expected.exists())

    if args.plot:
        plot_outdir = args.plot_outdir or str(Path(args.outdir) / "plots")
        plots = _plot_contributions(
            plot_outdir,
            base_outdir=args.outdir,
            odor_colmap=odor_colmap,
            conditions=conditions,
            plot_top_n=args.plot_top_n,
        )
        print("plots written:", len(plots))

    if args.plot_weight_delta:
        if baseline_df is None:
            raise ValueError("--plot-weight-delta requires --baseline-outdir")
        weight_plot_outdir = str(Path(args.outdir) / "plots_weights")
        weight_top_n = args.plot_weight_top_n if args.plot_weight_top_n is not None else args.min_nonzero
        weight_plots = _plot_weight_deltas(
            weight_plot_outdir,
            odor_colmap=odor_colmap,
            conditions=conditions,
            baseline_df=baseline_df,
            fit_results=fit_results,
            delta_centered=delta_centered if args.center_delta else delta_raw,
            top_n=weight_top_n,
            zero_eps=args.zero_eps,
        )
        print("weight+ΔPER plots written:", len(weight_plots))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
