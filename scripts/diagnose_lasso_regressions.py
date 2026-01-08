#!/usr/bin/env python3
"""
Diagnose LASSO behavioral prediction regressions and ΔPER stability.

This script runs repeated raw and control-subtracted (ΔPER) fits to detect
non-determinism, constant-prediction collapse, and feature drift.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add src to path for repo-local runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.pathways import LassoBehavioralPredictor

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    condition: str
    mode: str
    status: str
    cv_mse: Optional[float] = None
    cv_r2: Optional[float] = None
    lambda_value: Optional[float] = None
    n_receptors_selected: Optional[int] = None
    control_condition: Optional[str] = None
    error: Optional[str] = None


def _parse_conditions(value: str) -> List[str]:
    conditions = [token.strip() for token in value.split(",") if token.strip()]
    if not conditions:
        raise ValueError("No conditions provided.")
    return conditions


def _parse_lambda_range(value: str) -> List[float]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("lambda_range cannot be empty.")
    try:
        return [float(token) for token in tokens]
    except ValueError as exc:
        raise ValueError(f"Invalid lambda_range: {value}") from exc


def _compute_stats(vec: np.ndarray) -> Dict[str, float]:
    vec = np.asarray(vec, dtype=np.float64)
    return {
        "n": int(vec.size),
        "mean": float(np.nanmean(vec)) if vec.size else float("nan"),
        "std": float(np.nanstd(vec)) if vec.size else float("nan"),
        "min": float(np.nanmin(vec)) if vec.size else float("nan"),
        "max": float(np.nanmax(vec)) if vec.size else float("nan"),
    }


def _compute_x_stats(X: np.ndarray) -> Dict[str, float]:
    X = np.asarray(X, dtype=np.float64)
    col_std = np.nanstd(X, axis=0) if X.size else np.array([])
    zero_cols = int(np.sum(col_std <= 1e-12)) if col_std.size else 0
    return {
        "n_samples": int(X.shape[0]) if X.ndim == 2 else 0,
        "n_features": int(X.shape[1]) if X.ndim == 2 else 0,
        "col_std_min": float(np.nanmin(col_std)) if col_std.size else float("nan"),
        "col_std_median": float(np.nanmedian(col_std)) if col_std.size else float("nan"),
        "col_std_max": float(np.nanmax(col_std)) if col_std.size else float("nan"),
        "all_zero_columns": zero_cols,
    }


def _weights_table(weights: Dict[str, float]) -> List[Dict[str, float]]:
    rows = [
        {"receptor": k, "weight": float(v), "abs_weight": abs(float(v))}
        for k, v in weights.items()
    ]
    rows.sort(key=lambda x: x["abs_weight"], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def _hash_weights(weights: Dict[str, float], order: List[str]) -> np.ndarray:
    return np.array([weights.get(r, 0.0) for r in order], dtype=np.float64)


def run_mode(
    *,
    condition: str,
    mode: str,
    door_cache: str,
    behavior_csv: str,
    prediction_mode: str,
    cv_folds: int,
    lambda_range: List[float],
    seed: int,
    subtract_control: bool,
    control_condition: Optional[str],
    missing_control_policy: str,
) -> Tuple[Optional[RunResult], Optional[Dict]]:
    np.random.seed(seed)
    predictor = LassoBehavioralPredictor(
        doorcache_path=door_cache,
        behavior_csv_path=behavior_csv,
    )

    try:
        results = predictor.fit_behavior(
            condition_name=condition,
            prediction_mode=prediction_mode,
            cv_folds=cv_folds,
            lambda_range=lambda_range,
            subtract_control=subtract_control,
            control_condition=control_condition,
            missing_control_policy=missing_control_policy,
        )
    except Exception as exc:
        return (
            RunResult(
                condition=condition,
                mode=mode,
                status="error",
                error=str(exc),
            ),
            None,
        )

    X = results.feature_matrix if results.feature_matrix is not None else np.array([])
    y = results.actual_per
    y_pred = results.predicted_per

    y_stats = _compute_stats(y)
    x_stats = _compute_x_stats(X)
    pred_stats = _compute_stats(y_pred)
    pred_collapse = pred_stats["std"] < 1e-6 if not np.isnan(pred_stats["std"]) else False

    details = {
        "condition": condition,
        "mode": mode,
        "subtract_control": bool(subtract_control),
        "control_condition": results.control_condition,
        "n_pairs_used": int(results.n_pairs_used),
        "y_stats": y_stats,
        "x_stats": x_stats,
        "pred_stats": pred_stats,
        "pred_collapse": pred_collapse,
        "lambda_value": float(results.lambda_value),
        "cv_r2": float(results.cv_r2_score),
        "cv_mse": float(results.cv_mse),
        "n_receptors_selected": int(results.n_receptors_selected),
        "receptor_names": list(results.receptor_names),
        "weights": results.lasso_weights,
    }

    run_result = RunResult(
        condition=condition,
        mode=mode,
        status="ok",
        cv_mse=results.cv_mse,
        cv_r2=results.cv_r2_score,
        lambda_value=results.lambda_value,
        n_receptors_selected=results.n_receptors_selected,
        control_condition=results.control_condition,
    )

    return run_result, details


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose LASSO regressions for raw vs ΔPER fits.",
    )
    parser.add_argument("--door_cache", required=True, help="Path to DoOR cache directory.")
    parser.add_argument("--behavior_csv", required=True, help="Path to behavioral matrix CSV.")
    parser.add_argument(
        "--conditions",
        required=True,
        help="Comma-separated conditions (e.g., opto_hex,opto_EB).",
    )
    parser.add_argument(
        "--prediction_mode",
        choices=["test_odorant", "trained_odorant", "interaction"],
        default="test_odorant",
    )
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument(
        "--lambda_range",
        required=True,
        help="Comma-separated lambda values.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--subtract_control", action="store_true")
    parser.add_argument("--control_condition", default=None)
    parser.add_argument(
        "--missing_control_policy",
        choices=["skip", "zero", "error"],
        default="skip",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory (default: diagnostics/run_<timestamp>).",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()

    conditions = _parse_conditions(args.conditions)
    lambda_range = _parse_lambda_range(args.lambda_range)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) if args.output_dir else Path("diagnostics") / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)

    if not args.subtract_control:
        logger.warning(
            "--subtract_control not set; diagnostic will still run ΔPER modes for comparison."
        )

    modes = [
        ("baseline_raw", False),
        ("baseline_delta", True),
        ("baseline_raw_repeat", False),
        ("baseline_delta_repeat", True),
    ]

    per_condition_metrics: List[Dict] = []
    per_condition_top_coefs: List[Dict] = []
    per_condition_stats: Dict[str, Dict] = {}
    collapse_rows: List[Dict] = []

    run_details: Dict[Tuple[str, str], Dict] = {}

    for condition in conditions:
        per_condition_stats[condition] = {}
        for mode, subtract_control in modes:
            run_result, details = run_mode(
                condition=condition,
                mode=mode,
                door_cache=args.door_cache,
                behavior_csv=args.behavior_csv,
                prediction_mode=args.prediction_mode,
                cv_folds=args.cv_folds,
                lambda_range=lambda_range,
                seed=args.seed,
                subtract_control=subtract_control,
                control_condition=args.control_condition,
                missing_control_policy=args.missing_control_policy,
            )

            if run_result is None:
                continue

            per_condition_metrics.append(run_result.__dict__)

            if details is None:
                per_condition_stats[condition][mode] = {"status": "error", "error": run_result.error}
                continue

            per_condition_stats[condition][mode] = details
            run_details[(condition, mode)] = details

            for row in _weights_table(details["weights"]):
                per_condition_top_coefs.append(
                    {
                        "condition": condition,
                        "mode": mode,
                        **row,
                    }
                )

            collapse_rows.append(
                {
                    "condition": condition,
                    "mode": mode,
                    "pred_std": details["pred_stats"]["std"],
                    "pred_min": details["pred_stats"]["min"],
                    "pred_max": details["pred_stats"]["max"],
                    "pred_collapse": details["pred_collapse"],
                    "y_std": details["y_stats"]["std"],
                }
            )

    pd.DataFrame(per_condition_metrics).to_csv(output_dir / "per_condition_metrics.csv", index=False)
    pd.DataFrame(per_condition_top_coefs).to_csv(output_dir / "per_condition_top_coefs.csv", index=False)
    pd.DataFrame(collapse_rows).to_csv(output_dir / "collapse_flags.csv", index=False)

    with open(output_dir / "per_condition_yx_stats.json", "w", encoding="utf-8") as f:
        json.dump(per_condition_stats, f, indent=2)

    reproducibility_rows: List[Dict] = []
    for condition in conditions:
        for base_mode, repeat_mode in (
            ("baseline_raw", "baseline_raw_repeat"),
            ("baseline_delta", "baseline_delta_repeat"),
        ):
            base = run_details.get((condition, base_mode))
            rep = run_details.get((condition, repeat_mode))
            if base is None or rep is None:
                reproducibility_rows.append(
                    {
                        "condition": condition,
                        "mode": base_mode,
                        "status": "missing",
                    }
                )
                continue

            receptors = sorted(set(base["receptor_names"]) | set(rep["receptor_names"]))
            base_vec = _hash_weights(base["weights"], receptors)
            rep_vec = _hash_weights(rep["weights"], receptors)
            max_abs_coef_diff = float(np.max(np.abs(base_vec - rep_vec))) if receptors else 0.0

            pred_diff = float(abs(base["pred_stats"]["std"] - rep["pred_stats"]["std"]))
            cv_mse_diff = float(abs(base["cv_mse"] - rep["cv_mse"]))
            cv_r2_diff = float(abs(base["cv_r2"] - rep["cv_r2"]))

            reproducibility_rows.append(
                {
                    "condition": condition,
                    "mode": base_mode,
                    "status": "ok",
                    "lambda_equal": base["lambda_value"] == rep["lambda_value"],
                    "max_abs_coef_diff": max_abs_coef_diff,
                    "cv_mse_diff": cv_mse_diff,
                    "cv_r2_diff": cv_r2_diff,
                    "pred_std_diff": pred_diff,
                }
            )

    pd.DataFrame(reproducibility_rows).to_csv(
        output_dir / "reproducibility_check.csv", index=False
    )

    summary_lines = [
        "# LASSO Regression Diagnostics Summary",
        "",
        f"Run directory: {output_dir}",
        f"Conditions: {', '.join(conditions)}",
        f"Prediction mode: {args.prediction_mode}",
        f"Lambda range: {lambda_range}",
        f"Seed: {args.seed}",
        "",
        "## Reproducibility",
    ]

    repro_df = pd.DataFrame(reproducibility_rows)
    if not repro_df.empty:
        for _, row in repro_df.iterrows():
            if row.get("status") != "ok":
                summary_lines.append(
                    f"- {row['condition']} {row['mode']}: missing run output"
                )
                continue
            summary_lines.append(
                f"- {row['condition']} {row['mode']}: "
                f"lambda_equal={row['lambda_equal']}, "
                f"max_abs_coef_diff={row['max_abs_coef_diff']:.4g}, "
                f"cv_mse_diff={row['cv_mse_diff']:.4g}, "
                f"cv_r2_diff={row['cv_r2_diff']:.4g}, "
                f"pred_std_diff={row['pred_std_diff']:.4g}"
            )

    summary_lines.append("")
    summary_lines.append("## Collapse Flags (pred std < 1e-6)")
    collapse_df = pd.DataFrame(collapse_rows)
    if collapse_df.empty:
        summary_lines.append("- No results")
    else:
        collapsed = collapse_df[collapse_df["pred_collapse"] == True]
        if collapsed.empty:
            summary_lines.append("- No prediction collapses detected")
        else:
            for _, row in collapsed.iterrows():
                summary_lines.append(
                    f"- {row['condition']} {row['mode']}: pred_std={row['pred_std']:.4g}"
                )

    summary_lines.append("")
    summary_lines.append("## Raw vs ΔPER Differences (Top ORNs)")
    for condition in conditions:
        raw = run_details.get((condition, "baseline_raw"))
        delta = run_details.get((condition, "baseline_delta"))
        if raw is None or delta is None:
            summary_lines.append(f"- {condition}: missing raw or delta results")
            continue
        raw_top = _weights_table(raw["weights"])[:5]
        delta_top = _weights_table(delta["weights"])[:5]
        raw_list = ", ".join([f"{r['receptor']}({r['abs_weight']:.3g})" for r in raw_top])
        delta_list = ", ".join([f"{r['receptor']}({r['abs_weight']:.3g})" for r in delta_top])
        summary_lines.append(f"- {condition} raw top: {raw_list or 'none'}")
        summary_lines.append(f"  {condition} delta top: {delta_list or 'none'}")

    summary_lines.append("")
    summary_lines.append("## Notes")
    summary_lines.append(
        "- LassoCV uses random_state=42 in the predictor; cross_val_score uses deterministic folds (no shuffle)."
    )
    summary_lines.append(
        "- ΔPER reduces sample size if control data are missing; check n_pairs_used in per_condition_yx_stats.json."
    )

    (output_dir / "SUMMARY.md").write_text("\n".join(summary_lines), encoding="utf-8")
    logger.info("Wrote diagnostics to %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
