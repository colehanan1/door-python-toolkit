#!/usr/bin/env python3
"""
Run a full LASSO sweep: baseline, ablations, and focus mode per condition.

This script writes outputs into per-condition folders and produces
data-only summaries comparing each run to the baseline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Add src to path for repo-local runs (matches other scripts in this repo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.pathways.behavioral_prediction import (
    LassoBehavioralPredictor,
    apply_receptor_ablation,
    fit_lasso_with_fixed_scaler,
    get_top_receptors_by_weight,
    resolve_receptor_names,
    restrict_to_receptors,
)

logger = logging.getLogger(__name__)


@dataclass
class RunMetrics:
    run_type: str
    label: str
    cv_r2: float
    cv_mse: float
    lambda_value: float
    n_receptors_selected: int
    n_samples: int


def _parse_conditions(values: List[str]) -> List[str]:
    conditions: List[str] = []
    seen = set()
    for value in values:
        for token in value.split(","):
            token = token.strip()
            if not token or token in seen:
                continue
            conditions.append(token)
            seen.add(token)
    return conditions


def _parse_lambda_range(value: Optional[str]) -> np.ndarray:
    if value is None:
        return np.logspace(-4, 0, 50)
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("lambda_range cannot be empty.")
    return np.array([float(token) for token in tokens], dtype=np.float64)


def _parse_topn_list(value: str) -> List[int]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    return [int(token) for token in tokens]


def _parse_topn_list_with_all(value: str) -> List[Optional[int]]:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    parsed: List[Optional[int]] = []
    for token in tokens:
        if token.lower() == "all":
            parsed.append(None)
        else:
            parsed.append(int(token))
    return parsed


def _load_receptors_from_file(filepath: str) -> List[str]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Receptor file not found: {filepath}")
    receptors: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                receptors.append(line)
    return receptors


def _build_valid_odorants(
    predictor: LassoBehavioralPredictor,
    condition: str,
    subtract_control: bool,
    control_condition: Optional[str],
    missing_control_policy: str,
) -> Tuple[pd.Series, str, Optional[str], int]:
    resolved_condition = predictor._resolve_dataset_name(condition)
    if resolved_condition is None:
        raise ValueError(f"Condition '{condition}' not found in behavioral data")

    if not subtract_control:
        per_responses = predictor.behavioral_data.loc[resolved_condition]
        valid_odorants = per_responses.dropna()
        if len(valid_odorants) == 0:
            raise ValueError(f"No valid PER data for condition '{resolved_condition}'")
        return valid_odorants, resolved_condition, None, 0

    if missing_control_policy not in {"skip", "zero", "error"}:
        raise ValueError(
            f"Unknown missing_control_policy '{missing_control_policy}'. "
            "Expected one of: skip, zero, error."
        )

    if control_condition is not None:
        control_resolved = predictor._resolve_dataset_name(control_condition)
        if control_resolved is None:
            raise ValueError(f"Control condition '{control_condition}' not found in behavioral data")
    else:
        control_candidate = predictor._infer_control_condition(resolved_condition)
        if control_candidate is None:
            raise ValueError(
                f"No matched control mapping for '{resolved_condition}'. "
                "Provide --control_condition or disable --subtract_control."
            )
        control_resolved = predictor._resolve_dataset_name(control_candidate)
        if control_resolved is None:
            raise ValueError(f"Matched control '{control_candidate}' not found in behavioral data")

    if control_resolved == resolved_condition:
        raise ValueError(
            f"Control condition '{control_resolved}' matches opto condition '{resolved_condition}'."
        )

    per_opto = predictor.behavioral_data.loc[resolved_condition]
    per_ctrl = predictor.behavioral_data.loc[control_resolved]

    if missing_control_policy == "skip":
        valid_mask = per_opto.notna() & per_ctrl.notna()
        valid_odorants = (per_opto - per_ctrl)[valid_mask]
    elif missing_control_policy == "zero":
        valid_mask = per_opto.notna()
        per_ctrl_filled = per_ctrl.fillna(0.0)
        valid_odorants = (per_opto - per_ctrl_filled)[valid_mask]
    else:
        missing_mask = per_opto.notna() & per_ctrl.isna()
        if missing_mask.any():
            missing_odorants = [str(o) for o in per_opto.index[missing_mask]]
            preview = ", ".join(missing_odorants[:5])
            if len(missing_odorants) > 5:
                preview = f"{preview} (and {len(missing_odorants) - 5} more)"
            raise ValueError(
                f"Control condition '{control_resolved}' has missing values for odorants "
                f"present in '{resolved_condition}': {preview}"
            )
        valid_mask = per_opto.notna() & per_ctrl.notna()
        valid_odorants = (per_opto - per_ctrl)[valid_mask]

    if len(valid_odorants) == 0:
        raise ValueError(
            f"No valid opto/control pairs for '{resolved_condition}' and '{control_resolved}'."
        )

    return valid_odorants, resolved_condition, control_resolved, int(len(valid_odorants))


def _extract_features(
    predictor: LassoBehavioralPredictor,
    valid_odorants: pd.Series,
    condition: str,
    prediction_mode: str,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if prediction_mode == "test_odorant":
        return predictor._extract_test_odorant_features(valid_odorants)
    if prediction_mode == "trained_odorant":
        trained_odorant = predictor.CONDITION_ODORANT_MAPPING.get(condition)
        if not trained_odorant:
            raise ValueError(f"Could not determine trained odorant for {condition}")
        return predictor._extract_trained_odorant_features(trained_odorant, valid_odorants)
    if prediction_mode == "interaction":
        trained_odorant = predictor.CONDITION_ODORANT_MAPPING.get(condition)
        if not trained_odorant:
            raise ValueError(f"Could not determine trained odorant for {condition}")
        return predictor._extract_interaction_features(trained_odorant, valid_odorants)
    raise ValueError(f"Unknown prediction_mode: {prediction_mode}")


def _save_weights_csv(
    weights: Dict[str, float], filepath: Path, condition: str, label: str
) -> None:
    rows = []
    for receptor, weight in sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True):
        rows.append(
            {
                "condition": condition,
                "run_label": label,
                "receptor": receptor,
                "weight": weight,
                "abs_weight": abs(weight),
            }
        )
    df = pd.DataFrame(rows)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def _save_metrics_json(metrics: RunMetrics, filepath: Path, extra: Optional[Dict] = None) -> None:
    data = {
        "run_type": metrics.run_type,
        "label": metrics.label,
        "cv_r2": metrics.cv_r2,
        "cv_mse": metrics.cv_mse,
        "lambda_value": metrics.lambda_value,
        "n_receptors_selected": metrics.n_receptors_selected,
        "n_samples": metrics.n_samples,
    }
    if extra:
        data.update(extra)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _delta_label(delta: float, tol: float = 1e-12) -> str:
    if delta > tol:
        return "higher"
    if delta < -tol:
        return "lower"
    return "same"


def _summary_markdown(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline + ablation + focus LASSO sweeps with summaries.",
    )
    parser.add_argument("--door_cache", required=True, help="Path to DoOR cache directory.")
    parser.add_argument("--behavior_csv", required=True, help="Path to behavioral matrix CSV.")
    parser.add_argument(
        "--condition",
        required=True,
        action="append",
        help="Condition name(s). Repeat flag or pass comma-separated list.",
    )
    parser.add_argument("--output_dir", required=True, help="Directory to write outputs.")

    parser.add_argument(
        "--prediction_mode",
        choices=["test_odorant", "trained_odorant", "interaction"],
        default="test_odorant",
        help="Feature extraction mode.",
    )
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds.")
    parser.add_argument(
        "--lambda_range",
        default="0.0001,0.001,0.01,0.1,1.0",
        help="Comma-separated lambda values (e.g., 0.0001,0.001,0.01).",
    )
    parser.add_argument(
        "--scale_features",
        action="store_true",
        default=True,
        help="Standardize receptor features (default: True).",
    )
    parser.add_argument(
        "--no_scale_features",
        action="store_false",
        dest="scale_features",
        help="Do not standardize receptor features.",
    )

    parser.add_argument("--subtract_control", action="store_true", help="Fit ΔPER = opto - control.")
    parser.add_argument(
        "--control_condition",
        default=None,
        help="Optional control dataset override (applies to all conditions).",
    )
    parser.add_argument(
        "--missing_control_policy",
        choices=["skip", "zero", "error"],
        default="error",
        help="How to handle missing control values.",
    )

    parser.add_argument(
        "--ablate",
        type=str,
        default=None,
        help="Comma-separated list of receptors for specific ablation.",
    )
    parser.add_argument(
        "--ablate_file",
        type=str,
        default=None,
        help="File with receptor names (one per line) for specific ablation.",
    )
    parser.add_argument(
        "--specific_ablation_mode",
        choices=["single", "all_in_one"],
        default="all_in_one",
        help="Ablation mode for --ablate/--ablate_file (default: all_in_one).",
    )
    parser.add_argument(
        "--top_k_max",
        type=int,
        default=0,
        help="Max number of top receptors to consider (0 = all).",
    )
    parser.add_argument(
        "--top_k_group_list",
        type=str,
        default="2,3",
        help="Comma-separated list for cumulative top-K ablations (default: 2,3).",
    )
    parser.add_argument(
        "--all_but_top_list",
        type=str,
        default="2,3,all",
        help="Comma-separated list for ablate-all-but-top-K (default: 2,3,all).",
    )
    parser.add_argument(
        "--no_top_single",
        action="store_true",
        help="Disable single ablations for each top receptor.",
    )
    parser.add_argument(
        "--no_ablate_all_top",
        action="store_true",
        help="Disable ablation of all top receptors together.",
    )
    parser.add_argument(
        "--no_ablate_pos_neg",
        action="store_true",
        help="Disable ablation of positive/negative top sets.",
    )
    parser.add_argument(
        "--no_all_but_top",
        action="store_true",
        help="Disable ablate-all-but-top runs.",
    )
    parser.add_argument(
        "--focus_topn_list",
        type=str,
        default="1,2,3",
        help="Comma-separated list of top-N values for focus runs (default: 1,2,3).",
    )
    parser.add_argument(
        "--focus_receptors",
        type=str,
        default=None,
        help="Comma-separated list of receptors to focus on (overrides --focus_topn_list).",
    )
    parser.add_argument(
        "--missing_receptor_policy",
        choices=["error", "skip"],
        default="error",
        help="Policy for unresolved receptor names in ablation/focus.",
    )

    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    conditions = _parse_conditions(args.condition)
    if not conditions:
        logger.error("No valid conditions provided.")
        return 1

    lambda_range = _parse_lambda_range(args.lambda_range)
    focus_topn_list = _parse_topn_list(args.focus_topn_list)
    top_k_group_list = _parse_topn_list(args.top_k_group_list)
    all_but_top_list = _parse_topn_list_with_all(args.all_but_top_list)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictor = LassoBehavioralPredictor(
        doorcache_path=args.door_cache,
        behavior_csv_path=args.behavior_csv,
        scale_features=False,
        scale_targets=False,
    )

    available_receptors = list(predictor.masked_receptor_names)
    strict_receptors = args.missing_receptor_policy == "error"

    all_summary_rows: List[Dict] = []
    exit_code = 0

    for condition in conditions:
        try:
            resolved_condition = predictor._resolve_dataset_name(condition)
        except ValueError as exc:
            logger.error(str(exc))
            exit_code = 1
            continue

        if resolved_condition is None:
            logger.error(
                "Condition '%s' not found. Available: %s",
                condition,
                list(predictor.behavioral_data.index),
            )
            exit_code = 1
            continue

        if (
            args.subtract_control
            and args.missing_control_policy == "skip"
            and args.control_condition is None
        ):
            control_candidate = predictor._infer_control_condition(resolved_condition)
            if control_candidate is None:
                logger.warning(
                    "No matched control mapping for '%s'; skipping ΔPER run "
                    "(missing_control_policy=skip).",
                    resolved_condition,
                )
                continue
            control_resolved = predictor._resolve_dataset_name(control_candidate)
            if control_resolved is None:
                logger.warning(
                    "Matched control '%s' not found for '%s'; skipping ΔPER run "
                    "(missing_control_policy=skip).",
                    control_candidate,
                    resolved_condition,
                )
                continue

        try:
            valid_odorants, _, control_resolved, n_pairs_used = _build_valid_odorants(
                predictor,
                condition=resolved_condition,
                subtract_control=args.subtract_control,
                control_condition=args.control_condition,
                missing_control_policy=args.missing_control_policy,
            )
        except Exception as exc:
            logger.error("Condition '%s' failed: %s", resolved_condition, exc)
            exit_code = 1
            continue

        try:
            X, test_odorants, y = _extract_features(
                predictor,
                valid_odorants,
                resolved_condition,
                args.prediction_mode,
            )
        except Exception as exc:
            logger.error("Feature extraction failed for %s: %s", resolved_condition, exc)
            exit_code = 1
            continue

        if X.shape[0] < 3:
            logger.error("Insufficient samples for %s: %d", resolved_condition, X.shape[0])
            exit_code = 1
            continue

        condition_dir = output_dir / resolved_condition
        condition_dir.mkdir(parents=True, exist_ok=True)

        baseline_scaler = StandardScaler().fit(X) if args.scale_features else None
        baseline_weights, baseline_r2, baseline_mse, baseline_lambda, baseline_pred = (
            fit_lasso_with_fixed_scaler(
                X=X,
                y=y,
                receptor_names=available_receptors,
                scaler=baseline_scaler,
                lambda_range=lambda_range,
                cv_folds=args.cv_folds,
            )
        )

        baseline_metrics = RunMetrics(
            run_type="baseline",
            label="baseline",
            cv_r2=baseline_r2,
            cv_mse=baseline_mse,
            lambda_value=baseline_lambda,
            n_receptors_selected=len(baseline_weights),
            n_samples=X.shape[0],
        )

        _save_weights_csv(
            baseline_weights,
            condition_dir / "baseline_weights.csv",
            resolved_condition,
            "baseline",
        )
        _save_metrics_json(
            baseline_metrics,
            condition_dir / "baseline_metrics.json",
            extra={
                "prediction_mode": args.prediction_mode,
                "subtract_control": args.subtract_control,
                "control_condition": control_resolved,
                "missing_control_policy": args.missing_control_policy,
                "n_pairs_used": int(n_pairs_used),
                "n_receptors_total": X.shape[1],
            },
        )

        top_receptors_all = get_top_receptors_by_weight(
            baseline_weights, len(baseline_weights)
        )
        if args.top_k_max and args.top_k_max > 0:
            top_receptors = top_receptors_all[: args.top_k_max]
        else:
            top_receptors = top_receptors_all

        summary_rows: List[Dict] = []

        def add_summary(metrics: RunMetrics, extra: Optional[Dict] = None) -> None:
            delta_r2 = metrics.cv_r2 - baseline_metrics.cv_r2
            delta_mse = metrics.cv_mse - baseline_metrics.cv_mse
            row = {
                "condition": resolved_condition,
                "run_type": metrics.run_type,
                "label": metrics.label,
                "cv_r2": metrics.cv_r2,
                "cv_mse": metrics.cv_mse,
                "lambda_value": metrics.lambda_value,
                "n_receptors_selected": metrics.n_receptors_selected,
                "n_samples": metrics.n_samples,
                "n_total_receptors": X.shape[1],
                "delta_r2": delta_r2,
                "delta_mse": delta_mse,
                "r2_vs_baseline": _delta_label(delta_r2),
                "mse_vs_baseline": _delta_label(delta_mse),
            }
            if extra:
                row.update(extra)
            summary_rows.append(row)

        add_summary(
            baseline_metrics,
            extra={
                "n_ablated": 0,
                "n_kept": X.shape[1],
            },
        )

        ablation_labels_seen: set[str] = set()

        def run_ablation_set(
            *,
            run_type: str,
            label: str,
            receptors_to_ablate: List[str],
        ) -> None:
            if not receptors_to_ablate:
                return
            if len(label) > 120:
                label = f"{label[:80]}_{len(receptors_to_ablate)}"
            if label in ablation_labels_seen:
                return
            ablation_labels_seen.add(label)

            X_ablated, _ = apply_receptor_ablation(
                X=X,
                receptor_names=available_receptors,
                receptors_to_ablate=receptors_to_ablate,
            )
            weights, cv_r2, cv_mse, best_lambda, _pred = fit_lasso_with_fixed_scaler(
                X=X_ablated,
                y=y,
                receptor_names=available_receptors,
                scaler=baseline_scaler,
                lambda_range=lambda_range,
                cv_folds=args.cv_folds,
            )
            metrics = RunMetrics(
                run_type=run_type,
                label=label,
                cv_r2=cv_r2,
                cv_mse=cv_mse,
                lambda_value=best_lambda,
                n_receptors_selected=len(weights),
                n_samples=X_ablated.shape[0],
            )
            ablation_dir = condition_dir / "ablations" / label
            _save_weights_csv(
                weights,
                ablation_dir / "weights.csv",
                resolved_condition,
                label,
            )
            _save_metrics_json(
                metrics,
                ablation_dir / "metrics.json",
                extra={
                    "receptors_ablated": receptors_to_ablate,
                    "n_ablated": len(receptors_to_ablate),
                },
            )
            add_summary(
                metrics,
                extra={
                    "n_ablated": len(receptors_to_ablate),
                    "n_kept": X.shape[1] - len(receptors_to_ablate),
                },
            )

        # Specific ablations (optional)
        receptors_to_ablate: List[str] = []
        if args.ablate:
            receptors_to_ablate = [r.strip() for r in args.ablate.split(",") if r.strip()]
        elif args.ablate_file:
            receptors_to_ablate = _load_receptors_from_file(args.ablate_file)

        if receptors_to_ablate:
            matched, unmatched = resolve_receptor_names(
                receptors_to_ablate, available_receptors, strict=strict_receptors
            )
            if unmatched and not strict_receptors:
                logger.warning("Skipping unmatched receptors: %s", unmatched)

            if matched:
                if args.specific_ablation_mode == "single":
                    for receptor in matched:
                        run_ablation_set(
                            run_type="ablation_specific",
                            label=f"ablate_specific_{receptor}",
                            receptors_to_ablate=[receptor],
                        )
                else:
                    run_ablation_set(
                        run_type="ablation_specific",
                        label="ablate_specific_all",
                        receptors_to_ablate=matched,
                    )

        # Auto ablations from baseline top receptors
        if not top_receptors:
            logger.warning("No baseline weights for %s; skipping auto ablations", resolved_condition)
        else:
            if not args.no_top_single:
                for receptor in top_receptors:
                    run_ablation_set(
                        run_type="ablation_top_single",
                        label=f"ablate_top_single_{receptor}",
                        receptors_to_ablate=[receptor],
                    )

            if top_k_group_list:
                for k in top_k_group_list:
                    if k <= 0:
                        logger.warning("Invalid top-K value %d; skipping", k)
                        continue
                    if k > len(top_receptors):
                        logger.warning(
                            "Top-K value %d exceeds available top receptors (%d); skipping",
                            k,
                            len(top_receptors),
                        )
                        continue
                    run_ablation_set(
                        run_type="ablation_top_group",
                        label=f"ablate_top_{k}",
                        receptors_to_ablate=top_receptors[:k],
                    )

            if not args.no_ablate_all_top:
                run_ablation_set(
                    run_type="ablation_top_all",
                    label="ablate_top_all",
                    receptors_to_ablate=top_receptors,
                )

            if not args.no_ablate_pos_neg:
                pos_receptors = [r for r in top_receptors if baseline_weights.get(r, 0.0) > 0]
                neg_receptors = [r for r in top_receptors if baseline_weights.get(r, 0.0) < 0]
                run_ablation_set(
                    run_type="ablation_top_positive",
                    label="ablate_top_positive",
                    receptors_to_ablate=pos_receptors,
                )
                run_ablation_set(
                    run_type="ablation_top_negative",
                    label="ablate_top_negative",
                    receptors_to_ablate=neg_receptors,
                )

            if not args.no_all_but_top:
                for keep_k in all_but_top_list:
                    if keep_k is None:
                        keep_list = top_receptors
                        label = "ablate_all_but_top_all"
                    else:
                        if keep_k <= 0:
                            logger.warning("Invalid keep-K value %d; skipping", keep_k)
                            continue
                        if keep_k > len(top_receptors):
                            logger.warning(
                                "Keep-K value %d exceeds available top receptors (%d); skipping",
                                keep_k,
                                len(top_receptors),
                            )
                            continue
                        keep_list = top_receptors[:keep_k]
                        label = f"ablate_all_but_top_{keep_k}"

                    keep_set = set(keep_list)
                    ablate_list = [r for r in available_receptors if r not in keep_set]
                    run_ablation_set(
                        run_type="ablation_all_but_top",
                        label=label,
                        receptors_to_ablate=ablate_list,
                    )

        # Focus runs
        focus_receptors: Optional[List[str]] = None
        if args.focus_receptors:
            focus_receptors = [r.strip() for r in args.focus_receptors.split(",") if r.strip()]

        if focus_receptors:
            matched, unmatched = resolve_receptor_names(
                focus_receptors, available_receptors, strict=strict_receptors
            )
            if unmatched and not strict_receptors:
                logger.warning("Skipping unmatched receptors: %s", unmatched)
            if matched:
                X_restricted, kept_names, _ = restrict_to_receptors(
                    X=X,
                    receptor_names=available_receptors,
                    receptors_to_keep=matched,
                )
                focus_scaler = StandardScaler().fit(X_restricted) if args.scale_features else None
                weights, cv_r2, cv_mse, best_lambda, _pred = fit_lasso_with_fixed_scaler(
                    X=X_restricted,
                    y=y,
                    receptor_names=kept_names,
                    scaler=focus_scaler,
                    lambda_range=lambda_range,
                    cv_folds=args.cv_folds,
                )
                label = f"focus_{len(kept_names)}"
                metrics = RunMetrics(
                    run_type="focus_specific",
                    label=label,
                    cv_r2=cv_r2,
                    cv_mse=cv_mse,
                    lambda_value=best_lambda,
                    n_receptors_selected=len(weights),
                    n_samples=X_restricted.shape[0],
                )
                focus_dir = condition_dir / "focus" / label
                _save_weights_csv(weights, focus_dir / "weights.csv", resolved_condition, label)
                _save_metrics_json(
                    metrics,
                    focus_dir / "metrics.json",
                    extra={
                        "receptors_kept": kept_names,
                        "n_kept": len(kept_names),
                    },
                )
                add_summary(
                    metrics,
                    extra={
                        "n_ablated": X.shape[1] - len(kept_names),
                        "n_kept": len(kept_names),
                    },
                )
        else:
            if not top_receptors_all:
                logger.warning("No baseline weights for %s; skipping focus runs", resolved_condition)
            else:
                for n in focus_topn_list:
                    if n <= 0:
                        logger.warning("Invalid top-N value %d; skipping", n)
                        continue
                    if n > len(top_receptors_all):
                        logger.warning(
                            "Top-N value %d exceeds available receptors (%d); skipping",
                            n,
                            len(top_receptors_all),
                        )
                        continue
                    top_n = top_receptors_all[:n]
                    X_restricted, kept_names, _ = restrict_to_receptors(
                        X=X,
                        receptor_names=available_receptors,
                        receptors_to_keep=top_n,
                    )
                    focus_scaler = (
                        StandardScaler().fit(X_restricted) if args.scale_features else None
                    )
                    weights, cv_r2, cv_mse, best_lambda, _pred = fit_lasso_with_fixed_scaler(
                        X=X_restricted,
                        y=y,
                        receptor_names=kept_names,
                        scaler=focus_scaler,
                        lambda_range=lambda_range,
                        cv_folds=args.cv_folds,
                    )
                    label = f"focus_top_{n}"
                    metrics = RunMetrics(
                        run_type="focus_topn",
                        label=label,
                        cv_r2=cv_r2,
                        cv_mse=cv_mse,
                        lambda_value=best_lambda,
                        n_receptors_selected=len(weights),
                        n_samples=X_restricted.shape[0],
                    )
                    focus_dir = condition_dir / "focus" / label
                    _save_weights_csv(weights, focus_dir / "weights.csv", resolved_condition, label)
                    _save_metrics_json(
                        metrics,
                        focus_dir / "metrics.json",
                        extra={
                            "receptors_kept": kept_names,
                            "n_kept": len(kept_names),
                        },
                    )
                    add_summary(
                        metrics,
                        extra={
                            "n_ablated": X.shape[1] - len(kept_names),
                            "n_kept": len(kept_names),
                        },
                    )

        summary_df = pd.DataFrame(summary_rows)
        summary_csv = condition_dir / "summary.csv"
        summary_df.to_csv(summary_csv, index=False)

        summary_md = condition_dir / "summary.md"
        summary_md.write_text(_summary_markdown(summary_df), encoding="utf-8")

        all_summary_rows.extend(summary_rows)

    if all_summary_rows:
        all_summary_df = pd.DataFrame(all_summary_rows)
        all_summary_df.to_csv(output_dir / "summary_all_conditions.csv", index=False)
        (output_dir / "summary_all_conditions.md").write_text(
            _summary_markdown(all_summary_df), encoding="utf-8"
        )

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
