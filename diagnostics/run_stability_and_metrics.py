#!/usr/bin/env python3
"""
Run stability scoring + standardized metrics for LASSO/Ridge/ElasticNet.

Outputs are written under diagnostics/stability_<timestamp>/.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

# Add src to path for repo-local runs (matches other scripts in this repo).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.pathways.behavioral_prediction import (
    LassoBehavioralPredictor,
    fit_lasso_with_fixed_scaler,
)

logger = logging.getLogger(__name__)


@dataclass
class ModelFit:
    modelclass: str
    lambda_value: float
    l1_ratio: Optional[float]
    cv_mse: float
    n_selected: int
    coef: np.ndarray
    y_pred: np.ndarray


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


def _parse_lambda_range(value: Optional[str]) -> Optional[np.ndarray]:
    if value is None:
        return None
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("lambda_range cannot be empty.")
    return np.array([float(token) for token in tokens], dtype=np.float64)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _loocv_intercept_mse(y: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    loo = LeaveOneOut()
    errors = []
    for train_idx, test_idx in loo.split(y):
        train_mean = float(np.mean(y[train_idx]))
        err = float((y[test_idx][0] - train_mean) ** 2)
        errors.append(err)
    return float(np.mean(errors))


def _is_intercept_only(n_selected: int, pred_std: float, cv_mse: float, intercept_only_mse: float) -> bool:
    return (
        n_selected == 0
        and pred_std < 1e-6
        and abs(cv_mse - intercept_only_mse) < 1e-12
    )


def _cv_mse_grid(model_factory, X: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> List[float]:
    loo = LeaveOneOut()
    mse_values: List[float] = []
    for lam in lambdas:
        model = model_factory(lam)
        scores = cross_val_score(model, X, y, cv=loo, scoring="neg_mean_squared_error")
        mse_values.append(float(-np.mean(scores)))
    return mse_values


def _fit_ridge(X: np.ndarray, y: np.ndarray, lambdas: np.ndarray, seed: int) -> ModelFit:
    def factory(alpha: float) -> Ridge:
        return Ridge(alpha=alpha, random_state=seed)

    mse_grid = _cv_mse_grid(factory, X, y, lambdas)
    best_idx = int(np.argmin(mse_grid))
    best_lambda = float(lambdas[best_idx])

    model = factory(best_lambda)
    model.fit(X, y)
    y_pred = model.predict(X)
    coef = model.coef_.copy()
    n_selected = int(np.sum(np.abs(coef) > 1e-6))

    return ModelFit(
        modelclass="ridge",
        lambda_value=best_lambda,
        l1_ratio=None,
        cv_mse=float(mse_grid[best_idx]),
        n_selected=n_selected,
        coef=coef,
        y_pred=y_pred,
    )


def _fit_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    l1_ratios: List[float],
    seed: int,
) -> ModelFit:
    best_lambda = None
    best_l1 = None
    best_mse = None
    best_coef = None
    best_pred = None

    for l1_ratio in l1_ratios:
        def factory(alpha: float) -> ElasticNet:
            return ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                random_state=seed,
                max_iter=10000,
            )

        mse_grid = _cv_mse_grid(factory, X, y, lambdas)
        idx = int(np.argmin(mse_grid))
        mse = float(mse_grid[idx])
        if best_mse is None or mse < best_mse:
            best_mse = mse
            best_lambda = float(lambdas[idx])
            best_l1 = float(l1_ratio)
            model = factory(best_lambda)
            model.fit(X, y)
            best_coef = model.coef_.copy()
            best_pred = model.predict(X)

    if best_lambda is None or best_coef is None or best_pred is None or best_l1 is None:
        raise RuntimeError("ElasticNet fitting failed")

    n_selected = int(np.sum(np.abs(best_coef) > 1e-6))
    return ModelFit(
        modelclass="elasticnet",
        lambda_value=float(best_lambda),
        l1_ratio=float(best_l1),
        cv_mse=float(best_mse),
        n_selected=n_selected,
        coef=best_coef,
        y_pred=best_pred,
    )


def _fit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    receptor_names: List[str],
    lambdas: np.ndarray,
    cv_folds: int,
    scaler: Optional[StandardScaler],
    seed: int,
) -> Tuple[Dict[str, float], float, float, np.ndarray]:
    weights, _cv_r2, cv_mse, best_lambda, y_pred = fit_lasso_with_fixed_scaler(
        X=X,
        y=y,
        receptor_names=receptor_names,
        scaler=scaler,
        lambda_range=lambdas,
        cv_folds=cv_folds,
        random_state=seed,
    )
    return weights, float(cv_mse), float(best_lambda), y_pred


def _collect_metrics(
    *,
    condition: str,
    mode: str,
    model: ModelFit,
    y: np.ndarray,
    intercept_only_mse: float,
) -> Dict[str, float]:
    y_std = float(np.std(y))
    y_var = float(np.var(y))
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    pred_std = float(np.std(model.y_pred))
    pred_min = float(np.min(model.y_pred))
    pred_max = float(np.max(model.y_pred))
    nmse = float(model.cv_mse / (y_var + 1e-12))
    rmse_over_y_std = float(math.sqrt(model.cv_mse) / (y_std + 1e-12))
    intercept_only_flag = _is_intercept_only(
        model.n_selected, pred_std, model.cv_mse, intercept_only_mse
    )

    return {
        "condition": condition,
        "mode": mode,
        "modelclass": model.modelclass,
        "lambda": model.lambda_value,
        "l1_ratio": model.l1_ratio,
        "n_selected": model.n_selected,
        "cv_mse": model.cv_mse,
        "y_std": y_std,
        "y_var": y_var,
        "y_min": y_min,
        "y_max": y_max,
        "pred_std": pred_std,
        "pred_min": pred_min,
        "pred_max": pred_max,
        "nmse": nmse,
        "rmse_over_y_std": rmse_over_y_std,
        "intercept_only_mse": intercept_only_mse,
        "intercept_only_flag": intercept_only_flag,
    }


def _rank_indices(abs_weights: np.ndarray, receptor_names: List[str]) -> List[int]:
    return sorted(
        range(len(abs_weights)),
        key=lambda idx: (-abs_weights[idx], receptor_names[idx]),
    )


def _compute_stability(
    *,
    X: np.ndarray,
    y: np.ndarray,
    receptor_names: List[str],
    modelclass: str,
    lambdas: np.ndarray,
    l1_ratios: List[float],
    cv_folds: int,
    scale_features: bool,
    seed: int,
) -> List[Dict[str, float]]:
    n_samples = X.shape[0]
    if n_samples < 2:
        return []

    stats = {
        name: {
            "selected": 0,
            "signs": [],
            "abs_weights": [],
            "ranks": [],
        }
        for name in receptor_names
    }

    for holdout_idx in range(n_samples):
        train_mask = np.ones(n_samples, dtype=bool)
        train_mask[holdout_idx] = False
        X_train = X[train_mask]
        y_train = y[train_mask]

        if modelclass == "lasso":
            weights, _cv_mse, _best_lambda, _pred = _fit_lasso(
                X_train,
                y_train,
                receptor_names,
                lambdas,
                cv_folds,
                StandardScaler().fit(X_train) if scale_features else None,
                seed,
            )
            coef = np.zeros(len(receptor_names), dtype=np.float64)
            for idx, name in enumerate(receptor_names):
                if name in weights:
                    coef[idx] = weights[name]
        elif modelclass.startswith("elasticnet"):
            if scale_features:
                X_train_scaled = StandardScaler().fit_transform(X_train)
            else:
                X_train_scaled = X_train
            fit = _fit_elasticnet(X_train_scaled, y_train, lambdas, l1_ratios, seed)
            coef = fit.coef
        elif modelclass == "ridge":
            if scale_features:
                X_train_scaled = StandardScaler().fit_transform(X_train)
            else:
                X_train_scaled = X_train
            fit = _fit_ridge(X_train_scaled, y_train, lambdas, seed)
            coef = fit.coef
        else:
            raise ValueError(f"Unknown modelclass: {modelclass}")

        abs_weights = np.abs(coef)
        if modelclass == "ridge":
            selected_indices = list(range(len(coef)))
            ranked_indices = _rank_indices(abs_weights, receptor_names)
        else:
            selected_indices = [i for i, w in enumerate(abs_weights) if w > 1e-6]
            ranked_indices = _rank_indices(abs_weights, receptor_names)

        rank_lookup = {idx: rank + 1 for rank, idx in enumerate(ranked_indices)}

        for idx in selected_indices:
            name = receptor_names[idx]
            weight = float(coef[idx])
            stats[name]["selected"] += 1
            stats[name]["signs"].append(1 if weight >= 0 else -1)
            stats[name]["abs_weights"].append(abs(weight))
            stats[name]["ranks"].append(rank_lookup[idx])

    rows = []
    for name, entry in stats.items():
        selected = entry["selected"]
        selection_frequency = selected / n_samples
        signs = entry["signs"]
        if signs:
            median_sign = 1 if float(np.median(signs)) >= 0 else -1
            sign_consistency = sum(1 for s in signs if s == median_sign) / len(signs)
        else:
            sign_consistency = 0.0

        abs_weights = entry["abs_weights"]
        ranks = entry["ranks"]

        rows.append(
            {
                "condition": None,
                "mode": None,
                "modelclass": modelclass,
                "orn_name": name,
                "selection_frequency": float(selection_frequency),
                "sign_consistency": float(sign_consistency),
                "mean_abs_weight": float(np.mean(abs_weights)) if abs_weights else 0.0,
                "std_abs_weight": float(np.std(abs_weights)) if abs_weights else 0.0,
                "mean_rank": float(np.mean(ranks)) if ranks else float("nan"),
                "n_folds": n_samples,
            }
        )

    return rows


def _format_shortlist_table(rows: List[Dict[str, float]]) -> str:
    if not rows:
        return "(no stable ORNs detected)\n"

    header = "| ORN | stability_score | selection_frequency | sign_consistency | mean_abs_weight | mean_rank |\n"
    header += "| --- | --- | --- | --- | --- | --- |\n"
    lines = [header]
    for row in rows:
        lines.append(
            f"| {row['orn_name']} | {row['stability_score']:.3f} | "
            f"{row['selection_frequency']:.3f} | {row['sign_consistency']:.3f} | "
            f"{row['mean_abs_weight']:.4f} | {row['mean_rank']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _is_missing_control_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if "no matched control mapping" in message:
        return True
    if "control" in message and "not found in behavioral data" in message:
        return True
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run stability scoring + standardized metrics for LASSO/Ridge/ElasticNet.",
    )

    parser.add_argument("--door_cache", required=True, help="Path to DoOR cache directory.")
    parser.add_argument("--behavior_csv", required=True, help="Path to behavioral matrix CSV.")
    parser.add_argument(
        "--conditions",
        required=True,
        action="append",
        help="Condition name(s). Repeat flag or pass comma-separated list.",
    )
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
        help="Comma-separated lambda values for raw runs.",
    )
    parser.add_argument(
        "--lambda_range_delta",
        default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0",
        help="Comma-separated lambda values for ΔPER runs.",
    )
    parser.add_argument(
        "--subtract_control",
        action="store_true",
        help="If set, run ΔPER (opto - control) in addition to raw.",
    )
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
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Defaults to diagnostics/stability_<timestamp>.",
    )
    parser.add_argument(
        "--include_ridge_stability",
        action="store_true",
        help="If set, compute ridge stability ranks (optional).",
    )
    parser.add_argument(
        "--adult_only_masking",
        action="store_true",
        help="Restrict to adult-only receptors via training_receptor_set.json.",
    )
    parser.add_argument(
        "--training_receptor_set_path",
        default=None,
        help="Optional path to training_receptor_set.json.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = _parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    conditions = _parse_conditions(args.conditions)
    if not conditions:
        raise ValueError("No valid conditions provided.")

    lambda_range = _parse_lambda_range(args.lambda_range)
    lambda_range_delta = _parse_lambda_range(args.lambda_range_delta)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("diagnostics") / f"stability_{timestamp}"
    )
    _ensure_dir(output_dir)

    command_line = " ".join(shlex.quote(arg) for arg in sys.argv)
    (output_dir / "RUN_COMMANDS.txt").write_text(command_line + "\n")

    predictor = LassoBehavioralPredictor(
        doorcache_path=args.door_cache,
        behavior_csv_path=args.behavior_csv,
        adult_only_masking=args.adult_only_masking,
        training_receptor_set_path=args.training_receptor_set_path,
    )

    model_metrics_rows: List[Dict[str, float]] = []
    stability_rows: List[Dict[str, float]] = []
    skipped_delta_conditions: List[str] = []

    l1_ratios = [0.2, 0.5, 0.8]

    modes_to_run = [("raw", False)]
    if args.subtract_control:
        modes_to_run.append(("delta", True))

    for condition in conditions:
        for mode, subtract_control in modes_to_run:
            try:
                lambda_values = lambda_range_delta if subtract_control else lambda_range
                results = predictor.fit_behavior(
                    condition_name=condition,
                    prediction_mode=args.prediction_mode,
                    lambda_range=lambda_values.tolist() if lambda_values is not None else None,
                    cv_folds=args.cv_folds,
                    subtract_control=subtract_control,
                    control_condition=args.control_condition,
                    missing_control_policy=args.missing_control_policy,
                )
            except ValueError as exc:
                if subtract_control and _is_missing_control_error(exc):
                    if args.missing_control_policy == "skip":
                        logger.warning(
                            "No control found for '%s'; skipping ΔPER run (missing_control_policy=skip).",
                            condition,
                        )
                        skipped_delta_conditions.append(condition)
                        continue
                raise

            X = results.feature_matrix
            y = results.actual_per
            receptor_names = list(results.receptor_names)

            intercept_only_mse = _loocv_intercept_mse(y)

            scaler = StandardScaler().fit(X) if predictor.scale_features else None

            lasso_weights, lasso_cv_mse, lasso_lambda, lasso_pred = _fit_lasso(
                X,
                y,
                receptor_names,
                lambda_values,
                args.cv_folds,
                scaler,
                args.seed,
            )
            lasso_coef = np.array([lasso_weights.get(name, 0.0) for name in receptor_names])
            lasso_fit = ModelFit(
                modelclass="lasso",
                lambda_value=lasso_lambda,
                l1_ratio=None,
                cv_mse=lasso_cv_mse,
                n_selected=len(lasso_weights),
                coef=lasso_coef,
                y_pred=lasso_pred,
            )
            model_metrics_rows.append(
                _collect_metrics(
                    condition=condition,
                    mode=mode,
                    model=lasso_fit,
                    y=y,
                    intercept_only_mse=intercept_only_mse,
                )
            )

            X_scaled = scaler.transform(X) if scaler is not None else X
            ridge_fit = _fit_ridge(X_scaled, y, lambda_values, args.seed)
            model_metrics_rows.append(
                _collect_metrics(
                    condition=condition,
                    mode=mode,
                    model=ridge_fit,
                    y=y,
                    intercept_only_mse=intercept_only_mse,
                )
            )

            elasticnet_fit = _fit_elasticnet(X_scaled, y, lambda_values, l1_ratios, args.seed)
            model_metrics_rows.append(
                _collect_metrics(
                    condition=condition,
                    mode=mode,
                    model=elasticnet_fit,
                    y=y,
                    intercept_only_mse=intercept_only_mse,
                )
            )

            lasso_intercept_only = _is_intercept_only(
                lasso_fit.n_selected,
                float(np.std(lasso_fit.y_pred)),
                lasso_fit.cv_mse,
                intercept_only_mse,
            )

            stability_models: List[str] = []
            if not lasso_intercept_only:
                stability_models.append("lasso")
            elif mode == "delta":
                stability_models.append(elasticnet_fit.modelclass)
            if args.include_ridge_stability:
                stability_models.append("ridge")

            for modelclass in stability_models:
                stability_model = modelclass
                if modelclass.startswith("elasticnet"):
                    stability_model = "elasticnet"

                rows = _compute_stability(
                    X=X,
                    y=y,
                    receptor_names=receptor_names,
                    modelclass=stability_model,
                    lambdas=lambda_values,
                    l1_ratios=l1_ratios,
                    cv_folds=args.cv_folds,
                    scale_features=predictor.scale_features,
                    seed=args.seed,
                )
                for row in rows:
                    row["condition"] = condition
                    row["mode"] = mode
                    row["modelclass"] = modelclass
                stability_rows.extend(rows)

    metrics_df = pd.DataFrame(model_metrics_rows)
    metrics_df = metrics_df.sort_values(["condition", "mode", "modelclass"])
    metrics_path = output_dir / "model_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    stability_columns = [
        "condition",
        "mode",
        "modelclass",
        "orn_name",
        "selection_frequency",
        "sign_consistency",
        "mean_abs_weight",
        "std_abs_weight",
        "mean_rank",
        "n_folds",
    ]
    stability_df = pd.DataFrame(stability_rows, columns=stability_columns)
    if not stability_df.empty:
        stability_df = stability_df.sort_values(
            ["condition", "mode", "modelclass", "orn_name"]
        )
    stability_path = output_dir / "stability_per_condition.csv"
    stability_df.to_csv(stability_path, index=False)

    _write_shortlist_and_summary(
        output_dir=output_dir,
        metrics_df=metrics_df,
        stability_df=stability_df,
        skipped_delta_conditions=skipped_delta_conditions,
    )

    logger.info("Wrote outputs to %s", output_dir)


def _write_shortlist_and_summary(
    *,
    output_dir: Path,
    metrics_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    skipped_delta_conditions: List[str],
) -> None:
    short_path = output_dir / "EXPERIMENT_SHORTLIST.md"
    summary_path = output_dir / "SUMMARY.md"

    modes = sorted(metrics_df["mode"].unique()) if not metrics_df.empty else []
    conditions = sorted(metrics_df["condition"].unique()) if not metrics_df.empty else []

    def _primary_model(condition: str, mode: str) -> str:
        lasso_row = metrics_df[
            (metrics_df["condition"] == condition)
            & (metrics_df["mode"] == mode)
            & (metrics_df["modelclass"] == "lasso")
        ]
        if lasso_row.empty:
            return "lasso"
        if bool(lasso_row.iloc[0]["intercept_only_flag"]):
            elasticnet_rows = metrics_df[
                (metrics_df["condition"] == condition)
                & (metrics_df["mode"] == mode)
                & (metrics_df["modelclass"] == "elasticnet")
            ]
            if not elasticnet_rows.empty:
                return "elasticnet"
        return "lasso"

    def _confidence_flags(condition: str, mode: str, modelclass: str) -> List[str]:
        row = metrics_df[
            (metrics_df["condition"] == condition)
            & (metrics_df["mode"] == mode)
            & (metrics_df["modelclass"] == modelclass)
        ]
        flags: List[str] = []
        if row.empty:
            if mode == "delta" and condition in skipped_delta_conditions:
                flags.append("ΔPER unavailable (missing control)")
            return flags
        row = row.iloc[0]
        if row["nmse"] >= 1.0:
            flags.append("no better than baseline scale-adjusted")
        if bool(row["intercept_only_flag"]):
            flags.append("no sparse signal; use ridge/elasticnet")
        if mode == "delta" and condition in skipped_delta_conditions:
            flags.append("ΔPER unavailable (missing control)")
        return flags

    short_lines = ["# Experiment Target Shortlist", ""]

    summary_lines = ["# Stability + Metrics Summary", ""]
    summary_lines.append("## Key findings")
    summary_lines.append("")

    bullet_points = []
    bullet_points.append(
        f"Ran stability for {len(conditions)} conditions across modes: {', '.join(modes) if modes else 'none'}"
    )
    if skipped_delta_conditions:
        bullet_points.append(
            f"ΔPER skipped (missing controls): {', '.join(sorted(set(skipped_delta_conditions)))}"
        )
    intercept_only_rows = metrics_df[metrics_df["intercept_only_flag"]]
    bullet_points.append(
        f"Intercept-only LASSO runs: {len(intercept_only_rows)} (see model_metrics.csv)"
    )
    if not metrics_df.empty:
        bullet_points.append(
            f"NMSE range: {metrics_df['nmse'].min():.3f}–{metrics_df['nmse'].max():.3f}"
        )
    bullet_points.append("Use nmse/rmse_over_y_std for cross-condition comparison (scale-aware).")
    bullet_points.append("Stability scores use selection_frequency × sign_consistency.")
    bullet_points.append("Shortlist excludes ORNs with stability_score=0.")
    bullet_points.append("Raw mode always run; ΔPER only if requested and controls available.")

    for bullet in bullet_points[:8]:
        summary_lines.append(f"- {bullet}")

    summary_lines.append("")

    for condition in conditions:
        for mode in modes:
            short_lines.append(f"## {condition} ({mode})")
            primary_model = _primary_model(condition, mode)

            subset = stability_df[
                (stability_df["condition"] == condition)
                & (stability_df["mode"] == mode)
                & (stability_df["modelclass"] == primary_model)
            ].copy()

            if subset.empty:
                short_lines.append("(no stability data)")
                short_lines.append("")
                continue

            subset["stability_score"] = (
                subset["selection_frequency"] * subset["sign_consistency"]
            )
            subset = subset[subset["stability_score"] > 0].sort_values(
                ["stability_score", "mean_abs_weight"], ascending=False
            )
            top_rows = subset.head(5).to_dict(orient="records")

            flags = _confidence_flags(condition, mode, primary_model)
            if flags:
                short_lines.append(f"Confidence flags: {', '.join(flags)}")
            short_lines.append("\n" + _format_shortlist_table(top_rows))

            summary_lines.append(f"## {condition} ({mode})")
            summary_lines.append(f"Primary model: {primary_model}")
            if flags:
                summary_lines.append(f"Confidence flags: {', '.join(flags)}")
            summary_lines.append("\n" + _format_shortlist_table(top_rows))

    short_path.write_text("\n".join(short_lines))
    summary_path.write_text("\n".join(summary_lines))


if __name__ == "__main__":
    main()
