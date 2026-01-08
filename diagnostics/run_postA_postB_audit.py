#!/usr/bin/env python3
"""
Audit LASSO behavioral prediction pipeline after Part A/B changes.

Writes a timestamped diagnostics run folder with metrics, artifacts, and plots.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

# Add src to path for repo-local runs.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from door_toolkit.pathways.behavioral_prediction import (
    LassoBehavioralPredictor,
    apply_receptor_ablation,
    fit_lasso_with_fixed_scaler,
    get_top_receptors_by_weight,
    restrict_to_receptors,
)

logger = logging.getLogger(__name__)


@dataclass
class RunResult:
    condition: str
    mode: str
    modelclass: str
    repeat: int
    cv_mse: float
    lambda_value: float
    n_selected: int
    pred_std: float
    y_std: float
    intercept_only_mse: float
    nmse: float
    mae: float
    mae_over_y_std: float
    intercept_only: bool
    chosen_params: Dict
    weights: Dict[str, float]
    y_stats: Dict
    pred_stats: Dict
    reproducible: bool
    mutation_ok: bool


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


def _parse_lambda_range(value: str) -> np.ndarray:
    tokens = [token.strip() for token in value.split(",") if token.strip()]
    if not tokens:
        raise ValueError("lambda_range cannot be empty")
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


def _cv_mse_grid(model_factory, X: np.ndarray, y: np.ndarray, lambdas: np.ndarray) -> List[float]:
    loo = LeaveOneOut()
    mse_values: List[float] = []
    for lam in lambdas:
        model = model_factory(lam)
        scores = cross_val_score(model, X, y, cv=loo, scoring="neg_mean_squared_error")
        mse_values.append(float(-np.mean(scores)))
    return mse_values


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
) -> Tuple[float, float, np.ndarray, float]:
    def factory(alpha: float) -> Ridge:
        return Ridge(alpha=alpha, random_state=42)

    mse_grid = _cv_mse_grid(factory, X, y, lambdas)
    best_idx = int(np.argmin(mse_grid))
    best_lambda = float(lambdas[best_idx])

    model = factory(best_lambda)
    model.fit(X, y)
    y_pred = model.predict(X)
    return best_lambda, float(mse_grid[best_idx]), model.coef_, y_pred


def _fit_elasticnet(
    X: np.ndarray,
    y: np.ndarray,
    lambdas: np.ndarray,
    l1_ratios: List[float],
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
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
                random_state=42,
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

    return best_lambda, float(best_mse), best_l1, best_coef, best_pred


def _fit_lasso(
    X: np.ndarray,
    y: np.ndarray,
    receptor_names: List[str],
    lambdas: np.ndarray,
    cv_folds: int,
    scaler: Optional[StandardScaler],
) -> Tuple[Dict[str, float], float, float, np.ndarray]:
    weights, _cv_r2, cv_mse, best_lambda, y_pred = fit_lasso_with_fixed_scaler(
        X=X,
        y=y,
        receptor_names=receptor_names,
        scaler=scaler,
        lambda_range=lambdas,
        cv_folds=cv_folds,
    )
    return weights, float(cv_mse), float(best_lambda), y_pred


def _get_stats(values: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "n": int(len(values)),
    }


def _plot_mse_grid(
    lambdas: np.ndarray,
    mse_values: List[float],
    title: str,
    path: Path,
) -> None:
    _ensure_dir(path.parent)
    plt.figure(figsize=(6, 4))
    plt.plot(lambdas, mse_values, marker="o")
    plt.xscale("log")
    plt.xlabel("lambda")
    plt.ylabel("LOOCV MSE")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _plot_scatter(y: np.ndarray, y_pred: np.ndarray, title: str, path: Path) -> None:
    _ensure_dir(path.parent)
    plt.figure(figsize=(5, 5))
    plt.scatter(y, y_pred, alpha=0.7, edgecolors="k")
    min_val = float(min(np.min(y), np.min(y_pred)))
    max_val = float(max(np.max(y), np.max(y_pred)))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _mutation_check(
    X: np.ndarray,
    receptor_names: List[str],
    receptors_to_use: List[str],
) -> bool:
    X_before = X.copy()
    _ = apply_receptor_ablation(
        X=X,
        receptor_names=receptor_names,
        receptors_to_ablate=receptors_to_use,
    )
    _ = restrict_to_receptors(
        X=X,
        receptor_names=receptor_names,
        receptors_to_keep=receptors_to_use,
    )
    return np.array_equal(X, X_before)


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
                "Provide control_condition or disable subtract_control."
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


def _run_one(
    *,
    condition: str,
    mode: str,
    predictor: LassoBehavioralPredictor,
    lambdas: np.ndarray,
    prediction_mode: str,
    subtract_control: bool,
    control_condition: Optional[str],
    missing_control_policy: str,
    seed: int,
    output_dir: Path,
) -> Tuple[List[RunResult], Dict]:
    valid_odorants, resolved_condition, control_resolved, n_pairs_used = _build_valid_odorants(
        predictor,
        condition=condition,
        subtract_control=subtract_control,
        control_condition=control_condition,
        missing_control_policy=missing_control_policy,
    )

    X, test_odorants, y = _extract_features(
        predictor,
        valid_odorants,
        resolved_condition,
        prediction_mode,
    )

    if X.shape[0] < 3:
        raise ValueError(f"Insufficient samples for {condition}: {X.shape[0]}")

    receptor_names = list(predictor.masked_receptor_names)
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    top_receptors = get_top_receptors_by_weight(
        fit_lasso_with_fixed_scaler(
            X=X,
            y=y,
            receptor_names=receptor_names,
            scaler=scaler,
            lambda_range=lambdas,
            cv_folds=X.shape[0],
        )[0],
        3,
    )
    if not top_receptors:
        top_receptors = receptor_names[:1]

    mutation_ok = _mutation_check(X, receptor_names, top_receptors)

    focus_receptors = top_receptors[:2] if len(top_receptors) >= 2 else top_receptors
    X_focus, focus_names, _ = restrict_to_receptors(
        X=X,
        receptor_names=receptor_names,
        receptors_to_keep=focus_receptors,
    )
    focus_scaler = StandardScaler().fit(X_focus)
    X_focus_scaled = focus_scaler.transform(X_focus)

    intercept_mse = _loocv_intercept_mse(y)
    y_stats = _get_stats(y)

    results: List[RunResult] = []
    artifacts: Dict = {
        "condition": resolved_condition,
        "mode": mode,
        "control_condition": control_resolved,
        "n_pairs_used": int(n_pairs_used),
        "test_odorants": test_odorants,
        "y_stats": y_stats,
        "intercept_only_mse": intercept_mse,
        "runs": [],
        "mutation_ok": mutation_ok,
    }

    model_configs = [
        ("lasso", None),
        ("ridge", None),
        ("elasticnet", 0.2),
        ("elasticnet", 0.5),
        ("elasticnet", 0.8),
    ]

    for modelclass, l1_ratio in model_configs:
        for repeat_idx in (1, 2):
            random.seed(seed)
            np.random.seed(seed)

            if modelclass == "lasso":
                weights, cv_mse, best_lambda, y_pred = _fit_lasso(
                    X=X,
                    y=y,
                    receptor_names=receptor_names,
                    lambdas=lambdas,
                    cv_folds=X.shape[0],
                    scaler=scaler,
                )
                coef = np.zeros(len(receptor_names))
                for idx, name in enumerate(receptor_names):
                    if name in weights:
                        coef[idx] = weights[name]
                chosen_params = {"lambda": best_lambda}
                n_selected = int(np.sum(np.abs(coef) > 1e-6))
            elif modelclass == "ridge":
                best_lambda, cv_mse, coef, y_pred = _fit_ridge(
                    X=X_scaled,
                    y=y,
                    lambdas=lambdas,
                )
                weights = {
                    receptor_names[i]: float(coef[i])
                    for i in range(len(receptor_names))
                    if abs(coef[i]) > 1e-6
                }
                chosen_params = {"lambda": best_lambda}
                n_selected = int(np.sum(np.abs(coef) > 1e-6))
            else:
                best_lambda, cv_mse, best_l1, coef, y_pred = _fit_elasticnet(
                    X=X_scaled,
                    y=y,
                    lambdas=lambdas,
                    l1_ratios=[l1_ratio],
                )
                weights = {
                    receptor_names[i]: float(coef[i])
                    for i in range(len(receptor_names))
                    if abs(coef[i]) > 1e-6
                }
                chosen_params = {"lambda": best_lambda, "l1_ratio": best_l1}
                n_selected = int(np.sum(np.abs(coef) > 1e-6))

            pred_stats = _get_stats(y_pred)
            mae = float(np.mean(np.abs(y - y_pred)))
            y_std = float(y_stats["std"])
            nmse = float(cv_mse) / (y_std**2 + 1e-12)
            mae_over_y_std = mae / (y_std + 1e-12)
            intercept_only = (
                n_selected == 0 and abs(float(cv_mse) - float(intercept_mse)) < 1e-12
            )
            reproducible = True

            run = RunResult(
                condition=resolved_condition,
                mode=mode,
                modelclass=modelclass if modelclass != "elasticnet" else f"elasticnet_{l1_ratio}",
                repeat=repeat_idx,
                cv_mse=float(cv_mse),
                lambda_value=float(chosen_params["lambda"]),
                n_selected=n_selected,
                pred_std=float(pred_stats["std"]),
                y_std=y_std,
                intercept_only_mse=float(intercept_mse),
                nmse=nmse,
                mae=mae,
                mae_over_y_std=mae_over_y_std,
                intercept_only=intercept_only,
                chosen_params=chosen_params,
                weights=weights,
                y_stats=y_stats,
                pred_stats=pred_stats,
                reproducible=reproducible,
                mutation_ok=mutation_ok,
            )

            artifacts["runs"].append(
                {
                    "repeat": repeat_idx,
                    "modelclass": run.modelclass,
                    "cv_mse": run.cv_mse,
                    "lambda": run.lambda_value,
                    "n_selected": run.n_selected,
                    "pred_stats": pred_stats,
                    "mae": mae,
                    "nmse": nmse,
                    "mae_over_y_std": mae_over_y_std,
                    "intercept_only": intercept_only,
                    "weights": weights,
                }
            )

            # Plot for first repeat only
            if repeat_idx == 1:
                model_dir = output_dir / "plots"
                if modelclass == "lasso":
                    mse_grid = _cv_mse_grid(
                        lambda alpha: Lasso(alpha=alpha, max_iter=10000, random_state=42),
                        X_scaled,
                        y,
                        lambdas,
                    )
                    title = f"{resolved_condition} {mode} lasso"
                    plot_path = model_dir / f"cv_mse_vs_lambda_{resolved_condition}_{mode}_lasso.png"
                    _plot_mse_grid(lambdas, mse_grid, title, plot_path)
                    scatter_path = model_dir / f"y_vs_pred_scatter_{resolved_condition}_{mode}_lasso.png"
                    _plot_scatter(y, y_pred, title, scatter_path)
                elif modelclass == "ridge":
                    mse_grid = _cv_mse_grid(
                        lambda alpha: Ridge(alpha=alpha, random_state=42),
                        X_scaled,
                        y,
                        lambdas,
                    )
                    title = f"{resolved_condition} {mode} ridge"
                    plot_path = model_dir / f"cv_mse_vs_lambda_{resolved_condition}_{mode}_ridge.png"
                    _plot_mse_grid(lambdas, mse_grid, title, plot_path)
                    scatter_path = model_dir / f"y_vs_pred_scatter_{resolved_condition}_{mode}_ridge.png"
                    _plot_scatter(y, y_pred, title, scatter_path)
                else:
                    title = f"{resolved_condition} {mode} elasticnet l1={l1_ratio}"
                    mse_grid = _cv_mse_grid(
                        lambda alpha: ElasticNet(
                            alpha=alpha,
                            l1_ratio=l1_ratio,
                            random_state=42,
                            max_iter=10000,
                        ),
                        X_scaled,
                        y,
                        lambdas,
                    )
                    suffix = f"elasticnet_{l1_ratio}"
                    plot_path = model_dir / f"cv_mse_vs_lambda_{resolved_condition}_{mode}_{suffix}.png"
                    _plot_mse_grid(lambdas, mse_grid, title, plot_path)
                    scatter_path = model_dir / f"y_vs_pred_scatter_{resolved_condition}_{mode}_{suffix}.png"
                    _plot_scatter(y, y_pred, title, scatter_path)

            results.append(run)

    # Focus-mode LASSO sanity run (top-2 by baseline weights)
    for repeat_idx in (1, 2):
        random.seed(seed)
        np.random.seed(seed)

        weights, cv_mse, best_lambda, y_pred = _fit_lasso(
            X=X_focus,
            y=y,
            receptor_names=focus_names,
            lambdas=lambdas,
            cv_folds=X_focus.shape[0],
            scaler=focus_scaler,
        )
        coef = np.zeros(len(focus_names))
        for idx, name in enumerate(focus_names):
            if name in weights:
                coef[idx] = weights[name]
        pred_stats = _get_stats(y_pred)
        n_selected = int(np.sum(np.abs(coef) > 1e-6))
        mae = float(np.mean(np.abs(y - y_pred)))
        y_std = float(y_stats["std"])
        nmse = float(cv_mse) / (y_std**2 + 1e-12)
        mae_over_y_std = mae / (y_std + 1e-12)
        intercept_only = (
            n_selected == 0 and abs(float(cv_mse) - float(intercept_mse)) < 1e-12
        )
        run = RunResult(
            condition=resolved_condition,
            mode=mode,
            modelclass="lasso_focus_top2",
            repeat=repeat_idx,
            cv_mse=float(cv_mse),
            lambda_value=float(best_lambda),
            n_selected=n_selected,
            pred_std=float(pred_stats["std"]),
            y_std=y_std,
            intercept_only_mse=float(intercept_mse),
            nmse=nmse,
            mae=mae,
            mae_over_y_std=mae_over_y_std,
            intercept_only=intercept_only,
            chosen_params={"lambda": best_lambda, "focus_receptors": focus_names},
            weights=weights,
            y_stats=y_stats,
            pred_stats=pred_stats,
            reproducible=True,
            mutation_ok=mutation_ok,
        )

        artifacts["runs"].append(
            {
                "repeat": repeat_idx,
                "modelclass": "lasso_focus_top2",
                "cv_mse": run.cv_mse,
                "lambda": run.lambda_value,
                "n_selected": run.n_selected,
                "pred_stats": pred_stats,
                "mae": mae,
                "nmse": nmse,
                "mae_over_y_std": mae_over_y_std,
                "intercept_only": intercept_only,
                "weights": weights,
            }
        )

        if repeat_idx == 1:
            mse_grid = _cv_mse_grid(
                lambda alpha: Lasso(alpha=alpha, max_iter=10000, random_state=42),
                X_focus_scaled,
                y,
                lambdas,
            )
            title = f"{resolved_condition} {mode} lasso focus top2"
            plot_path = output_dir / "plots" / (
                f"cv_mse_vs_lambda_{resolved_condition}_{mode}_lasso_focus_top2.png"
            )
            _plot_mse_grid(lambdas, mse_grid, title, plot_path)
            scatter_path = output_dir / "plots" / (
                f"y_vs_pred_scatter_{resolved_condition}_{mode}_lasso_focus_top2.png"
            )
            _plot_scatter(y, y_pred, title, scatter_path)

        results.append(run)

    # Reproducibility check (compare repeat 1 vs 2 per modelclass)
    for modelclass in {r.modelclass for r in results}:
        runs = [r for r in results if r.modelclass == modelclass]
        runs_sorted = sorted(runs, key=lambda r: r.repeat)
        if len(runs_sorted) < 2:
            continue
        r1, r2 = runs_sorted[0], runs_sorted[1]
        same = (
            abs(r1.cv_mse - r2.cv_mse) < 1e-12
            and abs(r1.lambda_value - r2.lambda_value) < 1e-12
            and abs(r1.pred_std - r2.pred_std) < 1e-12
            and r1.weights == r2.weights
        )
        r1.reproducible = same
        r2.reproducible = same

    return results, artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit post-A/post-B LASSO pipeline with diagnostics outputs.",
    )
    parser.add_argument("--door_cache", required=True, help="Path to DoOR cache directory.")
    parser.add_argument("--behavior_csv", required=True, help="Path to behavioral matrix CSV.")
    parser.add_argument(
        "--conditions",
        required=True,
        help="Comma-separated condition list (e.g., opto_hex,opto_EB)",
    )
    parser.add_argument(
        "--lambda_range",
        default="1e-4,1e-3,1e-2,1e-1,1.0",
        help="Comma-separated lambda values.",
    )
    parser.add_argument(
        "--lambda_range_delta",
        default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0",
        help="Comma-separated lambda values for ΔPER runs.",
    )
    parser.add_argument(
        "--prediction_mode",
        choices=["test_odorant", "trained_odorant", "interaction"],
        default="test_odorant",
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
        help="Optional output dir; default creates diagnostics/postA_postB_audit_<timestamp>",
    )

    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    conditions = _parse_conditions([args.conditions])
    lambdas_raw = _parse_lambda_range(args.lambda_range)
    lambdas_delta = _parse_lambda_range(args.lambda_range_delta)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("diagnostics") / f"postA_postB_audit_{timestamp}"

    _ensure_dir(output_dir)

    predictor = LassoBehavioralPredictor(
        doorcache_path=args.door_cache,
        behavior_csv_path=args.behavior_csv,
        scale_features=False,
        scale_targets=False,
    )

    all_metrics: List[Dict] = []
    all_artifacts: Dict[str, Dict] = {}

    for condition in conditions:
        mode_specs: List[Tuple[str, bool, np.ndarray]] = [("raw", False, lambdas_raw)]
        if args.subtract_control:
            if np.array_equal(lambdas_raw, lambdas_delta):
                mode_specs.append(("delta", True, lambdas_delta))
            else:
                mode_specs.append(("delta_base", True, lambdas_raw))
                mode_specs.append(("delta_extended", True, lambdas_delta))

        for mode, subtract_control, lambdas in mode_specs:
            if (
                subtract_control
                and args.missing_control_policy == "skip"
                and args.control_condition is None
            ):
                control_candidate = predictor._infer_control_condition(condition)
                if control_candidate is None:
                    logger.warning(
                        "%s %s skipped: no matched control mapping (missing_control_policy=skip).",
                        condition,
                        mode,
                    )
                    all_artifacts[f"{condition}_{mode}"] = {
                        "skipped": "no matched control mapping",
                    }
                    continue
                control_resolved = predictor._resolve_dataset_name(control_candidate)
                if control_resolved is None:
                    logger.warning(
                        "%s %s skipped: matched control '%s' not found (missing_control_policy=skip).",
                        condition,
                        mode,
                        control_candidate,
                    )
                    all_artifacts[f"{condition}_{mode}"] = {
                        "skipped": f"matched control '{control_candidate}' not found",
                    }
                    continue

            try:
                results, artifacts = _run_one(
                    condition=condition,
                    mode=mode,
                    predictor=predictor,
                    lambdas=lambdas,
                    prediction_mode=args.prediction_mode,
                    subtract_control=subtract_control,
                    control_condition=args.control_condition,
                    missing_control_policy=args.missing_control_policy,
                    seed=args.seed,
                    output_dir=output_dir,
                )
            except Exception as exc:
                logger.error("%s %s failed: %s", condition, mode, exc)
                all_artifacts[f"{condition}_{mode}"] = {"error": str(exc)}
                continue

            all_artifacts[f"{condition}_{mode}"] = artifacts

            for run in results:
                if run.repeat != 1:
                    continue
                all_metrics.append(
                    {
                        "condition": run.condition,
                        "mode": run.mode,
                        "modelclass": run.modelclass,
                        "cv_mse": run.cv_mse,
                        "lambda": run.lambda_value,
                        "n_selected": run.n_selected,
                        "pred_std": run.pred_std,
                        "y_std": run.y_std,
                        "intercept_only_mse": run.intercept_only_mse,
                        "nmse": run.nmse,
                        "mae": run.mae,
                        "mae_over_y_std": run.mae_over_y_std,
                        "intercept_only": run.intercept_only,
                        "reproducible": run.reproducible,
                        "mutation_ok": run.mutation_ok,
                    }
                )

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = output_dir / "audit_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)

    artifacts_path = output_dir / "audit_artifacts.json"
    with open(artifacts_path, "w", encoding="utf-8") as f:
        json.dump(all_artifacts, f, indent=2)

    summary_lines = [
        "# Audit Summary",
        f"Run folder: {output_dir}",
        f"Conditions: {', '.join(conditions)}",
        f"Prediction mode: {args.prediction_mode}",
        f"Subtract control: {args.subtract_control}",
        f"Lambda grid: {args.lambda_range}",
        f"Lambda grid (ΔPER): {args.lambda_range_delta}",
        "",
    ]

    if not metrics_df.empty:
        reproducible_counts = metrics_df.groupby(["modelclass", "mode"])["reproducible"].mean()
        summary_lines.append("## Reproducibility (repeat 1 vs 2)")
        for (modelclass, mode), ratio in reproducible_counts.items():
            summary_lines.append(f"- {modelclass} {mode}: {ratio:.2f} reproducible fraction")

        errors = {k: v for k, v in all_artifacts.items() if isinstance(v, dict) and "error" in v}
        skipped = {k: v for k, v in all_artifacts.items() if isinstance(v, dict) and "skipped" in v}
        if errors or skipped:
            summary_lines.append("")
            summary_lines.append("## Missing Controls / Skipped Runs")
            for key, value in errors.items():
                summary_lines.append(f"- {key}: {value['error']}")
            for key, value in skipped.items():
                summary_lines.append(f"- {key}: {value['skipped']}")

        mutation_issues = metrics_df.loc[~metrics_df["mutation_ok"]]
        summary_lines.append("")
        summary_lines.append("## Mutation Check (focus/ablation on copies)")
        if mutation_issues.empty:
            summary_lines.append("- All conditions reported mutation_ok=True")
        else:
            for _, row in mutation_issues.iterrows():
                summary_lines.append(
                    f"- Mutation detected: {row['condition']} {row['mode']} {row['modelclass']}"
                )

        delta_modes = sorted({m for m in metrics_df["mode"].unique() if m.startswith("delta")})
        summary_lines.append("")
        summary_lines.append("## ΔPER Collapse (constant predictions)")
        if not delta_modes:
            summary_lines.append("- No ΔPER runs were executed")
        else:
            preferred_delta = "delta_extended" if "delta_extended" in delta_modes else delta_modes[0]
            lasso_delta = metrics_df[
                (metrics_df["modelclass"] == "lasso") & (metrics_df["mode"] == preferred_delta)
            ]
            collapsed = lasso_delta[lasso_delta["pred_std"] < 1e-6]
            if collapsed.empty:
                summary_lines.append(f"- No LASSO {preferred_delta} runs had pred_std < 1e-6")
            else:
                for _, row in collapsed.iterrows():
                    summary_lines.append(
                        f"- {row['condition']}: pred_std={row['pred_std']:.6g}, "
                        f"n_selected={int(row['n_selected'])}, "
                        f"lambda={row['lambda']:.6g}, "
                        f"cv_mse={row['cv_mse']:.6g}, "
                        f"intercept_only_mse={row['intercept_only_mse']:.6g}"
                    )

            if "delta_base" in delta_modes and "delta_extended" in delta_modes:
                base_collapsed = metrics_df[
                    (metrics_df["modelclass"] == "lasso") & (metrics_df["mode"] == "delta_base")
                ]
                ext_collapsed = metrics_df[
                    (metrics_df["modelclass"] == "lasso")
                    & (metrics_df["mode"] == "delta_extended")
                ]
                summary_lines.append("")
                summary_lines.append("## ΔPER Grid Comparison (LASSO)")
                summary_lines.append(
                    f"- delta_base collapsed count: {int((base_collapsed['pred_std'] < 1e-6).sum())}"
                )
                summary_lines.append(
                    f"- delta_extended collapsed count: {int((ext_collapsed['pred_std'] < 1e-6).sum())}"
                )

            # Primary model selection for ΔPER (fallback to ElasticNet if LASSO intercept-only)
            summary_lines.append("")
            summary_lines.append("## ΔPER Primary Model (fallback if LASSO intercept-only)")
            primary_rows = []
            for condition in conditions:
                mode = preferred_delta
                lasso_row = metrics_df[
                    (metrics_df["condition"] == condition)
                    & (metrics_df["mode"] == mode)
                    & (metrics_df["modelclass"] == "lasso")
                ]
                if lasso_row.empty:
                    continue
                lasso_row = lasso_row.iloc[0]
                if bool(lasso_row["intercept_only"]):
                    candidates = metrics_df[
                        (metrics_df["condition"] == condition)
                        & (metrics_df["mode"] == mode)
                        & (metrics_df["modelclass"].str.startswith("elasticnet"))
                    ]
                    if candidates.empty:
                        primary_rows.append(
                            {
                                "condition": condition,
                                "mode": mode,
                                "primary_modelclass": "intercept_only",
                                "cv_mse": float(lasso_row["cv_mse"]),
                                "lambda": float(lasso_row["lambda"]),
                            }
                        )
                    else:
                        best = candidates.loc[candidates["cv_mse"].idxmin()]
                        primary_rows.append(
                            {
                                "condition": condition,
                                "mode": mode,
                                "primary_modelclass": best["modelclass"],
                                "cv_mse": float(best["cv_mse"]),
                                "lambda": float(best["lambda"]),
                            }
                        )
                else:
                    primary_rows.append(
                        {
                            "condition": condition,
                            "mode": mode,
                            "primary_modelclass": "lasso",
                            "cv_mse": float(lasso_row["cv_mse"]),
                            "lambda": float(lasso_row["lambda"]),
                        }
                    )

            if primary_rows:
                primary_df = pd.DataFrame(primary_rows)
                primary_df.to_csv(output_dir / "audit_primary_models.csv", index=False)
                for row in primary_rows:
                    summary_lines.append(
                        f"- {row['condition']} ({row['mode']}): {row['primary_modelclass']} "
                        f"(cv_mse={row['cv_mse']:.6g}, lambda={row['lambda']:.6g})"
                    )

        summary_lines.append("")
        summary_lines.append("## Top ORNs (raw LASSO, repeat 1)")
        for key, artifact in all_artifacts.items():
            if not key.endswith("_raw"):
                continue
            if "runs" not in artifact:
                continue
            lasso_runs = [
                r
                for r in artifact["runs"]
                if r.get("modelclass") == "lasso" and r.get("repeat") == 1
            ]
            if not lasso_runs:
                continue
            weights = lasso_runs[0].get("weights", {})
            top_receptors = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            receptor_str = ", ".join([f"{r}({w:.3g})" for r, w in top_receptors])
            summary_lines.append(
                f"- {artifact['condition']}: {receptor_str if receptor_str else 'no nonzero weights'}"
            )

        raw_lasso = metrics_df[(metrics_df["modelclass"] == "lasso") & (metrics_df["mode"] == "raw")]
        summary_lines.append("")
        summary_lines.append("## Scale vs Error (raw LASSO)")
        if len(raw_lasso) >= 3:
            corr = np.corrcoef(raw_lasso["y_std"], raw_lasso["cv_mse"])[0, 1]
            summary_lines.append(
                f"- Pearson corr(y_std, cv_mse) = {corr:.3f} (n={len(raw_lasso)})"
            )
        else:
            summary_lines.append("- Not enough conditions to compute correlation")
        summary_lines.append(
            "- Note: raw MSE is scale-dependent; compare nmse or mae/y_std for cross-condition checks."
        )

    summary_lines.append("")
    summary_lines.append("## Files")
    summary_lines.append("- audit_metrics.csv")
    summary_lines.append("- audit_primary_models.csv")
    summary_lines.append("- audit_artifacts.json")
    summary_lines.append("- plots/*.png")

    summary_path = output_dir / "AUDIT_SUMMARY.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    logger.info("Wrote audit outputs to %s", output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
