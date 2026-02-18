"""
Glomerulus Weight Vector Regression
=====================================

Fit a regularized regression from glomerulus feature vectors to behavioral
labels and classify weights as positive / negative / zero.

Functions:
    fit_glomerulus_weight_vector: Fit LASSO/Ridge/ElasticNet and return weights.
    export_weight_report: Write weights.csv and model_summary.json to disk.
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.linear_model import Lasso, LassoCV, Ridge, RidgeCV, ElasticNet, ElasticNetCV

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_glomerulus_weight_vector(
    X: np.ndarray,
    y: np.ndarray,
    receptor_names: List[str],
    *,
    model: Literal["lasso", "ridge", "elasticnet"] = "lasso",
    standardize: bool = True,
    random_state: int = 0,
    zero_eps: float = 1e-6,
    cv: bool = True,
    target_mode: Literal["raw", "centered"] = "centered",
    y_raw: Optional[np.ndarray] = None,
    y_mean: Optional[float] = None,
) -> dict:
    """Fit a regularized regression and classify weights by sign.

    Args:
        X: Design matrix of shape (n_odors, n_receptors).
        y: Behavioral labels of shape (n_odors,). Can be raw or centered.
        receptor_names: DoOR receptor names for each column of X.
        model: Regularization type.
        standardize: Whether to standardize features before fitting.
        random_state: Random seed for reproducibility.
        zero_eps: Threshold for classifying a weight as zero.
        cv: Use cross-validated alpha selection (ignored if n_samples < 3).
        target_mode: "raw" for raw PER labels, "centered" for y - mean(y).
        y_raw: Original raw PER labels (for metadata tracking).
        y_mean: Mean of raw labels (for back-transformation).

    Returns:
        Dict with keys: weights, intercept, alpha, y_pred, r2, mse,
        weight_signs, receptor_names, model_name, random_state, zero_eps,
        standardize, n_samples, n_features, target_mode, y_raw, y_mean, y_fitted.
    """
    n_samples, n_features = X.shape
    if n_features != len(receptor_names):
        raise ValueError(
            f"X has {n_features} columns but {len(receptor_names)} receptor names provided"
        )

    # Standardize
    scaler = None
    X_fit = X.copy()
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_fit)

    # With very few samples, CV is unreliable — fall back to fixed alpha
    use_cv = cv and n_samples >= 3
    alphas = np.logspace(-4, 1, 50)

    if model == "lasso":
        if use_cv:
            reg = LassoCV(alphas=alphas, cv=min(n_samples, 5), max_iter=10000,
                          random_state=random_state)
        else:
            reg = Lasso(alpha=0.01, max_iter=10000, random_state=random_state)
    elif model == "ridge":
        if use_cv:
            reg = RidgeCV(alphas=alphas, cv=min(n_samples, 5))
        else:
            reg = Ridge(alpha=1.0)
    elif model == "elasticnet":
        if use_cv:
            reg = ElasticNetCV(alphas=alphas, cv=min(n_samples, 5), max_iter=10000,
                               random_state=random_state)
        else:
            reg = ElasticNet(alpha=0.01, max_iter=10000, random_state=random_state)
    else:
        raise ValueError(f"model must be lasso/ridge/elasticnet, got '{model}'")

    reg.fit(X_fit, y)

    weights = reg.coef_.ravel().astype(np.float64)
    intercept = float(reg.intercept_)
    alpha = float(reg.alpha_) if hasattr(reg, "alpha_") else float(reg.alpha)

    y_pred = reg.predict(X_fit)
    r2 = float(r2_score(y, y_pred))
    mse = float(mean_squared_error(y, y_pred))

    # Classify signs
    signs = np.where(weights > zero_eps, 1, np.where(weights < -zero_eps, -1, 0)).astype(int)

    return {
        "weights": weights,
        "intercept": intercept,
        "alpha": alpha,
        "y_pred": y_pred,
        "r2": r2,
        "mse": mse,
        "weight_signs": signs,
        "receptor_names": list(receptor_names),
        "model_name": model,
        "random_state": random_state,
        "zero_eps": zero_eps,
        "standardize": standardize,
        "n_samples": n_samples,
        "n_features": n_features,
        "target_mode": target_mode,
        "y_raw": y_raw.tolist() if y_raw is not None else None,
        "y_mean": y_mean,
        "y_fitted": y.tolist(),  # The y actually used for fitting
    }


def export_weight_report(
    results: dict,
    outdir: str,
    *,
    extra_meta: Optional[dict] = None,
) -> dict:
    """Write weights.csv and model_summary.json.

    Args:
        results: Output from fit_glomerulus_weight_vector.
        outdir: Directory to write files into (created if needed).
        extra_meta: Additional metadata to include in the JSON summary.

    Returns:
        Dict with paths to written files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    receptor_names = results["receptor_names"]
    weights = results["weights"]
    signs = results["weight_signs"]

    # Build weights dataframe (using DoOR receptor names, not FlyWire glomeruli)
    df = pd.DataFrame({
        "receptor": receptor_names,
        "weight": weights,
        "sign": ["+1" if s == 1 else "-1" if s == -1 else "0" for s in signs],
        "abs_weight": np.abs(weights),
    })
    df = df.sort_values("abs_weight", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    weights_path = outdir / "weights.csv"
    df.to_csv(weights_path, index=False)

    # Build JSON summary
    summary = {
        "model": results["model_name"],
        "alpha": results["alpha"],
        "intercept": results["intercept"],
        "r2": results["r2"],
        "mse": results["mse"],
        "n_samples": results["n_samples"],
        "n_features": results["n_features"],
        "n_positive": int((signs > 0).sum()),
        "n_negative": int((signs < 0).sum()),
        "n_zero": int((signs == 0).sum()),
        "random_state": results["random_state"],
        "zero_eps": results["zero_eps"],
        "standardize": results["standardize"],
        "target_mode": results.get("target_mode", "raw"),
        "y_raw": results.get("y_raw"),
        "y_mean": results.get("y_mean"),
        "y_fitted": results.get("y_fitted"),
    }
    if extra_meta:
        summary["config"] = extra_meta

    json_path = outdir / "model_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("[export_weight_report] Wrote %s and %s", weights_path, json_path)
    return {"weights_csv": str(weights_path), "model_summary_json": str(json_path)}
