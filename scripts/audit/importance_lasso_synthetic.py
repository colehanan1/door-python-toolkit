#!/usr/bin/env python3
"""
Audit: synthetic LASSO importance invariants.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from door_toolkit.encoder import DoOREncoder
from door_toolkit.pathways.behavioral_prediction import LassoBehavioralPredictor


def _clean_token(text: str) -> str:
    return str(text).lower().replace("_", "").replace(" ", "").replace("-", "")


def resolve_door_name(csv_name: str, encoder: DoOREncoder) -> str | None:
    mapping = LassoBehavioralPredictor.ODORANT_NAME_MAPPING
    key = _clean_token(csv_name)
    candidates = mapping.get(key)
    if candidates is None:
        return None
    for candidate in candidates:
        try:
            encoder.encode(candidate)
            return candidate
        except KeyError:
            continue
    raise ValueError(f"No DoOR match for {csv_name}")


def fit_lasso_weights(
    X: np.ndarray,
    y: np.ndarray,
    receptor_names: List[str],
    lambda_range: np.ndarray,
    cv_folds: int = 5,
) -> Tuple[Dict[str, float], float]:
    X_scaled = StandardScaler().fit_transform(X)
    if X_scaled.shape[0] < 10:
        cv_folds_adjusted = X_scaled.shape[0]
    else:
        cv_folds_adjusted = min(cv_folds, X_scaled.shape[0])

    lasso_cv = LassoCV(
        alphas=lambda_range, cv=cv_folds_adjusted, max_iter=10000, random_state=42
    )
    lasso_cv.fit(X_scaled, y)

    lasso = Lasso(alpha=lasso_cv.alpha_, max_iter=10000, random_state=42)
    lasso.fit(X_scaled, y)

    weights = {
        receptor_names[i]: float(coef)
        for i, coef in enumerate(lasso.coef_)
        if abs(coef) > 1e-6
    }
    cv_scores = cross_val_score(
        lasso, X_scaled, y, cv=cv_folds_adjusted, scoring="neg_mean_squared_error"
    )
    cv_mse = float(-np.mean(cv_scores))
    return weights, cv_mse


def main() -> None:
    out_dir = Path("outputs/audit/importance_lasso_synthetic")
    out_dir.mkdir(parents=True, exist_ok=True)

    odors = [
        "Hexanol",
        "Ethyl_Butyrate",
        "Ethyl_Butyrate_(6-Training)",
        "Benzaldehyde",
        "Apple_Cider_Vinegar",
        "3-Octonol",
        "Citral",
        "Linalool",
    ]

    encoder = DoOREncoder("door_cache", use_torch=False)
    door_names = [resolve_door_name(o, encoder) for o in odors]
    if any(name is None for name in door_names):
        missing = [o for o, name in zip(odors, door_names) if name is None]
        raise ValueError(f"Unresolved odors in synthetic set: {missing}")
    profiles = np.stack(
        [
            np.asarray(encoder.encode(name, fill_missing=0.0), dtype=np.float64)
            for name in door_names
            if name is not None
        ],
        axis=0,
    )

    # Choose a receptor with high variance across odors.
    variances = profiles.var(axis=0)
    target_idx = int(np.argmax(variances))
    target_receptor = encoder.receptor_names[target_idx]
    responses = profiles[:, target_idx]
    resp_min = float(np.min(responses))
    resp_max = float(np.max(responses))
    if resp_max <= resp_min:
        raise ValueError("No dynamic range in synthetic responses.")
    y = (responses - resp_min) / (resp_max - resp_min)

    behavior_df = pd.DataFrame([y], index=["opto_hex"], columns=odors)
    behavior_csv = out_dir / "synthetic_behavior.csv"
    behavior_df.to_csv(behavior_csv)

    predictor = LassoBehavioralPredictor(
        doorcache_path="door_cache",
        behavior_csv_path=str(behavior_csv),
        scale_features=True,
        scale_targets=False,
    )
    lambda_range = np.logspace(-4, 0, 30)
    results = predictor.fit_behavior(
        condition_name="opto_hex",
        prediction_mode="test_odorant",
        lambda_range=lambda_range.tolist(),
        cv_folds=5,
    )

    # Invariant 1: target receptor should be top by absolute weight.
    if not results.lasso_weights:
        raise ValueError("No non-zero LASSO weights found.")
    top_receptor = max(results.lasso_weights.items(), key=lambda x: abs(x[1]))[0]
    assert top_receptor == target_receptor, (
        f"Expected top receptor {target_receptor}, got {top_receptor}"
    )

    # Invariant 2: permuting receptor order preserves top receptors by name.
    X = results.feature_matrix
    names = results.receptor_names
    weights_orig, cv_mse_orig = fit_lasso_weights(X, results.actual_per, names, lambda_range)

    perm = np.array([2, 0, 4, 1, 3, 5, 6, 7] + list(range(8, X.shape[1])), dtype=int)
    X_perm = X[:, perm]
    names_perm = [names[i] for i in perm]
    weights_perm, _ = fit_lasso_weights(X_perm, results.actual_per, names_perm, lambda_range)

    def top_receptors(weights: Dict[str, float], k: int = 3) -> List[str]:
        ordered = sorted(weights.items(), key=lambda x: abs(x[1]), reverse=True)
        return [name for name, _ in ordered[:k]]

    top_orig = top_receptors(weights_orig, k=3)
    top_perm = top_receptors(weights_perm, k=3)
    assert target_receptor in top_orig
    assert target_receptor in top_perm

    # Invariant 3: random labels worsen CV MSE.
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(results.actual_per)
    _, cv_mse_shuf = fit_lasso_weights(X, shuffled, names, lambda_range)
    assert cv_mse_shuf >= cv_mse_orig * 1.1, (
        f"Label shuffle did not degrade MSE (orig={cv_mse_orig:.4f}, shuf={cv_mse_shuf:.4f})"
    )

    print("OK: LASSO importance invariants passed.")


if __name__ == "__main__":
    main()
