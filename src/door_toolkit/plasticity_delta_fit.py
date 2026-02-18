"""
Plasticity Delta Fitting
========================

Fit sparse delta-weight models per training condition using:
    delta_PER = PER_trained - PER_opto_AIR

By default each condition's delta vector is mean-centered across the selected
odor set so weights capture relative plasticity shifts.
"""

import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_CONTROL_ROW = "opto_AIR"
DEFAULT_FOUR_ODOR_KEYS = (
    "benzaldehyde",
    "ethyl_butyrate",
    "1-hexanol",
    "3-octanol",
)

# Alias resolution is exact on normalized labels (lowercase, punctuation removed).
ODOR_ALIAS_PRIORITY = {
    "benzaldehyde": ["benzaldehyde", "benz"],
    "ethyl_butyrate": ["ethylbutyrate", "eb"],
    "1-hexanol": ["1hexanol", "hexanol"],
    "3-octanol": ["3octanol", "3octonol", "octanol", "octonol"],
}

# DoOR names used by the existing glomerulus/receptor feature builder.
ODOR_KEY_TO_DOOR = {
    "benzaldehyde": "benzaldehyde",
    "ethyl_butyrate": "ethyl butyrate",
    "1-hexanol": "1-hexanol",
    "3-octanol": "3-octanol",
}


def normalize_label(label: str) -> str:
    """Normalize labels for case/format-insensitive matching."""
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def sanitize_filename_token(value: str) -> str:
    """Create a stable filename token."""
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower())
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def condition_filename_token(value: str) -> str:
    """Prefer original condition label if filesystem-safe; else sanitize."""
    text = str(value).strip()
    if re.match(r"^[A-Za-z0-9._-]+$", text):
        return text
    return sanitize_filename_token(text)


def _resolve_row_label(index_values: Iterable[str], requested: str) -> str:
    """Resolve a row label exactly or by normalized match."""
    index_list = [str(x) for x in index_values]
    if requested in index_list:
        return requested

    requested_norm = normalize_label(requested)
    matches = [idx for idx in index_list if normalize_label(idx) == requested_norm]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Ambiguous row label '{0}' matches multiple rows: {1}".format(requested, matches)
        )
    raise ValueError("Row label '{0}' was not found".format(requested))


def resolve_odor_keys(odors: Optional[Sequence[str]] = None) -> List[str]:
    """Resolve user-requested odors to canonical odor keys."""
    if odors is None:
        return list(DEFAULT_FOUR_ODOR_KEYS)

    resolved: List[str] = []
    for odor in odors:
        normalized = normalize_label(odor)
        matched: List[str] = []
        for odor_key, aliases in ODOR_ALIAS_PRIORITY.items():
            alias_norm = [normalize_label(alias) for alias in aliases]
            if normalized in alias_norm:
                matched.append(odor_key)
        if len(matched) == 1:
            if matched[0] not in resolved:
                resolved.append(matched[0])
            continue
        if len(matched) > 1:
            raise ValueError(
                "Odor '{0}' is ambiguous across canonical keys: {1}".format(odor, matched)
            )
        raise ValueError(
            "Unrecognized odor '{0}'. Supported odors: {1}".format(
                odor, ", ".join(DEFAULT_FOUR_ODOR_KEYS)
            )
        )

    if len(resolved) != 4:
        raise ValueError(
            "Exactly 4 target odors are required; got {0}: {1}".format(len(resolved), resolved)
        )
    return resolved


def parse_odor_argument(odors: Optional[Sequence[str]]) -> List[str]:
    """Normalize CLI-style odor arguments into canonical odor keys."""
    if odors is None:
        return resolve_odor_keys(None)
    if isinstance(odors, str):
        tokens = [tok.strip() for tok in odors.split(",") if tok.strip()]
        return resolve_odor_keys(tokens)
    return resolve_odor_keys(list(odors))


def _resolve_odor_columns(
    columns: Sequence[str],
    odor_keys: Sequence[str],
) -> Dict[str, str]:
    """Map canonical odor keys to exact CSV column names."""
    normalized_to_columns: Dict[str, List[str]] = {}
    for col in columns:
        normalized_to_columns.setdefault(normalize_label(col), []).append(col)

    colmap: Dict[str, str] = {}
    used_columns = set()
    for odor_key in odor_keys:
        candidates = [normalize_label(x) for x in ODOR_ALIAS_PRIORITY[odor_key]]
        match: Optional[str] = None
        for candidate in candidates:
            candidate_columns = normalized_to_columns.get(candidate, [])
            candidate_columns = [c for c in candidate_columns if c not in used_columns]
            if candidate_columns:
                # Deterministic tie-break: shortest then lexicographic.
                candidate_columns = sorted(
                    candidate_columns,
                    key=lambda c: (len(normalize_label(c)), str(c).lower()),
                )
                match = candidate_columns[0]
                break
        if match is None:
            raise ValueError(
                "Could not resolve CSV column for odor '{0}'. Available columns: {1}".format(
                    odor_key, list(columns)
                )
            )
        colmap[odor_key] = match
        used_columns.add(match)
    return colmap


def infer_training_conditions(
    df: pd.DataFrame,
    control_row: str = DEFAULT_CONTROL_ROW,
) -> List[str]:
    """Infer trained conditions from CSV rows (opto_* except control)."""
    resolved_control = _resolve_row_label(df.index, control_row)
    conditions: List[str] = []
    for row_name in df.index:
        row_name = str(row_name)
        if row_name == resolved_control:
            continue
        if row_name.lower().startswith("opto_"):
            conditions.append(row_name)

    # Fallback: if no opto_* rows exist, use every non-control row.
    if not conditions:
        conditions = [str(row_name) for row_name in df.index if str(row_name) != resolved_control]
    return conditions


def resolve_condition_list(
    index_values: Iterable[str],
    conditions: Optional[Sequence[str]],
) -> List[str]:
    """Resolve condition labels against the dataframe index."""
    if conditions is None:
        return []
    resolved: List[str] = []
    for condition in conditions:
        resolved.append(_resolve_row_label(index_values, condition))
    return resolved


def filter_training_conditions(
    df: pd.DataFrame,
    *,
    control_row: str = DEFAULT_CONTROL_ROW,
    allowlist: Optional[Sequence[str]] = None,
    blocklist: Optional[Sequence[str]] = None,
) -> List[str]:
    """Apply allow/block filters to inferred training conditions."""
    all_conditions = infer_training_conditions(df, control_row=control_row)
    resolved_control = _resolve_row_label(df.index, control_row)

    if allowlist:
        resolved_allow = resolve_condition_list(df.index, allowlist)
        unknown = [c for c in resolved_allow if c not in all_conditions]
        if unknown:
            raise ValueError(
                "Allowlisted conditions not in training set: {0}".format(unknown)
            )
        filtered = list(resolved_allow)
    else:
        filtered = list(all_conditions)

    if blocklist:
        resolved_block = set(resolve_condition_list(df.index, blocklist))
        filtered = [c for c in filtered if c not in resolved_block]

    filtered = [c for c in filtered if c != resolved_control]
    if not filtered:
        raise ValueError("No training conditions remain after filtering.")
    return filtered

def load_per_matrix(
    csv_path: str,
    odors: Optional[Sequence[str]] = None,
    control_row: str = DEFAULT_CONTROL_ROW,
) -> Tuple[pd.DataFrame, Dict[str, str], List[str]]:
    """Load PER CSV, resolve 4 odor columns, and infer training conditions."""
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        raise ValueError("Behavior CSV is empty: {0}".format(csv_path))

    odor_keys = parse_odor_argument(odors)
    _ = _resolve_row_label(df.index, control_row)
    odor_colmap = _resolve_odor_columns(list(df.columns), odor_keys)
    conditions = infer_training_conditions(df, control_row=control_row)

    return df, odor_colmap, conditions


def extract_four_odors(df: pd.DataFrame, odor_colmap: Mapping[str, str]) -> pd.DataFrame:
    """Extract just the mapped odor columns in deterministic key order."""
    odor_keys = list(odor_colmap.keys())
    csv_columns = [odor_colmap[key] for key in odor_keys]
    df4 = df.loc[:, csv_columns].copy()
    df4.columns = odor_keys
    return df4


def compute_delta(
    df4: pd.DataFrame,
    control_row: str = DEFAULT_CONTROL_ROW,
    trained_conditions: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Compute delta PER per condition: trained - control."""
    resolved_control = _resolve_row_label(df4.index, control_row)
    control_vec = df4.loc[resolved_control]

    if trained_conditions is None:
        train_rows = [str(r) for r in df4.index if str(r) != resolved_control]
    else:
        train_rows = [_resolve_row_label(df4.index, row_name) for row_name in trained_conditions]

    trained_df = df4.loc[train_rows]
    delta_df = trained_df.subtract(control_vec, axis="columns")
    return delta_df


def mean_center_rows(delta_df: pd.DataFrame) -> pd.DataFrame:
    """Mean-center each condition's delta vector."""
    row_means = delta_df.mean(axis=1, skipna=True)
    return delta_df.sub(row_means, axis=0)


def classify_weight_signs(weights: np.ndarray, zero_eps: float = 1e-6) -> np.ndarray:
    """Classify signs as +1 / -1 / 0."""
    return np.where(weights > zero_eps, 1, np.where(weights < -zero_eps, -1, 0)).astype(int)


def _sign_symbol(sign_value: int) -> str:
    if sign_value > 0:
        return "+"
    if sign_value < 0:
        return "-"
    return "0"


def _make_sparse_regressor(
    model: str,
    alpha: float,
    random_state: int,
    l1_ratio: float,
):
    """Factory for deterministic sparse linear models."""
    if model == "lasso":
        return Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=100000,
            random_state=random_state,
            selection="cyclic",
        )
    if model == "elasticnet":
        return ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=100000,
            random_state=random_state,
            selection="cyclic",
        )
    raise ValueError("model must be 'lasso' or 'elasticnet', got '{0}'".format(model))


def _fit_sparse_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    alpha: float,
    random_state: int,
    l1_ratio: float,
    standardize: bool,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Fit sparse model and return coefficients in the original feature space.

    Returns:
        coef_raw: Feature weights compatible with raw X.
        intercept_raw: Intercept in raw X space.
        y_pred: In-sample predictions on raw X.
    """
    X_fit = X
    scaler = None
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    reg = _make_sparse_regressor(
        model=model, alpha=alpha, random_state=random_state, l1_ratio=l1_ratio
    )
    reg.fit(X_fit, y)
    coef = np.asarray(reg.coef_, dtype=np.float64).ravel()
    intercept = float(reg.intercept_)

    if standardize and scaler is not None:
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale[scale == 0.0] = 1.0
        coef = coef / scale
        intercept = intercept - float(np.dot(coef, np.asarray(scaler.mean_, dtype=np.float64)))

    y_pred = np.dot(X, coef) + intercept
    return coef, intercept, y_pred


def _select_alpha_loocv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    alpha_grid: Sequence[float],
    random_state: int,
    l1_ratio: float,
    standardize: bool,
    zero_eps: float,
    min_nonzero: int,
) -> Tuple[float, float, pd.DataFrame]:
    """Choose alpha by deterministic leave-one-out CV with sparse tie-break."""
    n_samples = X.shape[0]
    if n_samples < 2:
        raise ValueError("LOOCV requires at least 2 samples, got {0}".format(n_samples))

    records: List[Dict[str, float]] = []
    for alpha in alpha_grid:
        fold_mse: List[float] = []
        for hold_idx in range(n_samples):
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[hold_idx] = False
            coef, intercept, _ = _fit_sparse_model(
                X[train_mask],
                y[train_mask],
                model=model,
                alpha=float(alpha),
                random_state=random_state,
                l1_ratio=l1_ratio,
                standardize=standardize,
            )
            y_hat = float(np.dot(X[hold_idx], coef) + intercept)
            fold_mse.append((float(y[hold_idx]) - y_hat) ** 2)

        coef_full, _, _ = _fit_sparse_model(
            X,
            y,
            model=model,
            alpha=float(alpha),
            random_state=random_state,
            l1_ratio=l1_ratio,
            standardize=standardize,
        )
        n_nonzero = int(np.sum(np.abs(coef_full) > zero_eps))

        records.append(
            {
                "alpha": float(alpha),
                "loo_mse": float(np.mean(fold_mse)),
                "n_nonzero": float(n_nonzero),
            }
        )

    score_df = pd.DataFrame(records)
    if min_nonzero > 0:
        candidates = score_df[score_df["n_nonzero"] >= min_nonzero]
    else:
        candidates = score_df
    if candidates.empty:
        # Fall back to the most non-zero coefficients available.
        max_nonzero = score_df["n_nonzero"].max()
        candidates = score_df[score_df["n_nonzero"] == max_nonzero]
    best_idx = min(
        candidates.index,
        key=lambda i: (
            round(float(candidates.loc[i, "loo_mse"]), 12),
            int(candidates.loc[i, "n_nonzero"]),
            float(candidates.loc[i, "alpha"]),
        ),
    )
    best_alpha = float(score_df.loc[best_idx, "alpha"])
    best_loo_mse = float(score_df.loc[best_idx, "loo_mse"])
    return best_alpha, best_loo_mse, score_df


def fit_delta_weights_per_condition(
    odors: Sequence[str],
    y_centered: pd.DataFrame,
    feature_builder: Callable[[Sequence[str]], Tuple[np.ndarray, Sequence[str], dict]],
    *,
    model: str = "lasso",
    alpha_grid: Optional[Sequence[float]] = None,
    l1_ratio: float = 0.5,
    standardize: bool = True,
    random_state: int = 0,
    zero_eps: float = 1e-6,
    min_nonzero: int = 1,
) -> dict:
    """
    Fit one sparse delta-weight model per training condition.

    Args:
        odors: Ordered odor keys matching y_centered columns.
        y_centered: DataFrame indexed by condition, columns=odors.
        feature_builder: Callable returning (X, feature_names, meta).
        model: "lasso" or "elasticnet".
        alpha_grid: Fixed deterministic alphas to evaluate by LOOCV.
        l1_ratio: ElasticNet l1_ratio.
        standardize: Standardize X before fitting.
        random_state: Deterministic seed.
        zero_eps: Sign classification threshold.
        min_nonzero: Prefer solutions with at least this many non-zero weights.
    """
    if model not in {"lasso", "elasticnet"}:
        raise ValueError("model must be 'lasso' or 'elasticnet', got '{0}'".format(model))

    if alpha_grid is None:
        alpha_grid = np.logspace(-4, 1, 60)
    alpha_grid = [float(a) for a in alpha_grid]

    feature_output = feature_builder(list(odors))
    if not isinstance(feature_output, tuple):
        raise TypeError("feature_builder must return a tuple, got {0}".format(type(feature_output)))
    if len(feature_output) == 2:
        X, feature_names = feature_output
        feature_meta = {}
    elif len(feature_output) == 3:
        X, feature_names, feature_meta = feature_output
    else:
        raise ValueError(
            "feature_builder must return (X, feature_names[, meta]); got {0} values".format(
                len(feature_output)
            )
        )

    X = np.asarray(X, dtype=np.float64)
    feature_names = list(feature_names)
    if X.shape[0] != len(odors):
        raise ValueError(
            "Feature matrix has {0} rows but {1} odors were provided".format(X.shape[0], len(odors))
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            "Feature matrix has {0} columns but {1} feature names were provided".format(
                X.shape[1], len(feature_names)
            )
        )

    condition_results: Dict[str, dict] = {}
    for condition_name, y_series in y_centered.iterrows():
        y_full = pd.to_numeric(y_series, errors="coerce").to_numpy(dtype=np.float64)
        valid_mask = np.isfinite(y_full)
        if valid_mask.sum() < 2:
            raise ValueError(
                "Condition '{0}' has fewer than 2 valid odor points after filtering".format(
                    condition_name
                )
            )

        X_used = X[valid_mask]
        y_used = y_full[valid_mask]
        odor_subset = [odors[i] for i in range(len(odors)) if valid_mask[i]]

        best_alpha, best_loo_mse, alpha_scores = _select_alpha_loocv(
            X_used,
            y_used,
            model=model,
            alpha_grid=alpha_grid,
            random_state=random_state,
            l1_ratio=l1_ratio,
            standardize=standardize,
            zero_eps=zero_eps,
            min_nonzero=min_nonzero,
        )

        weights, intercept, y_pred_used = _fit_sparse_model(
            X_used,
            y_used,
            model=model,
            alpha=best_alpha,
            random_state=random_state,
            l1_ratio=l1_ratio,
            standardize=standardize,
        )
        signs = classify_weight_signs(weights, zero_eps=zero_eps)
        n_nonzero = int(np.sum(signs != 0))

        y_pred_full = np.full(len(odors), np.nan, dtype=np.float64)
        y_pred_full[valid_mask] = y_pred_used
        mse_in = float(mean_squared_error(y_used, y_pred_used))

        condition_results[str(condition_name)] = {
            "condition": str(condition_name),
            "alpha": best_alpha,
            "loo_mse": best_loo_mse,
            "in_sample_mse": mse_in,
            "weights": weights,
            "intercept": float(intercept),
            "weight_signs": signs,
            "n_nonzero": n_nonzero,
            "valid_odor_mask": valid_mask,
            "odor_subset": odor_subset,
            "y_used": y_used,
            "y_pred_used": y_pred_used,
            "y_pred_full": y_pred_full,
            "alpha_scores": alpha_scores.sort_values("alpha").reset_index(drop=True),
        }

    return {
        "odors": list(odors),
        "feature_matrix": X,
        "feature_names": feature_names,
        "feature_meta": feature_meta,
        "model": model,
        "random_state": random_state,
        "zero_eps": zero_eps,
        "standardize": standardize,
        "alpha_grid": alpha_grid,
        "min_nonzero": min_nonzero,
        "condition_results": condition_results,
    }


def build_delta_debug_table(
    df4: pd.DataFrame,
    delta_raw: pd.DataFrame,
    delta_centered: pd.DataFrame,
    *,
    control_row: str = DEFAULT_CONTROL_ROW,
    odor_colmap: Optional[Mapping[str, str]] = None,
) -> pd.DataFrame:
    """Build long-form debug table with raw/control/delta values."""
    resolved_control = _resolve_row_label(df4.index, control_row)
    control_vec = df4.loc[resolved_control]

    rows: List[dict] = []
    for condition_name in delta_raw.index:
        raw_row = delta_raw.loc[condition_name]
        centered_row = delta_centered.loc[condition_name]
        row_mean = float(raw_row.mean(skipna=True))
        for odor_key in delta_raw.columns:
            rows.append(
                {
                    "condition": str(condition_name),
                    "odor_key": str(odor_key),
                    "odor_csv_column": (
                        odor_colmap.get(odor_key, odor_key) if odor_colmap is not None else odor_key
                    ),
                    "per_trained": float(df4.loc[condition_name, odor_key]),
                    "per_control": float(control_vec[odor_key]),
                    "delta_raw": float(raw_row[odor_key]),
                    "delta_raw_mean": row_mean,
                    "delta_centered": float(centered_row[odor_key]),
                }
            )
    return pd.DataFrame(rows)


def export_reports(
    outdir: str,
    *,
    df4: pd.DataFrame,
    control_row: str,
    odor_colmap: Mapping[str, str],
    delta_raw: pd.DataFrame,
    delta_centered: pd.DataFrame,
    fit_results: dict,
    zero_eps: float = 1e-6,
) -> Dict[str, List[str]]:
    """Export per-condition delta vectors, weights, and contribution rankings."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    resolved_control = _resolve_row_label(df4.index, control_row)
    control_vec = df4.loc[resolved_control]

    mapping_records = []
    for odor_key in odor_colmap:
        mapping_records.append(
            {
                "odor_key": odor_key,
                "csv_column": odor_colmap[odor_key],
                "door_name": ODOR_KEY_TO_DOOR[odor_key],
            }
        )
    mapping_df = pd.DataFrame(mapping_records)
    mapping_path = out_path / "odor_mapping_report.csv"
    mapping_df.to_csv(mapping_path, index=False)

    debug_df = build_delta_debug_table(
        df4,
        delta_raw,
        delta_centered,
        control_row=resolved_control,
        odor_colmap=odor_colmap,
    )
    debug_path = out_path / "delta_debug_table.csv"
    debug_df.to_csv(debug_path, index=False)

    weight_files: List[str] = []
    raw_files: List[str] = []
    centered_files: List[str] = []
    contrib_files: List[str] = []
    alpha_files: List[str] = []

    feature_names = fit_results["feature_names"]
    feature_matrix = np.asarray(fit_results["feature_matrix"], dtype=np.float64)
    odor_keys = list(fit_results["odors"])
    condition_results = fit_results["condition_results"]

    for condition_name in delta_raw.index:
        condition_name = str(condition_name)
        condition_slug = sanitize_filename_token(condition_name)
        if condition_name not in condition_results:
            raise ValueError("No fitted weights found for condition '{0}'".format(condition_name))

        raw_table = pd.DataFrame(
            {
                "condition": condition_name,
                "odor_key": list(delta_raw.columns),
                "odor_csv_column": [odor_colmap[k] for k in delta_raw.columns],
                "per_trained": [float(df4.loc[condition_name, k]) for k in delta_raw.columns],
                "per_control": [float(control_vec[k]) for k in delta_raw.columns],
                "delta_per": [float(delta_raw.loc[condition_name, k]) for k in delta_raw.columns],
            }
        )
        raw_path = out_path / "delta_per_raw_{0}.csv".format(condition_slug)
        raw_table.to_csv(raw_path, index=False)
        raw_files.append(str(raw_path))

        centered_table = pd.DataFrame(
            {
                "condition": condition_name,
                "odor_key": list(delta_centered.columns),
                "odor_csv_column": [odor_colmap[k] for k in delta_centered.columns],
                "delta_per_raw": [
                    float(delta_raw.loc[condition_name, k]) for k in delta_centered.columns
                ],
                "delta_per_centered": [
                    float(delta_centered.loc[condition_name, k]) for k in delta_centered.columns
                ],
            }
        )
        centered_table["centered_mean"] = float(
            centered_table["delta_per_centered"].mean(skipna=True)
        )
        centered_path = out_path / "delta_per_centered_{0}.csv".format(condition_slug)
        centered_table.to_csv(centered_path, index=False)
        centered_files.append(str(centered_path))

        result = condition_results[condition_name]
        weights = np.asarray(result["weights"], dtype=np.float64)
        signs = classify_weight_signs(weights, zero_eps=zero_eps)
        weights_df = pd.DataFrame(
            {
                "feature": feature_names,
                "delta_weight": weights,
                "sign": [_sign_symbol(int(s)) for s in signs],
                "abs_delta_weight": np.abs(weights),
            }
        )
        weights_df = weights_df.sort_values("abs_delta_weight", ascending=False).reset_index(drop=True)
        weights_df["rank"] = np.arange(1, len(weights_df) + 1)
        weights_df["alpha"] = float(result["alpha"])
        weights_df["loo_mse"] = float(result["loo_mse"])

        weights_path = out_path / "delta_weights_{0}.csv".format(condition_slug)
        weights_df.to_csv(weights_path, index=False)
        weight_files.append(str(weights_path))

        alpha_scores = result["alpha_scores"].copy()
        alpha_path = out_path / "alpha_loocv_{0}.csv".format(condition_slug)
        alpha_scores.to_csv(alpha_path, index=False)
        alpha_files.append(str(alpha_path))

        for odor_index, odor_key in enumerate(odor_keys):
            x_vec = feature_matrix[odor_index, :]
            contribution = x_vec * weights
            csigns = classify_weight_signs(contribution, zero_eps=zero_eps)
            contrib_df = pd.DataFrame(
                {
                    "feature": feature_names,
                    "x_value": x_vec,
                    "delta_weight": weights,
                    "contribution": contribution,
                    "contribution_sign": [_sign_symbol(int(s)) for s in csigns],
                    "abs_contribution": np.abs(contribution),
                }
            )
            contrib_df = contrib_df.sort_values("abs_contribution", ascending=False).reset_index(
                drop=True
            )
            contrib_df["rank"] = np.arange(1, len(contrib_df) + 1)
            odor_slug = sanitize_filename_token(odor_colmap.get(odor_key, odor_key))
            contrib_path = out_path / "contrib_{0}_{1}.csv".format(condition_slug, odor_slug)
            contrib_df.to_csv(contrib_path, index=False)
            contrib_files.append(str(contrib_path))

    manifest = {
        "control_row": resolved_control,
        "odors": list(odor_keys),
        "odor_column_map": dict(odor_colmap),
        "door_odor_map": {k: ODOR_KEY_TO_DOOR[k] for k in odor_colmap},
        "model": fit_results.get("model"),
        "standardize": bool(fit_results.get("standardize", True)),
        "random_state": int(fit_results.get("random_state", 0)),
        "alpha_grid": [float(a) for a in fit_results.get("alpha_grid", [])],
        "n_conditions_fit": int(len(condition_results)),
    }
    manifest_path = out_path / "run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info("Wrote plasticity delta outputs to %s", out_path)
    return {
        "mapping_report": [str(mapping_path)],
        "debug_table": [str(debug_path)],
        "delta_raw": raw_files,
        "delta_centered": centered_files,
        "delta_weights": weight_files,
        "contributions": contrib_files,
        "alpha_scores": alpha_files,
        "manifest": [str(manifest_path)],
    }


def load_baseline_weights(baseline_outdir: str) -> Tuple[pd.DataFrame, dict]:
    """Load baseline weights.csv and model_summary.json."""
    base_path = Path(baseline_outdir)
    weights_path = base_path / "weights.csv"
    summary_path = base_path / "model_summary.json"
    if not weights_path.exists():
        raise FileNotFoundError("Baseline weights.csv not found: {0}".format(weights_path))
    if not summary_path.exists():
        raise FileNotFoundError("Baseline model_summary.json not found: {0}".format(summary_path))

    weights_df = pd.read_csv(weights_path)
    if "receptor" in weights_df.columns:
        feature_col = "receptor"
    elif "feature" in weights_df.columns:
        feature_col = "feature"
    else:
        raise ValueError(
            "Baseline weights.csv missing feature column (expected 'receptor' or 'feature')."
        )
    if "weight" not in weights_df.columns:
        raise ValueError("Baseline weights.csv missing 'weight' column.")

    baseline_df = weights_df[[feature_col, "weight"]].copy()
    baseline_df = baseline_df.rename(columns={feature_col: "feature", "weight": "baseline_w"})

    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)

    return baseline_df, summary


def validate_feature_alignment(
    baseline_features: Sequence[str],
    delta_features: Sequence[str],
) -> None:
    """Ensure baseline and delta features match exactly."""
    baseline_set = set(baseline_features)
    delta_set = set(delta_features)
    missing = sorted(delta_set - baseline_set)
    extra = sorted(baseline_set - delta_set)
    if missing or extra:
        parts = []
        if missing:
            parts.append("missing in baseline: {0}".format(missing))
        if extra:
            parts.append("extra in baseline: {0}".format(extra))
        raise ValueError("Feature alignment failed ({0}).".format("; ".join(parts)))


def _rank_by_abs(weights: pd.Series, features: pd.Series) -> pd.Series:
    """Deterministic rank by absolute weight (desc), tie-broken by feature name."""
    temp = pd.DataFrame({"feature": features, "weight": weights})
    temp["abs_weight"] = temp["weight"].abs()
    temp = temp.sort_values(["abs_weight", "feature"], ascending=[False, True]).reset_index(drop=True)
    temp["rank"] = np.arange(1, len(temp) + 1)
    return temp.set_index("feature")["rank"]


def build_weight_comparison_table(
    *,
    baseline_df: pd.DataFrame,
    delta_features: Sequence[str],
    delta_weights: np.ndarray,
    zero_eps: float = 1e-6,
) -> pd.DataFrame:
    """Create baseline vs delta comparison table for one condition."""
    baseline_df = baseline_df.copy()
    validate_feature_alignment(baseline_df["feature"], delta_features)

    delta_df = pd.DataFrame(
        {
            "feature": list(delta_features),
            "delta_w": np.asarray(delta_weights, dtype=np.float64),
        }
    )
    merged = delta_df.merge(baseline_df, on="feature", how="left")
    if merged["baseline_w"].isna().any():
        missing = merged.loc[merged["baseline_w"].isna(), "feature"].tolist()
        raise ValueError("Baseline weights missing for features: {0}".format(missing))

    merged["combined_w"] = merged["baseline_w"] + merged["delta_w"]
    merged["abs_delta"] = merged["delta_w"].abs()

    merged["baseline_sign"] = [
        _sign_symbol(int(s))
        for s in classify_weight_signs(merged["baseline_w"].to_numpy(), zero_eps=zero_eps)
    ]
    merged["delta_sign"] = [
        _sign_symbol(int(s))
        for s in classify_weight_signs(merged["delta_w"].to_numpy(), zero_eps=zero_eps)
    ]
    merged["combined_sign"] = [
        _sign_symbol(int(s))
        for s in classify_weight_signs(merged["combined_w"].to_numpy(), zero_eps=zero_eps)
    ]

    rank_baseline = _rank_by_abs(merged["baseline_w"], merged["feature"])
    rank_delta = _rank_by_abs(merged["delta_w"], merged["feature"])
    rank_combined = _rank_by_abs(merged["combined_w"], merged["feature"])

    merged["rank_baseline"] = merged["feature"].map(rank_baseline).astype(int)
    merged["rank_delta"] = merged["feature"].map(rank_delta).astype(int)
    merged["rank_combined"] = merged["feature"].map(rank_combined).astype(int)
    merged["rank_change"] = merged["rank_baseline"] - merged["rank_combined"]

    merged = merged.sort_values(["abs_delta", "feature"], ascending=[False, True]).reset_index(drop=True)
    return merged[
        [
            "feature",
            "baseline_w",
            "delta_w",
            "combined_w",
            "baseline_sign",
            "delta_sign",
            "combined_sign",
            "abs_delta",
            "rank_baseline",
            "rank_delta",
            "rank_combined",
            "rank_change",
        ]
    ]


def build_reconstruction_table(
    *,
    df4: pd.DataFrame,
    delta_raw: pd.DataFrame,
    delta_centered: pd.DataFrame,
    fit_results: dict,
    condition_name: str,
    control_row: str,
    centered: bool,
) -> pd.DataFrame:
    """Build reconstruction table for one condition."""
    odor_keys = list(fit_results["odors"])
    control_row = _resolve_row_label(df4.index, control_row)
    condition_name = _resolve_row_label(df4.index, condition_name)

    control_vec = df4.loc[control_row]
    trained_vec = df4.loc[condition_name]
    delta_raw_vec = delta_raw.loc[condition_name]
    delta_centered_vec = delta_centered.loc[condition_name]
    delta_raw_mean = float(delta_raw_vec.mean(skipna=True))

    condition_results = fit_results["condition_results"][condition_name]
    delta_pred = np.asarray(condition_results["y_pred_full"], dtype=np.float64)
    if centered:
        delta_pred_raw = delta_pred + delta_raw_mean
    else:
        delta_pred_raw = delta_pred

    trained_pred = control_vec.to_numpy(dtype=np.float64) + delta_pred_raw

    rows = []
    for i, odor_key in enumerate(odor_keys):
        rows.append(
            {
                "odor": odor_key,
                "control_per_true": float(control_vec[odor_key]),
                "trained_per_true": float(trained_vec[odor_key]),
                "delta_per_true": float(delta_raw_vec[odor_key]),
                "delta_per_centered_true": float(delta_centered_vec[odor_key]),
                "delta_per_pred": float(delta_pred[i]) if np.isfinite(delta_pred[i]) else np.nan,
                "delta_per_pred_raw": float(delta_pred_raw[i]) if np.isfinite(delta_pred_raw[i]) else np.nan,
                "trained_per_pred": float(trained_pred[i]) if np.isfinite(trained_pred[i]) else np.nan,
                "delta_raw_mean": delta_raw_mean,
            }
        )
    return pd.DataFrame(rows)


def export_baseline_comparison_reports(
    outdir: str,
    *,
    baseline_df: pd.DataFrame,
    df4: pd.DataFrame,
    delta_raw: pd.DataFrame,
    delta_centered: pd.DataFrame,
    fit_results: dict,
    control_row: str,
    centered: bool,
    zero_eps: float = 1e-6,
) -> Dict[str, List[str]]:
    """Export weight comparison and reconstruction reports per condition."""
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    comparison_paths: List[str] = []
    reconstruction_paths: List[str] = []

    for condition_name, result in fit_results["condition_results"].items():
        weights = np.asarray(result["weights"], dtype=np.float64)
        comparison_df = build_weight_comparison_table(
            baseline_df=baseline_df,
            delta_features=fit_results["feature_names"],
            delta_weights=weights,
            zero_eps=zero_eps,
        )
        comp_path = out_path / "weight_comparison_{0}.csv".format(
            condition_filename_token(condition_name)
        )
        comparison_df.to_csv(comp_path, index=False)
        comparison_paths.append(str(comp_path))

        recon_df = build_reconstruction_table(
            df4=df4,
            delta_raw=delta_raw,
            delta_centered=delta_centered,
            fit_results=fit_results,
            condition_name=condition_name,
            control_row=control_row,
            centered=centered,
        )
        recon_path = out_path / "reconstruction_{0}.csv".format(
            condition_filename_token(condition_name)
        )
        recon_df.to_csv(recon_path, index=False)
        reconstruction_paths.append(str(recon_path))

    return {
        "weight_comparisons": comparison_paths,
        "reconstructions": reconstruction_paths,
    }
