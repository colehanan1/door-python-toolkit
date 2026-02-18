"""
Multi-Condition Leave-One-Odor-Out Regression
==============================================

Fit and evaluate one regression model per condition using LOOCV
(train on N-1 odors, predict the held-out), where:

* Control (``opto_AIR``) uses **raw PER**, mean-centered across odors.
* Trained conditions use **ΔPER = trained − control**, then mean-centered.

Feature intersection across all odors is enforced so the feature set is
identical for every fold and every condition.
"""

import json
import logging
import re
from pathlib import Path
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NON_ODOR_COLUMNS = frozenset({"dataset", "index", "unnamed: 0"})


def _normalize(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(label).lower())


def _condition_slug(name: str) -> str:
    text = str(name).strip()
    if re.match(r"^[A-Za-z0-9._-]+$", text):
        return text
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text.lower())).strip("_")


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------


def load_per_csv(csv_path: str) -> pd.DataFrame:
    """Load PER CSV with condition rows and odor columns.

    Returns a DataFrame indexed by condition (dataset), with odor columns.
    """
    df = pd.read_csv(csv_path, index_col=0)
    if df.empty:
        raise ValueError("PER CSV is empty: {0}".format(csv_path))
    return df


def detect_odor_columns(
    df: pd.DataFrame,
    max_missing_frac: float = 0.5,
) -> List[str]:
    """Detect odor columns: non-metadata columns with sufficient data.

    Columns where more than *max_missing_frac* of rows are NaN are
    excluded (e.g. ``AIR``, ``Ethyl_Butyrate_(6-Training)``).

    Raises if the count is not exactly 7.
    """
    odors: List[str] = []
    n_rows = len(df)
    for col in df.columns:
        if _normalize(col) in _NON_ODOR_COLUMNS:
            continue
        n_missing = int(pd.to_numeric(df[col], errors="coerce").isna().sum())
        if n_rows > 0 and n_missing / n_rows > max_missing_frac:
            logger.info(
                "Excluding column '%s' (%d/%d missing)", col, n_missing, n_rows
            )
            continue
        odors.append(col)

    logger.info("Detected odor columns (%d): %s", len(odors), odors)
    if len(odors) < 3:
        raise ValueError(
            "Expected at least 3 odor columns but found {0}: {1}".format(
                len(odors), odors
            )
        )
    return odors


# ---------------------------------------------------------------------------
# Control imputation
# ---------------------------------------------------------------------------

# Rows whose values are averaged to fill missing control (opto_AIR) entries.
_CONTROL_IMPUTE_ROWS = ("hex_control", "Benz_control", "EB_control")


def impute_control_missing(
    df: pd.DataFrame,
    control_row: str,
    impute_from: Sequence[str] = _CONTROL_IMPUTE_ROWS,
) -> pd.DataFrame:
    """Fill NaN entries in the control row by averaging matching rows.

    For each odor column where the control row is NaN, the value is set to
    the mean of that column across the *impute_from* rows (only those
    present in the DataFrame).

    Returns a modified copy of the DataFrame.
    """
    df = df.copy()
    if control_row not in df.index:
        return df

    available = [r for r in impute_from if r in df.index]
    if not available:
        logger.warning(
            "No imputation rows found in CSV (looked for %s)", list(impute_from)
        )
        return df

    for col in df.columns:
        val = df.loc[control_row, col]
        try:
            if pd.isna(val):
                source_vals = pd.to_numeric(
                    df.loc[available, col], errors="coerce"
                ).dropna()
                if len(source_vals) > 0:
                    imputed = float(source_vals.mean())
                    df.loc[control_row, col] = imputed
                    logger.info(
                        "Imputed %s[%s] = mean(%s) = %.6f",
                        control_row, col, list(source_vals.index), imputed,
                    )
        except (TypeError, ValueError):
            pass
    return df


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------


def build_X_for_odors(
    odors: List[str],
    feature_builder: Callable[
        [List[str]], Tuple[np.ndarray, List[str], dict]
    ],
    intersection: bool = True,
) -> Tuple[np.ndarray, List[str], dict]:
    """Build a feature matrix for the given odors using *feature_builder*.

    When *intersection* is True the builder is expected to already restrict
    to the intersection feature set (``feature_set='intersection'``).  This
    function validates that the returned matrix has consistent shape.
    """
    result = feature_builder(odors)
    if not isinstance(result, tuple) or len(result) < 2:
        raise TypeError(
            "feature_builder must return (X, feature_names[, meta])"
        )

    if len(result) == 2:
        X, feature_names = result
        meta = {}
    else:
        X, feature_names, meta = result

    X = np.asarray(X, dtype=np.float64)
    feature_names = list(feature_names)

    if X.shape[0] != len(odors):
        raise ValueError(
            "Feature matrix rows ({0}) != number of odors ({1})".format(
                X.shape[0], len(odors)
            )
        )
    if X.shape[1] != len(feature_names):
        raise ValueError(
            "Feature matrix cols ({0}) != feature names ({1})".format(
                X.shape[1], len(feature_names)
            )
        )
    if X.shape[1] == 0:
        raise ValueError("Feature intersection produced 0 features.")

    meta["intersection_enforced"] = intersection
    return X, feature_names, meta


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def make_targets(
    df: pd.DataFrame,
    odors: List[str],
    control_row: str,
    condition_row: str,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Build the y vector and its mean-centered version.

    For control (``condition_row == control_row``):
        y_raw = PER_control across *odors*
        y_centered = y_raw − mean(y_raw)

    For trained conditions:
        y_raw = PER_trained − PER_control   (i.e. ΔPER)
        y_centered = y_raw − mean(y_raw)

    Returns (y_raw, y_centered, centering_mean).
    """
    if control_row not in df.index:
        raise ValueError(
            "Control row '{0}' not found in CSV index".format(control_row)
        )
    if condition_row not in df.index:
        raise ValueError(
            "Condition row '{0}' not found in CSV index".format(condition_row)
        )

    control_vals = pd.to_numeric(
        df.loc[control_row, odors], errors="coerce"
    ).to_numpy(dtype=np.float64)

    if condition_row == control_row:
        # Control condition: use raw PER
        y_raw = control_vals.copy()
    else:
        # Trained condition: ΔPER = trained − control
        trained_vals = pd.to_numeric(
            df.loc[condition_row, odors], errors="coerce"
        ).to_numpy(dtype=np.float64)
        y_raw = trained_vals - control_vals

    centering_mean = float(np.nanmean(y_raw))
    y_centered = y_raw - centering_mean
    return y_raw, y_centered, centering_mean


# ---------------------------------------------------------------------------
# Sparse regressor factory (reuses pattern from plasticity_delta_fit)
# ---------------------------------------------------------------------------


def _make_regressor(
    model: str,
    alpha: float,
    random_state: int,
    l1_ratio: float = 0.5,
):
    if model == "lasso":
        return Lasso(
            alpha=float(alpha),
            fit_intercept=True,
            max_iter=100_000,
            random_state=random_state,
            selection="cyclic",
        )
    if model == "elasticnet":
        return ElasticNet(
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            fit_intercept=True,
            max_iter=100_000,
            random_state=random_state,
            selection="cyclic",
        )
    raise ValueError("model must be 'lasso' or 'elasticnet', got '{0}'".format(model))


def _fit_one(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    alpha: float,
    random_state: int,
    l1_ratio: float,
    standardize: bool,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fit sparse model; return (coef_raw, intercept_raw, y_pred)."""
    scaler = None
    X_fit = X
    if standardize:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X)

    reg = _make_regressor(model, alpha, random_state, l1_ratio)
    reg.fit(X_fit, y)
    coef = np.asarray(reg.coef_, dtype=np.float64).ravel()
    intercept = float(reg.intercept_)

    # Map back to raw feature space
    if standardize and scaler is not None:
        scale = np.asarray(scaler.scale_, dtype=np.float64)
        scale[scale == 0.0] = 1.0
        coef = coef / scale
        intercept = intercept - float(
            np.dot(coef, np.asarray(scaler.mean_, dtype=np.float64))
        )

    y_pred = np.dot(X, coef) + intercept
    return coef, intercept, y_pred


# ---------------------------------------------------------------------------
# Alpha selection via inner LOOCV (same pattern as plasticity_delta_fit)
# ---------------------------------------------------------------------------


def _select_alpha(
    X: np.ndarray,
    y: np.ndarray,
    *,
    model: str,
    alpha_grid: Sequence[float],
    l1_ratio: float,
    random_state: int,
    standardize: bool,
    zero_eps: float,
    min_nonzero: int,
) -> Tuple[float, float, pd.DataFrame]:
    """Inner LOOCV alpha selection with sparse tie-break."""
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need >= 2 samples for LOOCV, got {0}".format(n))

    records: List[dict] = []
    for alpha in alpha_grid:
        fold_se: List[float] = []
        for hold in range(n):
            mask = np.ones(n, dtype=bool)
            mask[hold] = False
            coef, intercept, _ = _fit_one(
                X[mask], y[mask],
                model=model, alpha=float(alpha),
                random_state=random_state, l1_ratio=l1_ratio,
                standardize=standardize,
            )
            y_hat = float(np.dot(X[hold], coef) + intercept)
            fold_se.append((float(y[hold]) - y_hat) ** 2)

        coef_full, _, _ = _fit_one(
            X, y, model=model, alpha=float(alpha),
            random_state=random_state, l1_ratio=l1_ratio,
            standardize=standardize,
        )
        n_nz = int(np.sum(np.abs(coef_full) > zero_eps))
        records.append({
            "alpha": float(alpha),
            "loo_mse": float(np.mean(fold_se)),
            "n_nonzero": n_nz,
        })

    score_df = pd.DataFrame(records)
    cands = score_df[score_df["n_nonzero"] >= min_nonzero] if min_nonzero > 0 else score_df
    if cands.empty:
        mx = score_df["n_nonzero"].max()
        cands = score_df[score_df["n_nonzero"] == mx]

    best_idx = min(
        cands.index,
        key=lambda i: (
            round(float(cands.loc[i, "loo_mse"]), 12),
            int(cands.loc[i, "n_nonzero"]),
            float(cands.loc[i, "alpha"]),
        ),
    )
    return (
        float(score_df.loc[best_idx, "alpha"]),
        float(score_df.loc[best_idx, "loo_mse"]),
        score_df,
    )


# ---------------------------------------------------------------------------
# Outer LOOCV (leave-one-odor-out) with per-fold weights
# ---------------------------------------------------------------------------


def loocv_fit_predict(
    X: np.ndarray,
    y_centered: np.ndarray,
    *,
    model: str = "lasso",
    alpha_grid: Optional[Sequence[float]] = None,
    l1_ratio: float = 0.5,
    l1_ratio_grid: Optional[Sequence[float]] = None,
    seed: int = 0,
    standardize: bool = True,
    zero_eps: float = 1e-6,
    min_nonzero: int = 1,
) -> Dict:
    """Run leave-one-odor-out CV returning fold predictions and weights.

    For each fold the held-out sample is one odor.  Alpha is chosen via
    inner LOOCV on the training subset of N-1 odors.

    Returns dict with keys:
        fold_preds, fold_true, fold_weights, fold_intercepts,
        selected_hparams, metrics
    """
    if alpha_grid is None:
        alpha_grid = list(np.logspace(-4, 1, 60))
    alpha_grid = [float(a) for a in alpha_grid]

    if l1_ratio_grid is not None:
        l1_ratio_grid = [float(r) for r in l1_ratio_grid]

    n = X.shape[0]
    n_feat = X.shape[1]

    # Identify valid (non-NaN) samples
    valid_mask = np.isfinite(y_centered)
    valid_indices = np.where(valid_mask)[0]
    n_valid = int(valid_mask.sum())

    if n_valid < 3:
        raise ValueError(
            "Need >= 3 valid (non-NaN) samples for LOOCV, got {0}".format(
                n_valid
            )
        )

    fold_preds = np.full(n, np.nan, dtype=np.float64)
    fold_true = y_centered.copy()
    fold_weights = np.full((n, n_feat), np.nan, dtype=np.float64)
    fold_intercepts = np.full(n, np.nan, dtype=np.float64)
    fold_alphas: List[Optional[float]] = [None] * n
    fold_l1_ratios: List[Optional[float]] = [None] * n
    fold_inner_mse: List[Optional[float]] = [None] * n

    # Only run folds for valid samples; training set also excludes NaN
    for hold in valid_indices:
        train_idx = [i for i in valid_indices if i != hold]
        X_train = X[train_idx]
        y_train = y_centered[train_idx]
        X_test = X[hold:hold + 1]

        # If l1_ratio_grid is given, search over it too
        best_alpha = alpha_grid[0]
        best_lr = l1_ratio
        best_inner = float("inf")

        lr_candidates = l1_ratio_grid if l1_ratio_grid else [l1_ratio]
        for lr in lr_candidates:
            alpha, inner_mse, _ = _select_alpha(
                X_train, y_train,
                model=model, alpha_grid=alpha_grid, l1_ratio=lr,
                random_state=seed, standardize=standardize,
                zero_eps=zero_eps, min_nonzero=min_nonzero,
            )
            if inner_mse < best_inner:
                best_inner = inner_mse
                best_alpha = alpha
                best_lr = lr

        # Fit on training set with best alpha
        coef, intercept, _ = _fit_one(
            X_train, y_train,
            model=model, alpha=best_alpha, random_state=seed,
            l1_ratio=best_lr, standardize=standardize,
        )
        y_hat = float(np.dot(X_test.ravel(), coef) + intercept)

        fold_preds[hold] = y_hat
        fold_weights[hold] = coef
        fold_intercepts[hold] = intercept
        fold_alphas[hold] = best_alpha
        fold_l1_ratios[hold] = best_lr
        fold_inner_mse[hold] = best_inner

    # Metrics on the held-out predictions (only valid folds)
    valid = np.isfinite(fold_preds) & np.isfinite(fold_true)
    n_valid_folds = int(valid.sum())
    mse = float(mean_squared_error(fold_true[valid], fold_preds[valid])) if n_valid_folds > 0 else float("nan")
    if valid.sum() >= 3:
        r_val, r_pval = pearsonr(fold_true[valid], fold_preds[valid])
    else:
        r_val, r_pval = float("nan"), float("nan")

    ss_res = float(np.sum((fold_true[valid] - fold_preds[valid]) ** 2))
    ss_tot = float(np.sum((fold_true[valid] - np.mean(fold_true[valid])) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "fold_preds": fold_preds,
        "fold_true": fold_true,
        "fold_weights": fold_weights,
        "fold_intercepts": fold_intercepts,
        "selected_hparams": {
            "fold_alphas": fold_alphas,
            "fold_l1_ratios": fold_l1_ratios,
            "fold_inner_mse": fold_inner_mse,
        },
        "metrics": {
            "loo_mse": mse,
            "loo_r": float(r_val),
            "loo_r_pval": float(r_pval),
            "loo_r2": r2,
            "n_folds": n,
            "n_valid_folds": n_valid_folds,
        },
    }


# ---------------------------------------------------------------------------
# Weight aggregation
# ---------------------------------------------------------------------------


def aggregate_weights(
    fold_weights: np.ndarray,
    zero_eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-fold weights into mean, std, and sign stability.

    Sign stability is the fraction of folds that agree on the sign of
    each feature (1.0 = all folds agree).

    Returns (mean_w, std_w, sign_stability).
    """
    mean_w = np.mean(fold_weights, axis=0)
    std_w = np.std(fold_weights, axis=0, ddof=0)

    n_folds = fold_weights.shape[0]
    signs = np.where(
        fold_weights > zero_eps, 1,
        np.where(fold_weights < -zero_eps, -1, 0),
    )
    # For each feature, fraction of folds matching the majority sign
    stability = np.zeros(fold_weights.shape[1], dtype=np.float64)
    for j in range(fold_weights.shape[1]):
        col_signs = signs[:, j]
        if n_folds == 0:
            stability[j] = 0.0
        else:
            counts = np.array([
                np.sum(col_signs == 1),
                np.sum(col_signs == -1),
                np.sum(col_signs == 0),
            ])
            stability[j] = float(counts.max()) / n_folds

    return mean_w, std_w, stability


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_condition(
    outdir: Path,
    condition: str,
    odors: List[str],
    feature_names: List[str],
    y_raw: np.ndarray,
    y_centered: np.ndarray,
    centering_mean: float,
    loocv_result: Dict,
    zero_eps: float = 1e-6,
) -> dict:
    """Export all per-condition artifacts and return summary dict."""
    slug = _condition_slug(condition)

    # predictions_<condition>.csv
    pred_df = pd.DataFrame({
        "odor": odors,
        "y_centered_true": loocv_result["fold_true"],
        "y_centered_pred": loocv_result["fold_preds"],
        "fold_id": list(range(len(odors))),
    })
    pred_path = outdir / "predictions_{0}.csv".format(slug)
    pred_df.to_csv(pred_path, index=False)

    # weights_folds_<condition>.csv
    fw = loocv_result["fold_weights"]
    fold_cols = {"feature": feature_names}
    for i in range(fw.shape[0]):
        fold_cols["fold{0}_w".format(i)] = fw[i]
    folds_df = pd.DataFrame(fold_cols)
    folds_path = outdir / "weights_folds_{0}.csv".format(slug)
    folds_df.to_csv(folds_path, index=False)

    # weights_mean_<condition>.csv
    mean_w, std_w, sign_stab = aggregate_weights(fw, zero_eps=zero_eps)
    mean_df = pd.DataFrame({
        "feature": feature_names,
        "mean_w": mean_w,
        "std_w": std_w,
        "sign_stability": sign_stab,
    })
    mean_path = outdir / "weights_mean_{0}.csv".format(slug)
    mean_df.to_csv(mean_path, index=False)

    # summary_<condition>.json
    metrics = loocv_result["metrics"]
    hparams = loocv_result["selected_hparams"]
    summary = {
        "condition": condition,
        "target_type": "raw_centered" if condition == "opto_AIR" else "delta_centered",
        "centering_mean": centering_mean,
        "y_raw": y_raw.tolist(),
        "y_centered": y_centered.tolist(),
        "hparams": {
            "fold_alphas": hparams["fold_alphas"],
            "fold_l1_ratios": hparams["fold_l1_ratios"],
        },
        "metrics": metrics,
        "feature_count": len(feature_names),
        "odor_list": odors,
    }
    summary_path = outdir / "summary_{0}.csv".format(slug)
    # Write as JSON despite .csv extension? No, use .json.
    summary_path = outdir / "summary_{0}.json".format(slug)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "condition": condition,
        "slug": slug,
        "predictions_csv": str(pred_path),
        "weights_folds_csv": str(folds_path),
        "weights_mean_csv": str(mean_path),
        "summary_json": str(summary_path),
        "metrics": metrics,
    }


def export_all(
    outdir: str,
    *,
    conditions: List[str],
    control_row: str,
    odors: List[str],
    feature_names: List[str],
    condition_data: Dict[str, dict],
    zero_eps: float = 1e-6,
) -> str:
    """Export outputs for all conditions and write conditions_overview.csv.

    *condition_data* maps condition name -> dict with keys:
        y_raw, y_centered, centering_mean, loocv_result

    Returns path to conditions_overview.csv.
    """
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)

    overview_rows: List[dict] = []

    for cond in conditions:
        cd = condition_data[cond]
        info = _export_condition(
            out_path,
            cond,
            odors,
            feature_names,
            cd["y_raw"],
            cd["y_centered"],
            cd["centering_mean"],
            cd["loocv_result"],
            zero_eps=zero_eps,
        )
        overview_rows.append({
            "condition": cond,
            "target_type": "raw_centered" if cond == control_row else "delta_centered",
            "loo_mse": info["metrics"]["loo_mse"],
            "loo_r": info["metrics"]["loo_r"],
            "loo_r2": info["metrics"]["loo_r2"],
            "n_folds": info["metrics"]["n_folds"],
            "feature_count": len(feature_names),
            "predictions_csv": info["predictions_csv"],
            "weights_folds_csv": info["weights_folds_csv"],
            "weights_mean_csv": info["weights_mean_csv"],
            "summary_json": info["summary_json"],
        })

    overview_df = pd.DataFrame(overview_rows)
    overview_path = out_path / "conditions_overview.csv"
    overview_df.to_csv(overview_path, index=False)
    logger.info("Wrote conditions_overview.csv to %s", overview_path)
    return str(overview_path)


# ---------------------------------------------------------------------------
# Top-level runner
# ---------------------------------------------------------------------------


def run_multicond_loocv(
    csv_path: str,
    control_row: str,
    conditions: List[str],
    feature_builder: Callable[
        [List[str]], Tuple[np.ndarray, List[str], dict]
    ],
    *,
    model: str = "lasso",
    alpha_grid: Optional[Sequence[float]] = None,
    l1_ratio: float = 0.5,
    l1_ratio_grid: Optional[Sequence[float]] = None,
    seed: int = 0,
    standardize: bool = True,
    zero_eps: float = 1e-6,
    min_nonzero: int = 1,
    outdir: str = "out/multicond_loocv",
) -> Dict:
    """Run the full multi-condition LOOCV pipeline.

    Steps:
        1. Load CSV, detect 7 odor columns.
        2. Build intersection feature matrix for all 7 odors.
        3. For each condition, compute targets and run LOOCV.
        4. Export all outputs.
    """
    # Step 1: Load, impute missing control values, and validate
    df = load_per_csv(csv_path)
    df = impute_control_missing(df, control_row)
    odors = detect_odor_columns(df)

    print("=== Multi-Condition LOOCV ===")
    print("CSV rows (conditions): {0}".format(list(df.index)))
    print("Detected odor columns ({0}): {1}".format(len(odors), odors))

    # Validate condition rows exist
    if control_row not in df.index:
        raise ValueError(
            "Control row '{0}' not in CSV. Available: {1}".format(
                control_row, list(df.index)
            )
        )
    for cond in conditions:
        if cond not in df.index:
            raise ValueError(
                "Condition '{0}' not in CSV. Available: {1}".format(
                    cond, list(df.index)
                )
            )

    # Step 2: Build feature matrix (intersection)
    print("\nBuilding intersection feature matrix for {0} odors...".format(
        len(odors)
    ))
    X, feature_names, feat_meta = build_X_for_odors(
        odors, feature_builder, intersection=True
    )
    print("Feature matrix shape: {0} (odors x features)".format(X.shape))
    print("Features: {0}".format(feature_names))

    # Step 3: Fit each condition
    condition_data: Dict[str, dict] = {}
    for cond in conditions:
        print("\n--- Condition: {0} ---".format(cond))
        y_raw, y_centered, cmean = make_targets(
            df, odors, control_row, cond
        )
        target_type = "raw PER (centered)" if cond == control_row else "ΔPER (centered)"
        print("  Target type: {0}".format(target_type))
        print("  y_raw:      {0}".format(np.round(y_raw, 4)))
        print("  centering mean: {0:.6f}".format(cmean))
        print("  y_centered: {0}".format(np.round(y_centered, 4)))

        result = loocv_fit_predict(
            X, y_centered,
            model=model, alpha_grid=alpha_grid, l1_ratio=l1_ratio,
            l1_ratio_grid=l1_ratio_grid, seed=seed,
            standardize=standardize, zero_eps=zero_eps,
            min_nonzero=min_nonzero,
        )
        m = result["metrics"]
        print("  LOO-MSE={0:.6f}  LOO-r={1:.4f}  LOO-R²={2:.4f}".format(
            m["loo_mse"], m["loo_r"], m["loo_r2"]
        ))

        condition_data[cond] = {
            "y_raw": y_raw,
            "y_centered": y_centered,
            "centering_mean": cmean,
            "loocv_result": result,
        }

    # Step 4: Export
    print("\nExporting to {0}...".format(outdir))
    overview_path = export_all(
        outdir,
        conditions=conditions,
        control_row=control_row,
        odors=odors,
        feature_names=feature_names,
        condition_data=condition_data,
        zero_eps=zero_eps,
    )
    print("Done. Overview: {0}".format(overview_path))

    return {
        "overview_csv": overview_path,
        "conditions": conditions,
        "odors": odors,
        "feature_names": feature_names,
        "condition_data": condition_data,
    }
