"""Odor-odor similarity with missing-data handling.

Supports two modes:
  - **pairwise** (default): each pair uses only features measured in both odors.
  - **global**: all pairs share the same feature set (intersection of all odors).

Every similarity value is accompanied by overlap count and coverage metadata.
A configurable ``min_overlap`` guard prevents silent garbage results.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as _cosine_dist
from scipy.stats import pearsonr, spearmanr

if TYPE_CHECKING:
    from door_toolkit.encoder import DoOREncoder

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_measured_mask(vec: np.ndarray) -> np.ndarray:
    """Return boolean mask: ``True`` where *vec* is not NaN.

    Parameters
    ----------
    vec : np.ndarray
        1-D response vector (may contain NaN for unmeasured features).

    Returns
    -------
    np.ndarray
        Boolean array of same shape.
    """
    return ~np.isnan(vec)


def _compute_similarity(a: np.ndarray, b: np.ndarray, method: str) -> float:
    """Compute similarity between two equal-length arrays (no NaN expected)."""
    if method == "cosine":
        # scipy cosine returns *distance*; convert to similarity
        d = _cosine_dist(a, b)
        return 1.0 - d
    elif method == "pearson":
        r, _ = pearsonr(a, b)
        return float(r)
    elif method == "spearman":
        r, _ = spearmanr(a, b)
        return float(r)
    else:
        raise ValueError(f"Unknown similarity method: {method!r}. "
                         f"Choose from 'cosine', 'pearson', 'spearman'.")


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def pairwise_similarity(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    method: str = "cosine",
    mask_mode: str = "pairwise",
    global_mask: Optional[np.ndarray] = None,
    min_overlap: int = 10,
) -> dict:
    """Compute similarity between two odor vectors with overlap reporting.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        1-D response vectors (may contain NaN).
    method : str
        ``'cosine'``, ``'pearson'``, or ``'spearman'``.
    mask_mode : str
        ``'pairwise'`` – use intersection of measured features for this pair.
        ``'global'`` – use *global_mask* (must be provided).
    global_mask : np.ndarray or None
        Boolean mask required when ``mask_mode='global'``.
    min_overlap : int
        Minimum number of shared measured features.  If fewer, the pair is
        skipped and similarity is NaN.

    Returns
    -------
    dict
        Keys: ``similarity`` (float | NaN), ``overlap_n`` (int),
        ``method``, ``mask_mode``, ``skipped`` (bool), ``reason`` (str | None).
    """
    if mask_mode == "pairwise":
        mask = get_measured_mask(vec_a) & get_measured_mask(vec_b)
    elif mask_mode == "global":
        if global_mask is None:
            raise ValueError("global_mask is required when mask_mode='global'")
        mask = global_mask
    else:
        raise ValueError(f"Unknown mask_mode: {mask_mode!r}")

    overlap_n = int(mask.sum())

    result = {
        "similarity": np.nan,
        "overlap_n": overlap_n,
        "method": method,
        "mask_mode": mask_mode,
        "skipped": False,
        "reason": None,
    }

    if overlap_n < min_overlap:
        result["skipped"] = True
        result["reason"] = (
            f"Overlap ({overlap_n}) < min_overlap ({min_overlap})"
        )
        warnings.warn(result["reason"], stacklevel=2)
        return result

    a_masked = vec_a[mask]
    b_masked = vec_b[mask]

    # Guard against zero-variance vectors (pearson/spearman would fail)
    if np.std(a_masked) == 0 or np.std(b_masked) == 0:
        result["skipped"] = True
        result["reason"] = "Zero variance in one or both vectors after masking"
        return result

    result["similarity"] = _compute_similarity(a_masked, b_masked, method)
    return result


def similarity_matrix(
    odor_names: list[str],
    encoder: "DoOREncoder",
    method: str = "cosine",
    mask_mode: str = "pairwise",
    min_overlap: int = 10,
) -> tuple[pd.DataFrame, dict]:
    """Compute full pairwise similarity matrix with metadata.

    Parameters
    ----------
    odor_names : list[str]
        Odorant names recognised by the encoder.
    encoder : DoOREncoder
        Encoder instance (used with ``fill_missing=NaN``).
    method : str
        Similarity method.
    mask_mode : str
        ``'pairwise'`` or ``'global'``.
    min_overlap : int
        Minimum overlap guard.

    Returns
    -------
    S : pd.DataFrame
        Symmetric odor × odor similarity matrix.
    meta : dict
        ``overlap_matrix`` (pd.DataFrame), ``coverage`` (pd.DataFrame),
        ``n_skipped`` (int), ``global_overlap_n`` (int | None).
    """
    n = len(odor_names)
    vectors = {
        name: encoder.encode(name, fill_missing=float("nan"))
        for name in odor_names
    }

    # Coverage per odor
    coverage_rows = []
    for name in odor_names:
        m = get_measured_mask(vectors[name])
        coverage_rows.append({
            "odor": name,
            "n_measured": int(m.sum()),
            "n_total": len(m),
            "coverage_pct": round(100.0 * m.sum() / len(m), 1),
        })
    coverage_df = pd.DataFrame(coverage_rows).set_index("odor")

    # Build global mask if needed
    global_mask = None
    global_overlap_n = None
    if mask_mode == "global":
        global_mask = np.ones(len(vectors[odor_names[0]]), dtype=bool)
        for name in odor_names:
            global_mask &= get_measured_mask(vectors[name])
        global_overlap_n = int(global_mask.sum())
        logger.info("Global intersection: %d features", global_overlap_n)

    sim_mat = np.full((n, n), np.nan)
    overlap_mat = np.zeros((n, n), dtype=int)
    n_skipped = 0

    for i in range(n):
        sim_mat[i, i] = 1.0
        overlap_mat[i, i] = int(get_measured_mask(vectors[odor_names[i]]).sum())
        for j in range(i + 1, n):
            res = pairwise_similarity(
                vectors[odor_names[i]],
                vectors[odor_names[j]],
                method=method,
                mask_mode=mask_mode,
                global_mask=global_mask,
                min_overlap=min_overlap,
            )
            sim_mat[i, j] = res["similarity"]
            sim_mat[j, i] = res["similarity"]
            overlap_mat[i, j] = res["overlap_n"]
            overlap_mat[j, i] = res["overlap_n"]
            if res["skipped"]:
                n_skipped += 1

    S = pd.DataFrame(sim_mat, index=odor_names, columns=odor_names)
    overlap_df = pd.DataFrame(overlap_mat, index=odor_names, columns=odor_names)

    meta = {
        "overlap_matrix": overlap_df,
        "coverage": coverage_df,
        "n_skipped": n_skipped,
        "global_overlap_n": global_overlap_n,
    }
    return S, meta


def find_similar(
    odor_names: list[str],
    query_odor: str,
    encoder: "DoOREncoder",
    top_k: int = 10,
    method: str = "cosine",
    mask_mode: str = "pairwise",
    min_overlap: int = 10,
) -> pd.DataFrame:
    """Find most similar odors to a query, with overlap reporting.

    Parameters
    ----------
    odor_names : list[str]
        Candidate odors (query is excluded automatically if present).
    query_odor : str
        Query odorant name.
    encoder : DoOREncoder
        Encoder instance.
    top_k : int
        Number of results to return.
    method, mask_mode, min_overlap
        Passed through to :func:`pairwise_similarity`.

    Returns
    -------
    pd.DataFrame
        Columns: ``target_odor``, ``similarity``, ``overlap_n``,
        ``target_coverage``, ``query_coverage``.
        Sorted by similarity descending (NaN-skipped rows at bottom).
    """
    query_vec = encoder.encode(query_odor, fill_missing=float("nan"))
    query_mask = get_measured_mask(query_vec)
    query_cov = int(query_mask.sum())

    targets = [o for o in odor_names if o.lower() != query_odor.lower()]

    # Build global mask if needed
    global_mask = None
    if mask_mode == "global":
        all_names = list(set([query_odor] + targets))
        global_mask = np.ones(len(query_vec), dtype=bool)
        for name in all_names:
            v = encoder.encode(name, fill_missing=float("nan"))
            global_mask &= get_measured_mask(v)

    rows = []
    for t in targets:
        t_vec = encoder.encode(t, fill_missing=float("nan"))
        t_mask = get_measured_mask(t_vec)
        res = pairwise_similarity(
            query_vec, t_vec,
            method=method,
            mask_mode=mask_mode,
            global_mask=global_mask,
            min_overlap=min_overlap,
        )
        rows.append({
            "target_odor": t,
            "similarity": res["similarity"],
            "overlap_n": res["overlap_n"],
            "target_coverage": int(t_mask.sum()),
            "query_coverage": query_cov,
        })

    df = pd.DataFrame(rows)
    # Sort: non-NaN similarities descending, then NaN rows
    df = df.sort_values("similarity", ascending=False, na_position="last")
    return df.head(top_k).reset_index(drop=True)
