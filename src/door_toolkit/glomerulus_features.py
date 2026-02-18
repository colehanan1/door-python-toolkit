"""
Glomerulus Feature Construction
================================

Map DoOR receptor-level response vectors to glomerulus-level feature vectors
using the authoritative receptor-to-glomerulus mapping.

Functions:
    load_receptor_to_glomerulus_mapping: Load mapping CSV into a lookup dict.
    odor_to_receptor_vector: Encode an odor to a receptor vector (thin wrapper).
    receptor_vector_to_glomerulus_vector: Aggregate receptor responses per glomerulus.
    build_design_matrix: Build (n_odors, n_glomeruli) feature matrix with filtering.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

from door_toolkit.encoder import DoOREncoder

logger = logging.getLogger(__name__)


def load_receptor_to_glomerulus_mapping(
    mapping_csv_path: str,
    *,
    adult_only: bool = True,
    drop_ambiguous: bool = False,
) -> Tuple[Dict[str, str], dict]:
    """Load receptor-to-glomerulus mapping from the authoritative CSV.

    Returns DoOR receptor names as keys (not FlyWire glomerulus codes).

    Args:
        mapping_csv_path: Path to door_to_flywire_mapping.csv.
        adult_only: If True, exclude larval-only receptors.
        drop_ambiguous: If True, exclude receptors flagged as ambiguous.

    Returns:
        mapping: Dict mapping DoOR receptor name → DoOR receptor name (identity).
                 (Keys are the DoOR names that will be used in outputs.)
        meta: Dict with mapping statistics.
    """
    df = pd.read_csv(mapping_csv_path)

    # Filter out rows with no glomerulus assignment
    df = df[df["flywire_glomerulus"].notna() & (df["flywire_glomerulus"] != "")]

    if adult_only:
        # Keep rows where life_stage is not exclusively Larval
        if "life_stage" in df.columns:
            df = df[df["life_stage"] != "Larval"]
        if "is_larval" in df.columns:
            df = df[df["is_larval"] != "Yes"]

    n_before_ambig = len(df)
    if drop_ambiguous and "is_ambiguous" in df.columns:
        df = df[df["is_ambiguous"] != "Yes"]

    # Build receptor → receptor mapping (keep DoOR names as identifiers)
    # Each DoOR receptor is its own "glomerulus" identifier in the output
    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        receptor = row["door_name"]
        # Use receptor name itself as the identifier (no FlyWire conversion)
        mapping[receptor] = receptor

    all_receptors = sorted(set(mapping.keys()))

    meta = {
        "n_receptors_mapped": len(mapping),
        "n_receptors_total": len(all_receptors),
        "n_rows_dropped_ambiguous": n_before_ambig - len(df) if drop_ambiguous else 0,
        "adult_only": adult_only,
        "drop_ambiguous": drop_ambiguous,
        "all_receptors": all_receptors,
        "note": "Using DoOR receptor names (not FlyWire glomerulus codes) for output",
    }

    logger.info(
        "[load_mapping] %d receptors mapped (adult_only=%s, drop_ambiguous=%s)",
        len(mapping),
        adult_only,
        drop_ambiguous,
    )
    return mapping, meta


def odor_to_receptor_vector(
    odor: str,
    encoder: DoOREncoder,
) -> Tuple[np.ndarray, List[str]]:
    """Encode an odor name to a receptor response vector.

    Thin wrapper around DoOREncoder.encode() that always returns numpy.

    Args:
        odor: Odorant name (case-insensitive).
        encoder: Initialized DoOREncoder instance.

    Returns:
        vector: 1-D numpy array of receptor responses (n_channels,).
        receptor_names: List of receptor identifiers matching vector indices.
    """
    raw = encoder.encode(odor, fill_missing=0.0)
    # Ensure numpy regardless of torch availability
    vec = np.asarray(raw, dtype=np.float32)
    return vec, list(encoder.receptor_names)


def receptor_vector_to_glomerulus_vector(
    receptor_vector: np.ndarray,
    receptor_names: List[str],
    mapping: Dict[str, str],
    *,
    agg: Literal["max", "mean", "sum"] = "max",
    glomerulus_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str], dict]:
    """Filter receptor responses to keep only mapped DoOR receptors.

    Each mapped DoOR receptor becomes a feature in the output (no aggregation).

    Args:
        receptor_vector: 1-D array of receptor responses.
        receptor_names: Receptor identifiers (same order as vector).
        mapping: Dict from receptor name → receptor name (identity mapping).
        agg: Aggregation function (unused, kept for API compatibility).
        glomerulus_order: Optional fixed ordering of features. If None,
            sorted alphabetically from the mapping.

    Returns:
        receptor_subset: 1-D array of mapped receptor responses only.
        mapped_receptor_names: Ordered list of DoOR receptor names (not aggregated).
        meta: Coverage and mapping statistics.
    """
    # Build receptor → index mapping for the input vector
    receptor_to_idx = {name: i for i, name in enumerate(receptor_names)}
    mapped_receptors = []
    unmapped_receptors = []

    # Identify which receptors are in the mapping
    for receptor in receptor_names:
        if receptor in mapping:
            mapped_receptors.append(receptor)
        else:
            unmapped_receptors.append(receptor)

    # Determine receptor ordering
    if glomerulus_order is not None:
        sorted_receptors = glomerulus_order
    else:
        sorted_receptors = sorted(mapped_receptors)

    # Extract responses for mapped receptors only (no aggregation)
    receptor_subset = np.zeros(len(sorted_receptors), dtype=np.float32)
    for j, receptor in enumerate(sorted_receptors):
        if receptor in receptor_to_idx:
            idx = receptor_to_idx[receptor]
            receptor_subset[j] = float(receptor_vector[idx])

    meta = {
        "n_receptors_mapped": len(mapped_receptors),
        "n_receptors_unmapped": len(unmapped_receptors),
        "n_receptors_output": len(sorted_receptors),
        "unmapped_receptors": unmapped_receptors,
        "aggregation": "none (DoOR receptor names preserved)",
        "mapping_coverage_frac": len(mapped_receptors) / max(len(receptor_names), 1),
    }

    return receptor_subset, sorted_receptors, meta


def build_design_matrix(
    odors: List[str],
    encoder: DoOREncoder,
    mapping: Dict[str, List[str]],
    *,
    feature_set: Literal["all", "union", "intersection"] = "union",
    activation_threshold: float = 0.05,
    agg: Literal["max", "mean", "sum"] = "max",
) -> Tuple[np.ndarray, List[str], dict]:
    """Build a (n_odors, n_receptors) design matrix using DoOR receptor names.

    Args:
        odors: List of odorant names.
        encoder: Initialized DoOREncoder.
        mapping: Receptor-to-glomerulus mapping dict (DoOR receptors only).
        feature_set: Which receptors to include:
            - "all": every receptor in the mapping.
            - "union": receptors active for at least one odor.
            - "intersection": receptors active for all odors.
        activation_threshold: Response above this marks a receptor as active.
        agg: Aggregation (ignored, kept for API compatibility).

    Returns:
        X: Design matrix of shape (n_odors, n_selected_receptors).
        selected_receptor_names: Ordered DoOR receptor names for columns.
        meta: Build metadata including per-odor active counts.
    """
    # Collect all possible receptors from the mapping (DoOR names only)
    all_receptors = sorted(set(mapping.keys()))

    # Compute receptor vectors for each odor (filtered to mapped receptors)
    vectors = []
    per_odor_meta = []
    for odor in odors:
        rvec, rnames = odor_to_receptor_vector(odor, encoder)
        # Filter to mapped receptors using identity mapping
        subset_vec, subset_names, meta_subset = receptor_vector_to_glomerulus_vector(
            rvec, rnames, mapping, agg=agg, glomerulus_order=all_receptors
        )
        vectors.append(subset_vec)
        per_odor_meta.append(meta_subset)

    full_matrix = np.stack(vectors)  # (n_odors, n_all_receptors)

    # Compute active masks (receptors with response > threshold)
    active_masks = np.abs(full_matrix) > activation_threshold  # (n_odors, n_all_receptors)
    per_odor_active_counts = {
        odors[i]: int(active_masks[i].sum()) for i in range(len(odors))
    }

    # Select columns based on feature_set
    if feature_set == "all":
        selected_mask = np.ones(len(all_receptors), dtype=bool)
    elif feature_set == "union":
        selected_mask = active_masks.any(axis=0)
    elif feature_set == "intersection":
        selected_mask = active_masks.all(axis=0)
    else:
        raise ValueError(f"feature_set must be all/union/intersection, got '{feature_set}'")

    selected_indices = np.where(selected_mask)[0]
    selected_receptor_names = [all_receptors[i] for i in selected_indices]
    X = full_matrix[:, selected_indices]

    meta = {
        "feature_set": feature_set,
        "activation_threshold": activation_threshold,
        "n_all_receptors": len(all_receptors),
        "n_selected_receptors": len(selected_receptor_names),
        "per_odor_active_counts": per_odor_active_counts,
        "per_odor_mapping_meta": per_odor_meta,
        "odors": odors,
        "note": "Using DoOR receptor names as features (not FlyWire glomerulus codes)",
    }

    logger.info(
        "[build_design_matrix] %d odors × %d receptors (feature_set=%s, threshold=%.3f)",
        len(odors),
        len(selected_receptor_names),
        feature_set,
        activation_threshold,
    )
    return X, selected_receptor_names, meta
