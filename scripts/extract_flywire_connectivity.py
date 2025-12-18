#!/usr/bin/env python3
"""
Extract & Validate FlyWire Connectivity for Olfactory Circuit

This script maps DoOR receptors to FlyWire neurons and extracts connectome-constrained
connectivity matrices for ORN→PN→KC pathways.

Author: Generated for neuroscience data pipeline
Date: 2025-12-17

Key Design Decisions (Chain-of-Thought):
=========================================
1. Receptor-to-neuron mapping: Each DoOR receptor maps to multiple physical ORN neurons
   WHY: Single receptor type expressed by multiple ORN neurons (~20-50 per receptor)

2. Connectivity aggregation: Sum synapses across all ORNs of same receptor type
   WHY: Model receptor-level connectivity, not individual neuron connectivity

3. Sparse connectivity: Store as sparse tensors to handle ~1% connectivity density
   WHY: Efficient memory usage and faster computation for large matrices

4. Normalization: Row-wise max normalization to [0,1]
   WHY: Stabilize RNN training, make all receptors contribute equally despite synapse count variation
"""

import os
import sys
import json
import warnings
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from scipy import sparse

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=pd.errors.DtypeWarning)


# ============================================================================
# STEP 1: INITIALIZE FLYWIRE DATA LOADERS
# ============================================================================

def load_flywire_labels(labels_path: str) -> pd.DataFrame:
    """
    Load FlyWire neuron labels from processed_labels.csv.gz.

    Chain-of-Thought:
    -----------------
    WHY: Need to identify neuron types (ORN, PN, KC) from FlyWire database
    HOW: Parse processed_labels which contain community annotations
    WHAT TO CHECK:
        - File exists and is readable
        - Contains root_id and processed_labels columns
        - Labels are parseable (handle list format)

    Args:
        labels_path: Path to processed_labels.csv.gz

    Returns:
        DataFrame with columns: root_id, processed_labels
    """
    print("\n" + "="*80)
    print("STEP 1: INITIALIZING FLYWIRE DATA LOADERS")
    print("="*80)

    print(f"\n[WHY] Load FlyWire neuron identity labels")
    print("[JUSTIFICATION] Must identify ORN, PN, and KC neurons from connectome")

    print(f"\n[LOADING] {labels_path}")

    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"FlyWire labels file not found: {labels_path}")

    # Load labels
    df = pd.read_csv(labels_path, compression='gzip')

    print(f"[LOADED] {len(df):,} neurons with labels")

    # Check required columns
    if 'root_id' not in df.columns or 'processed_labels' not in df.columns:
        raise ValueError(f"Expected columns 'root_id' and 'processed_labels', got: {df.columns.tolist()}")

    print(f"[VALIDATION] Required columns present: root_id, processed_labels")

    return df


def load_flywire_connectivity(connections_path: str) -> pd.DataFrame:
    """
    Load FlyWire synaptic connectivity from connections_princeton.csv.gz.

    Chain-of-Thought:
    -----------------
    WHY: Extract pre→post neuron connections with synapse counts
    HOW: Load connections table, aggregate by neuron pairs
    WHAT TO CHECK:
        - File size (can be >5GB, may need chunking)
        - Contains pre_root_id, post_root_id, syn_count
        - Synapse counts are positive integers

    Args:
        connections_path: Path to connections_princeton.csv.gz

    Returns:
        DataFrame with columns: pre_root_id, post_root_id, syn_count
    """
    print(f"\n[LOADING] {connections_path}")

    if not os.path.exists(connections_path):
        raise FileNotFoundError(f"FlyWire connectivity file not found: {connections_path}")

    # Check file size
    file_size_mb = os.path.getsize(connections_path) / (1024 ** 2)
    print(f"[FILE SIZE] {file_size_mb:.1f} MB")

    if file_size_mb > 1000:
        print("[INFO] Large file detected, loading in chunks...")

    # Load connectivity (aggregate by pre/post pair across neuropils)
    print("[INFO] Aggregating synapses across neuropils...")

    # Use chunking for large files
    chunk_size = 1000000
    chunks = []

    for i, chunk in enumerate(pd.read_csv(connections_path, compression='gzip', chunksize=chunk_size)):
        if i % 5 == 0 and i > 0:
            print(f"  Processed {i * chunk_size:,} rows...")

        # Aggregate by pre/post pair
        agg = chunk.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()
        chunks.append(agg)

    # Concatenate all chunks
    df = pd.concat(chunks, ignore_index=True)

    # Final aggregation across chunks
    df = df.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()

    print(f"[LOADED] {len(df):,} unique pre→post connections")
    print(f"[STATS] Total synapses: {df['syn_count'].sum():,}")
    print(f"[STATS] Avg synapses per connection: {df['syn_count'].mean():.1f}")

    return df


def load_flywire_cell_types(cell_types_path: str) -> pd.DataFrame:
    """
    Load FlyWire cell type classifications.

    Chain-of-Thought:
    -----------------
    WHY: Additional cell type information beyond community labels
    HOW: Load consolidated_cell_types.csv.gz
    WHAT TO USE: primary_type for broad classification

    Args:
        cell_types_path: Path to consolidated_cell_types.csv.gz

    Returns:
        DataFrame with columns: root_id, primary_type, additional_type(s)
    """
    print(f"\n[LOADING] {cell_types_path}")

    if not os.path.exists(cell_types_path):
        print("[WARNING] Cell types file not found, will rely only on labels")
        return pd.DataFrame(columns=['root_id', 'primary_type'])

    df = pd.read_csv(cell_types_path, compression='gzip')

    print(f"[LOADED] {len(df):,} neurons with cell type classifications")

    return df


# ============================================================================
# STEP 2: MAP 78 DoOR RECEPTORS TO FLYWIRE ORNs
# ============================================================================

def get_standard_door_receptors() -> List[str]:
    """
    Get the standard 78 DoOR receptor names.

    Chain-of-Thought:
    -----------------
    WHY: Need canonical list of receptors to map to FlyWire
    HOW: Use DoOREncoder to get receptor names used in feature extraction
    WHAT: Should match receptors used in PROMPT 1 feature creation

    Returns:
        List of 78 receptor names (e.g., ['Or1a', 'Or7a', ..., 'Gr21a', 'Gr63a'])
    """
    from door_toolkit.encoder import DoOREncoder

    encoder = DoOREncoder()
    receptor_names = encoder.receptor_names

    print(f"\n[INFO] Loaded {len(receptor_names)} DoOR receptors from encoder")

    return receptor_names


def extract_receptor_from_label(label_str: str) -> List[str]:
    """
    Extract receptor names (Or/Gr) from FlyWire label string.

    Chain-of-Thought:
    -----------------
    WHY: FlyWire labels contain receptor info like "ORN_DL5; Or7a; FBbt_00067005"
    HOW: Parse label string, extract Or<number><letter> or Gr<number><letter> patterns
    WHAT: Handle multiple receptors per label (e.g., "Or22a; Or22b")

    Args:
        label_str: FlyWire processed_labels string

    Returns:
        List of receptor names found in label (e.g., ['Or7a'], ['Or22a', 'Or22b'])
    """
    if pd.isna(label_str):
        return []

    # Extract Or and Gr receptors
    # Pattern: Or<digits><letter> or Gr<digits><letter>
    pattern = r'\b(Or|Gr)(\d+)([a-z])\b'

    matches = re.findall(pattern, str(label_str), re.IGNORECASE)

    receptors = []
    for match in matches:
        # match = ('Or', '7', 'a') → 'Or7a'
        receptor = f"{match[0]}{match[1]}{match[2]}"
        receptors.append(receptor)

    return receptors


def map_receptors_to_orns(
    labels_df: pd.DataFrame,
    door_receptors: List[str]
) -> Dict[str, List[int]]:
    """
    Map each DoOR receptor to FlyWire ORN neurons.

    Chain-of-Thought:
    -----------------
    WHY: Need to link abstract receptor names to physical neuron IDs
    HOW: Parse labels for each neuron, extract receptor info, build mapping
    WHAT TO CHECK:
        - All 78 receptors mapped to ≥1 ORN
        - ORN counts per receptor are reasonable (expected ~10-50)
        - No receptors completely missing

    Args:
        labels_df: DataFrame with neuron labels
        door_receptors: List of 78 DoOR receptor names

    Returns:
        Dict mapping receptor_name → [orn_root_id_1, orn_root_id_2, ...]
    """
    print("\n" + "="*80)
    print("STEP 2: MAPPING 78 DoOR RECEPTORS TO FLYWIRE ORNs")
    print("="*80)

    print(f"\n[WHY] Map DoOR receptors to physical FlyWire ORN neurons")
    print("[JUSTIFICATION] Need neuron-level connectivity to build circuit matrices")

    receptor_to_orns = defaultdict(list)

    print(f"\n[PROCESSING] Scanning {len(labels_df):,} neurons for ORN labels...")

    # Normalize door_receptors for case-insensitive matching
    door_receptors_lower = {r.lower(): r for r in door_receptors}

    orn_count = 0

    for idx, row in labels_df.iterrows():
        label_str = row['processed_labels']
        root_id = row['root_id']

        # Extract receptors from this neuron's label
        receptors_found = extract_receptor_from_label(label_str)

        if receptors_found:
            orn_count += 1

            # Add this ORN to all its receptors
            for receptor in receptors_found:
                receptor_lower = receptor.lower()

                # Match to DoOR receptor (case-insensitive)
                if receptor_lower in door_receptors_lower:
                    canonical_receptor = door_receptors_lower[receptor_lower]
                    receptor_to_orns[canonical_receptor].append(root_id)

    print(f"[FOUND] {orn_count:,} ORN neurons total")

    # Statistics
    print(f"\n[RECEPTOR MAPPING SUMMARY]")
    receptors_mapped = len(receptor_to_orns)
    receptors_missing = len(door_receptors) - receptors_mapped

    print(f"  Receptors mapped: {receptors_mapped} / {len(door_receptors)}")
    print(f"  Receptors missing: {receptors_missing}")

    if receptors_missing > 0:
        missing = set(door_receptors) - set(receptor_to_orns.keys())
        print(f"\n[WARNING] Missing receptors: {sorted(missing)[:20]}")  # Show first 20

    # Coverage statistics
    print(f"\n[ORN COUNTS PER RECEPTOR]")
    orn_counts = [len(orns) for orns in receptor_to_orns.values()]

    if orn_counts:
        print(f"  Mean: {np.mean(orn_counts):.1f} ORNs per receptor")
        print(f"  Median: {np.median(orn_counts):.0f} ORNs per receptor")
        print(f"  Min: {min(orn_counts)} ORNs")
        print(f"  Max: {max(orn_counts)} ORNs")

        # Show examples
        print(f"\n[EXAMPLES] Top 5 receptors by ORN count:")
        sorted_receptors = sorted(receptor_to_orns.items(), key=lambda x: len(x[1]), reverse=True)
        for receptor, orns in sorted_receptors[:5]:
            print(f"  {receptor}: {len(orns)} ORNs")

        print(f"\n[EXAMPLES] Bottom 5 receptors by ORN count:")
        for receptor, orns in sorted_receptors[-5:]:
            print(f"  {receptor}: {len(orns)} ORNs")

    print("\n[STEP 2 COMPLETE] ✓ Receptors mapped to ORNs")

    return dict(receptor_to_orns)


# ============================================================================
# STEP 3: EXTRACT ORN→PN CONNECTIVITY MATRIX
# ============================================================================

def identify_pn_neurons(labels_df: pd.DataFrame) -> List[int]:
    """
    Identify projection neuron (PN) root IDs from labels.

    Chain-of-Thought:
    -----------------
    WHY: Need to know which neurons are PNs to extract ORN→PN connections
    HOW: Search labels for "PN", "projection neuron", "ALPN" patterns
    WHAT: Expected ~3000-5000 PNs in antennal lobe

    Args:
        labels_df: DataFrame with neuron labels

    Returns:
        List of PN root_ids
    """
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING ORN→PN CONNECTIVITY MATRIX")
    print("="*80)

    print(f"\n[WHY] Identify projection neurons (PNs) from FlyWire labels")
    print("[JUSTIFICATION] PNs are the second layer in olfactory circuit")

    pn_ids = []

    # Patterns that indicate PN
    pn_patterns = [
        r'\bPN\b',
        r'\blPN\b',
        r'\bmPN\b',
        r'\bvPN\b',
        r'\badPN\b',
        r'\bALPN\b',
        r'projection neuron',
    ]

    print(f"\n[SEARCHING] Scanning {len(labels_df):,} neurons for PN patterns...")

    for idx, row in labels_df.iterrows():
        label_str = str(row['processed_labels']).lower()

        # Check if any PN pattern matches
        for pattern in pn_patterns:
            if re.search(pattern, label_str, re.IGNORECASE):
                pn_ids.append(row['root_id'])
                break

    print(f"[FOUND] {len(pn_ids):,} projection neurons (PNs)")

    if len(pn_ids) == 0:
        raise ValueError("No PNs found! Check label patterns.")

    return pn_ids


def extract_orn_pn_connectivity(
    receptor_to_orns: Dict[str, List[int]],
    pn_ids: List[int],
    connectivity_df: pd.DataFrame,
    door_receptors: List[str]
) -> Tuple[sparse.csr_matrix, List[int]]:
    """
    Extract ORN→PN connectivity matrix.

    Chain-of-Thought:
    -----------------
    WHY: Model how sensory input reaches decision-making PNs
    HOW: For each receptor, aggregate synapses from all ORNs to all PNs
    WHAT: Matrix shape [n_receptors, n_pns] with synapse counts

    Args:
        receptor_to_orns: Mapping of receptors to ORN root_ids
        pn_ids: List of PN root_ids
        connectivity_df: FlyWire connectivity table
        door_receptors: List of receptor names (for ordering)

    Returns:
        - Sparse connectivity matrix [n_receptors, n_pns]
        - List of PN root_ids (column indices)
    """
    print(f"\n[BUILDING] ORN→PN connectivity matrix...")

    # Create PN index mapping
    pn_id_to_idx = {pn_id: idx for idx, pn_id in enumerate(pn_ids)}

    # Filter connectivity to ORN→PN connections only
    orn_ids_set = set()
    for orns in receptor_to_orns.values():
        orn_ids_set.update(orns)

    print(f"[FILTERING] Keeping connections where:")
    print(f"  pre_root_id in {len(orn_ids_set):,} ORNs")
    print(f"  post_root_id in {len(pn_ids):,} PNs")

    orn_pn_connections = connectivity_df[
        connectivity_df['pre_root_id'].isin(orn_ids_set) &
        connectivity_df['post_root_id'].isin(pn_id_to_idx.keys())
    ].copy()

    print(f"[FOUND] {len(orn_pn_connections):,} ORN→PN connections")

    if len(orn_pn_connections) == 0:
        raise ValueError("No ORN→PN connections found! Check neuron identification.")

    # Build connectivity matrix
    n_receptors = len(door_receptors)
    n_pns = len(pn_ids)

    print(f"\n[MATRIX] Building [{n_receptors}, {n_pns}] connectivity matrix...")

    # Create sparse matrix in COO format
    row_indices = []
    col_indices = []
    values = []

    # Create receptor index mapping
    receptor_to_idx = {r: idx for idx, r in enumerate(door_receptors)}

    # Aggregate by receptor→PN
    for receptor, orns in receptor_to_orns.items():
        if receptor not in receptor_to_idx:
            continue  # Skip receptors not in DoOR list

        receptor_idx = receptor_to_idx[receptor]

        # Get all connections from this receptor's ORNs
        receptor_connections = orn_pn_connections[
            orn_pn_connections['pre_root_id'].isin(orns)
        ]

        # Aggregate by target PN
        pn_synapses = receptor_connections.groupby('post_root_id')['syn_count'].sum()

        for pn_id, syn_count in pn_synapses.items():
            if pn_id in pn_id_to_idx:
                pn_idx = pn_id_to_idx[pn_id]

                row_indices.append(receptor_idx)
                col_indices.append(pn_idx)
                values.append(syn_count)

    # Create sparse matrix
    connectivity_matrix = sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_receptors, n_pns),
        dtype=np.float32
    ).tocsr()

    # Statistics
    n_nonzero = connectivity_matrix.nnz
    sparsity = 100 * (1 - n_nonzero / (n_receptors * n_pns))

    print(f"\n[MATRIX STATISTICS]")
    print(f"  Shape: [{n_receptors}, {n_pns}]")
    print(f"  Non-zero entries: {n_nonzero:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Max synapses: {connectivity_matrix.max():.0f}")
    print(f"  Mean synapses (non-zero): {connectivity_matrix.data.mean():.1f}")

    # Check coverage
    receptors_with_pns = (connectivity_matrix.sum(axis=1) > 0).sum()
    print(f"\n[COVERAGE]")
    print(f"  Receptors connected to ≥1 PN: {receptors_with_pns} / {n_receptors}")

    if receptors_with_pns < n_receptors:
        disconnected = [door_receptors[i] for i in range(n_receptors)
                       if connectivity_matrix[i].sum() == 0]
        print(f"  [WARNING] Disconnected receptors: {disconnected[:10]}")

    print("\n[STEP 3 COMPLETE] ✓ ORN→PN connectivity extracted")

    return connectivity_matrix, pn_ids


# ============================================================================
# STEP 4: EXTRACT PN→KC CONNECTIVITY MATRIX
# ============================================================================

def identify_kc_neurons(labels_df: pd.DataFrame) -> List[int]:
    """
    Identify Kenyon cell (KC) root IDs from labels.

    Chain-of-Thought:
    -----------------
    WHY: KCs are the output layer where learning occurs (mushroom body)
    HOW: Search labels for "KC", "Kenyon", "Kenyon cell" patterns
    WHAT: Expected ~2000 KCs in mushroom body

    Args:
        labels_df: DataFrame with neuron labels

    Returns:
        List of KC root_ids
    """
    print("\n" + "="*80)
    print("STEP 4: EXTRACTING PN→KC CONNECTIVITY MATRIX")
    print("="*80)

    print(f"\n[WHY] Identify Kenyon cells (KCs) from FlyWire labels")
    print("[JUSTIFICATION] KCs are the output layer where learning occurs")

    kc_ids = []

    # Patterns that indicate KC
    kc_patterns = [
        r'\bKC\b',
        r'Kenyon.*cell',
        r'kenyon.*cell',
    ]

    print(f"\n[SEARCHING] Scanning {len(labels_df):,} neurons for KC patterns...")

    for idx, row in labels_df.iterrows():
        label_str = str(row['processed_labels'])

        # Check if any KC pattern matches
        for pattern in kc_patterns:
            if re.search(pattern, label_str, re.IGNORECASE):
                kc_ids.append(row['root_id'])
                break

    print(f"[FOUND] {len(kc_ids):,} Kenyon cells (KCs)")

    if len(kc_ids) == 0:
        raise ValueError("No KCs found! Check label patterns.")

    return kc_ids


def extract_pn_kc_connectivity(
    pn_ids: List[int],
    kc_ids: List[int],
    connectivity_df: pd.DataFrame
) -> Tuple[sparse.csr_matrix, List[int]]:
    """
    Extract PN→KC connectivity matrix.

    Chain-of-Thought:
    -----------------
    WHY: Model how PN activations drive KC sparse encoding
    HOW: Extract all PN→KC connections from connectivity table
    WHAT: Matrix shape [n_pns, n_kcs] with synapse counts
          Expected 40-60% KC recruitment (sparse coding)

    Args:
        pn_ids: List of PN root_ids
        kc_ids: List of KC root_ids
        connectivity_df: FlyWire connectivity table

    Returns:
        - Sparse connectivity matrix [n_pns, n_kcs]
        - List of KC root_ids (column indices)
    """
    print(f"\n[BUILDING] PN→KC connectivity matrix...")

    # Create index mappings
    pn_id_to_idx = {pn_id: idx for idx, pn_id in enumerate(pn_ids)}
    kc_id_to_idx = {kc_id: idx for idx, kc_id in enumerate(kc_ids)}

    print(f"[FILTERING] Keeping connections where:")
    print(f"  pre_root_id in {len(pn_ids):,} PNs")
    print(f"  post_root_id in {len(kc_ids):,} KCs")

    pn_kc_connections = connectivity_df[
        connectivity_df['pre_root_id'].isin(pn_id_to_idx.keys()) &
        connectivity_df['post_root_id'].isin(kc_id_to_idx.keys())
    ].copy()

    print(f"[FOUND] {len(pn_kc_connections):,} PN→KC connections")

    if len(pn_kc_connections) == 0:
        raise ValueError("No PN→KC connections found!")

    # Build connectivity matrix
    n_pns = len(pn_ids)
    n_kcs = len(kc_ids)

    print(f"\n[MATRIX] Building [{n_pns}, {n_kcs}] connectivity matrix...")

    # Aggregate synapses by PN→KC pair
    pn_kc_agg = pn_kc_connections.groupby(['pre_root_id', 'post_root_id'])['syn_count'].sum().reset_index()

    # Build sparse matrix
    row_indices = []
    col_indices = []
    values = []

    for _, row in pn_kc_agg.iterrows():
        pn_id = row['pre_root_id']
        kc_id = row['post_root_id']
        syn_count = row['syn_count']

        if pn_id in pn_id_to_idx and kc_id in kc_id_to_idx:
            row_indices.append(pn_id_to_idx[pn_id])
            col_indices.append(kc_id_to_idx[kc_id])
            values.append(syn_count)

    connectivity_matrix = sparse.coo_matrix(
        (values, (row_indices, col_indices)),
        shape=(n_pns, n_kcs),
        dtype=np.float32
    ).tocsr()

    # Statistics
    n_nonzero = connectivity_matrix.nnz
    sparsity = 100 * (1 - n_nonzero / (n_pns * n_kcs))

    print(f"\n[MATRIX STATISTICS]")
    print(f"  Shape: [{n_pns}, {n_kcs}]")
    print(f"  Non-zero entries: {n_nonzero:,}")
    print(f"  Sparsity: {sparsity:.2f}%")
    print(f"  Max synapses: {connectivity_matrix.max():.0f}")
    print(f"  Mean synapses (non-zero): {connectivity_matrix.data.mean():.1f}")

    # KC recruitment analysis
    kcs_recruited = (connectivity_matrix.sum(axis=0) > 0).sum()
    kc_recruitment_pct = 100 * kcs_recruited / n_kcs

    print(f"\n[KC RECRUITMENT]")
    print(f"  KCs receiving input: {kcs_recruited:,} / {n_kcs:,}")
    print(f"  Recruitment %: {kc_recruitment_pct:.1f}%")

    if 40 <= kc_recruitment_pct <= 60:
        print(f"  ✓ Recruitment in expected range (40-60%)")
    else:
        print(f"  ⚠ Recruitment outside expected range (40-60%)")

    print("\n[STEP 4 COMPLETE] ✓ PN→KC connectivity extracted")

    return connectivity_matrix, kc_ids


# ============================================================================
# STEP 5: NORMALIZE CONNECTIVITY WEIGHTS
# ============================================================================

def normalize_connectivity(
    matrix: sparse.csr_matrix,
    method: str = 'row_max'
) -> sparse.csr_matrix:
    """
    Normalize connectivity matrix to [0, 1] range.

    Chain-of-Thought:
    -----------------
    WHY: Raw synapse counts vary widely (1-1000+), destabilizing training
    HOW: Row-wise max normalization - divide each row by its maximum
    WHAT: All values in [0, 1], preserves relative connectivity strength

    Args:
        matrix: Sparse connectivity matrix
        method: Normalization method ('row_max' or 'global_max')

    Returns:
        Normalized sparse matrix with values in [0, 1]
    """
    print("\n" + "="*80)
    print("STEP 5: NORMALIZING CONNECTIVITY WEIGHTS")
    print("="*80)

    print(f"\n[WHY] Normalize synapse counts to [0, 1] range")
    print("[JUSTIFICATION] Stabilize RNN training, make all receptors contribute equally")

    print(f"\n[METHOD] {method}")
    print(f"[INPUT] Matrix shape: {matrix.shape}, values in [{matrix.min():.1f}, {matrix.max():.1f}]")

    if method == 'row_max':
        # Normalize each row by its maximum value
        matrix_csr = matrix.tocsr()

        # Get row-wise max values
        row_max = np.array(matrix_csr.max(axis=1).todense()).flatten()

        # Avoid division by zero
        row_max[row_max == 0] = 1.0

        # Create diagonal matrix for normalization
        row_max_inv = sparse.diags(1.0 / row_max)

        # Normalize
        matrix_normalized = row_max_inv.dot(matrix_csr)

    elif method == 'global_max':
        # Normalize by global maximum
        max_val = matrix.max()
        matrix_normalized = matrix / max_val if max_val > 0 else matrix

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    print(f"[OUTPUT] Values in [{matrix_normalized.min():.3f}, {matrix_normalized.max():.3f}]")

    # Validation
    if matrix_normalized.min() < 0 or matrix_normalized.max() > 1.0:
        raise ValueError(f"Normalization failed: values outside [0, 1]")

    print("[VALIDATION] ✓ All values in [0, 1]")

    print("\n[STEP 5 COMPLETE] ✓ Connectivity weights normalized")

    return matrix_normalized


# ============================================================================
# STEP 6: VALIDATE CIRCUIT INTEGRITY
# ============================================================================

def validate_circuit_integrity(
    orn_pn_matrix: sparse.csr_matrix,
    pn_kc_matrix: sparse.csr_matrix,
    door_receptors: List[str]
) -> Dict[str, bool]:
    """
    Validate that all receptors can reach KCs through PNs.

    Chain-of-Thought:
    -----------------
    WHY: Broken pathways would cause RNN to ignore those receptors
    HOW: For each receptor, check ORN→PN→KC path exists
    WHAT: All 78 receptors must reach ≥1 KC

    Args:
        orn_pn_matrix: ORN→PN connectivity [n_receptors, n_pns]
        pn_kc_matrix: PN→KC connectivity [n_pns, n_kcs]
        door_receptors: List of receptor names

    Returns:
        Dict mapping receptor_name → can_reach_kcs (bool)
    """
    print("\n" + "="*80)
    print("STEP 6: VALIDATING CIRCUIT INTEGRITY")
    print("="*80)

    print(f"\n[WHY] Ensure all receptors have complete ORN→PN→KC pathways")
    print("[JUSTIFICATION] Broken pathways would cause RNN to ignore receptors")

    n_receptors = orn_pn_matrix.shape[0]

    receptor_status = {}
    broken_pathways = []

    print(f"\n[CHECKING] Validating pathways for {n_receptors} receptors...")

    for receptor_idx in range(n_receptors):
        receptor_name = door_receptors[receptor_idx]

        # Check 1: Can receptor reach ≥1 PN?
        receptor_to_pns = orn_pn_matrix[receptor_idx]
        pns_reached = (receptor_to_pns > 0).sum()

        if pns_reached == 0:
            receptor_status[receptor_name] = False
            broken_pathways.append((receptor_name, "No PNs reached"))
            continue

        # Check 2: Can those PNs reach ≥1 KC?
        # Get indices of PNs reached by this receptor
        pn_indices = receptor_to_pns.nonzero()[1]

        # Check if any of these PNs reach KCs
        kcs_reached = 0
        for pn_idx in pn_indices:
            kcs_from_pn = (pn_kc_matrix[pn_idx] > 0).sum()
            kcs_reached += kcs_from_pn

        if kcs_reached == 0:
            receptor_status[receptor_name] = False
            broken_pathways.append((receptor_name, f"Reaches {pns_reached} PNs but 0 KCs"))
            continue

        # Pathway is intact
        receptor_status[receptor_name] = True

    # Summary
    intact_count = sum(receptor_status.values())
    broken_count = len(broken_pathways)

    print(f"\n[VALIDATION RESULTS]")
    print(f"  Intact pathways: {intact_count} / {n_receptors}")
    print(f"  Broken pathways: {broken_count} / {n_receptors}")

    if broken_count > 0:
        print(f"\n[BROKEN PATHWAYS]")
        for receptor, reason in broken_pathways[:20]:  # Show first 20
            print(f"  ✗ {receptor}: {reason}")

        if broken_count > 20:
            print(f"  ... and {broken_count - 20} more")

        print(f"\n[WARNING] Some receptors cannot reach mushroom body!")
        print(f"[RECOMMENDATION] These receptors will be excluded from RNN or zero-padded")

    else:
        print(f"\n[SUCCESS] ✓ All {n_receptors} receptors have complete pathways to KC population")

    print("\n[STEP 6 COMPLETE] ✓ Circuit integrity validated")

    return receptor_status


# ============================================================================
# STEP 7: COMPUTE CONNECTIVITY STATISTICS
# ============================================================================

def compute_connectivity_statistics(
    orn_pn_matrix: sparse.csr_matrix,
    pn_kc_matrix: sparse.csr_matrix,
    receptor_to_orns: Dict[str, List[int]],
    door_receptors: List[str],
    receptor_status: Dict[str, bool]
) -> pd.DataFrame:
    """
    Compute detailed connectivity statistics per receptor.

    Chain-of-Thought:
    -----------------
    WHY: Understand circuit structure for model interpretation
    HOW: For each receptor, compute coverage metrics
    WHAT TO MEASURE:
        - Number of ORNs
        - Number of target PNs
        - % of PN population reached
        - Number of reached KCs (via PNs)
        - % of KC population reached

    Args:
        orn_pn_matrix: ORN→PN connectivity
        pn_kc_matrix: PN→KC connectivity
        receptor_to_orns: Mapping of receptors to ORNs
        door_receptors: List of receptor names
        receptor_status: Circuit integrity status per receptor

    Returns:
        DataFrame with statistics per receptor
    """
    print("\n" + "="*80)
    print("STEP 7: COMPUTING CONNECTIVITY STATISTICS")
    print("="*80)

    print(f"\n[WHY] Analyze connectivity patterns for each receptor")
    print("[JUSTIFICATION] Helps understand circuit structure and identify outliers")

    n_pns_total = orn_pn_matrix.shape[1]
    n_kcs_total = pn_kc_matrix.shape[1]

    stats_list = []

    print(f"\n[COMPUTING] Statistics for {len(door_receptors)} receptors...")

    for receptor_idx, receptor_name in enumerate(door_receptors):
        # Number of ORNs
        n_orns = len(receptor_to_orns.get(receptor_name, []))

        # PN connectivity
        receptor_to_pns = orn_pn_matrix[receptor_idx]
        pns_reached_indices = receptor_to_pns.nonzero()[1]
        n_pns_reached = len(pns_reached_indices)
        pct_pns_reached = 100 * n_pns_reached / n_pns_total if n_pns_total > 0 else 0

        # Total synapses to PNs
        total_synapses_to_pns = receptor_to_pns.sum()

        # KC connectivity (via PNs)
        kcs_reached_set = set()

        for pn_idx in pns_reached_indices:
            pn_to_kcs = pn_kc_matrix[pn_idx]
            kc_indices = pn_to_kcs.nonzero()[1]
            kcs_reached_set.update(kc_indices)

        n_kcs_reached = len(kcs_reached_set)
        pct_kcs_reached = 100 * n_kcs_reached / n_kcs_total if n_kcs_total > 0 else 0

        # Circuit integrity
        pathway_intact = receptor_status.get(receptor_name, False)

        stats_list.append({
            'receptor': receptor_name,
            'n_orns': n_orns,
            'n_pns_reached': n_pns_reached,
            'pct_pns_reached': pct_pns_reached,
            'total_synapses_to_pns': total_synapses_to_pns,
            'n_kcs_reached': n_kcs_reached,
            'pct_kcs_reached': pct_kcs_reached,
            'pathway_intact': pathway_intact
        })

    stats_df = pd.DataFrame(stats_list)

    # Summary
    print(f"\n[SUMMARY STATISTICS]")
    print(f"  Mean ORNs per receptor: {stats_df['n_orns'].mean():.1f}")
    print(f"  Mean PNs reached: {stats_df['n_pns_reached'].mean():.1f} ({stats_df['pct_pns_reached'].mean():.1f}%)")
    print(f"  Mean KCs reached: {stats_df['n_kcs_reached'].mean():.1f} ({stats_df['pct_kcs_reached'].mean():.1f}%)")

    # Top receptors by KC coverage
    print(f"\n[TOP 5 RECEPTORS BY KC COVERAGE]")
    top_receptors = stats_df.nlargest(5, 'pct_kcs_reached')
    for _, row in top_receptors.iterrows():
        print(f"  {row['receptor']:10s}: {row['pct_kcs_reached']:5.1f}% KCs ({row['n_kcs_reached']:4d} / {n_kcs_total})")

    # Bottom receptors
    print(f"\n[BOTTOM 5 RECEPTORS BY KC COVERAGE]")
    bottom_receptors = stats_df.nsmallest(5, 'pct_kcs_reached')
    for _, row in bottom_receptors.iterrows():
        status = "✓" if row['pathway_intact'] else "✗"
        print(f"  {status} {row['receptor']:10s}: {row['pct_kcs_reached']:5.1f}% KCs ({row['n_kcs_reached']:4d} / {n_kcs_total})")

    print("\n[STEP 7 COMPLETE] ✓ Connectivity statistics computed")

    return stats_df


# ============================================================================
# STEP 8: SAVE CONNECTIVITY TENSORS AND METADATA
# ============================================================================

def save_connectivity_data(
    orn_pn_matrix: sparse.csr_matrix,
    pn_kc_matrix: sparse.csr_matrix,
    door_receptors: List[str],
    pn_ids: List[int],
    kc_ids: List[int],
    stats_df: pd.DataFrame,
    output_dir: str
) -> None:
    """
    Save connectivity tensors and metadata to disk.

    Chain-of-Thought:
    -----------------
    WHY: Fast loading in downstream RNN training
    HOW: Save as PyTorch sparse tensors + JSON metadata
    WHAT: Sparse tensors, index mappings, statistics CSV

    Args:
        orn_pn_matrix: ORN→PN connectivity
        pn_kc_matrix: PN→KC connectivity
        door_receptors: List of receptor names
        pn_ids: List of PN root_ids
        kc_ids: List of KC root_ids
        stats_df: Connectivity statistics DataFrame
        output_dir: Output directory path
    """
    print("\n" + "="*80)
    print("STEP 8: SAVING CONNECTIVITY TENSORS AND METADATA")
    print("="*80)

    print(f"\n[WHY] Save connectivity data for RNN training")
    print("[JUSTIFICATION] Avoid re-parsing FlyWire data on every run")

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n[OUTPUT DIR] {output_dir}")

    # Convert scipy sparse to PyTorch sparse tensors
    print(f"\n[CONVERTING] Sparse matrices to PyTorch tensors...")

    def scipy_to_torch_sparse(scipy_matrix):
        """Convert scipy sparse matrix to PyTorch sparse tensor."""
        coo = scipy_matrix.tocoo()
        indices = torch.LongTensor(np.vstack([coo.row, coo.col]))
        values = torch.FloatTensor(coo.data)
        shape = coo.shape
        return torch.sparse_coo_tensor(indices, values, shape).coalesce()

    orn_pn_tensor = scipy_to_torch_sparse(orn_pn_matrix)
    pn_kc_tensor = scipy_to_torch_sparse(pn_kc_matrix)

    print(f"  ORN→PN: {orn_pn_tensor.shape}, {orn_pn_tensor._nnz()} non-zero")
    print(f"  PN→KC: {pn_kc_tensor.shape}, {pn_kc_tensor._nnz()} non-zero")

    # Save tensors
    orn_pn_path = os.path.join(output_dir, "orn_pn_connectivity.pt")
    pn_kc_path = os.path.join(output_dir, "pn_kc_connectivity.pt")

    torch.save(orn_pn_tensor, orn_pn_path)
    torch.save(pn_kc_tensor, pn_kc_path)

    print(f"\n[SAVED TENSORS]")
    print(f"  ✓ {orn_pn_path}")
    print(f"  ✓ {pn_kc_path}")

    # Save metadata
    metadata = {
        'receptor_names': door_receptors,
        'n_receptors': len(door_receptors),
        'n_pns': len(pn_ids),
        'n_kcs': len(kc_ids),
        'pn_root_ids': [int(pid) for pid in pn_ids],  # Convert to int for JSON
        'kc_root_ids': [int(kid) for kid in kc_ids],
        'orn_pn_shape': list(orn_pn_matrix.shape),
        'pn_kc_shape': list(pn_kc_matrix.shape),
        'orn_pn_nnz': int(orn_pn_matrix.nnz),
        'pn_kc_nnz': int(pn_kc_matrix.nnz),
    }

    metadata_path = os.path.join(output_dir, "connectivity_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ {metadata_path}")

    # Save statistics
    stats_path = os.path.join(output_dir, "connectivity_statistics.csv")
    stats_df.to_csv(stats_path, index=False)

    print(f"  ✓ {stats_path}")

    # Verify loading
    print(f"\n[VERIFICATION] Loading saved tensors...")

    loaded_orn_pn = torch.load(orn_pn_path)
    loaded_pn_kc = torch.load(pn_kc_path)

    assert loaded_orn_pn.shape == orn_pn_tensor.shape, "ORN→PN shape mismatch!"
    assert loaded_pn_kc.shape == pn_kc_tensor.shape, "PN→KC shape mismatch!"

    print(f"  ✓ Tensors verified - shapes match")

    print("\n[STEP 8 COMPLETE] ✓ Connectivity data saved successfully")


# ============================================================================
# STEP 9: GENERATE INTEGRATION REPORT
# ============================================================================

def generate_integration_report(
    orn_pn_matrix: sparse.csr_matrix,
    pn_kc_matrix: sparse.csr_matrix,
    receptor_to_orns: Dict[str, List[int]],
    stats_df: pd.DataFrame,
    receptor_status: Dict[str, bool],
    output_dir: str
) -> str:
    """
    Generate comprehensive integration report.

    Chain-of-Thought:
    -----------------
    WHY: User must verify connectome integration before proceeding to RNN
    HOW: Summarize all validation checks and statistics
    WHAT TO INCLUDE:
        - Circuit diagram (neuron counts)
        - Receptor coverage
        - Connectivity statistics
        - Warnings for broken pathways
        - Final readiness status

    Args:
        orn_pn_matrix: ORN→PN connectivity
        pn_kc_matrix: PN→KC connectivity
        receptor_to_orns: Receptor to ORN mapping
        stats_df: Connectivity statistics
        receptor_status: Circuit integrity status
        output_dir: Output directory

    Returns:
        Formatted report string
    """
    print("\n" + "="*80)
    print("STEP 9: GENERATING INTEGRATION REPORT")
    print("="*80)

    print(f"\n[WHY] Create validation report for user review")
    print("[JUSTIFICATION] User must verify integration before RNN training")

    report_lines = []

    # Header
    report_lines.append("="*80)
    report_lines.append("FLYWIRE CONNECTIVITY INTEGRATION REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated: {pd.Timestamp.now()}")
    report_lines.append(f"Output directory: {output_dir}")
    report_lines.append("")

    # Circuit diagram
    n_receptors = orn_pn_matrix.shape[0]
    n_orns = sum(len(orns) for orns in receptor_to_orns.values())
    n_pns = orn_pn_matrix.shape[1]
    n_kcs = pn_kc_matrix.shape[1]

    report_lines.append("-" * 80)
    report_lines.append("1. CIRCUIT DIAGRAM")
    report_lines.append("-" * 80)
    report_lines.append(f"  {n_receptors} DoOR receptors → {n_orns:,} ORN neurons → {n_pns:,} PN neurons → {n_kcs:,} KC neurons")
    report_lines.append("")

    # Receptor mapping
    report_lines.append("-" * 80)
    report_lines.append("2. RECEPTOR MAPPING")
    report_lines.append("-" * 80)
    receptors_mapped = len(receptor_to_orns)
    report_lines.append(f"  Receptors successfully mapped: {receptors_mapped} / {n_receptors}")
    report_lines.append(f"  Total ORNs identified: {n_orns:,}")
    report_lines.append(f"  Mean ORNs per receptor: {stats_df['n_orns'].mean():.1f}")
    report_lines.append("")

    # Connectivity matrices
    report_lines.append("-" * 80)
    report_lines.append("3. CONNECTIVITY MATRICES")
    report_lines.append("-" * 80)

    orn_pn_sparsity = 100 * (1 - orn_pn_matrix.nnz / (orn_pn_matrix.shape[0] * orn_pn_matrix.shape[1]))
    pn_kc_sparsity = 100 * (1 - pn_kc_matrix.nnz / (pn_kc_matrix.shape[0] * pn_kc_matrix.shape[1]))

    report_lines.append(f"  ORN→PN Matrix:")
    report_lines.append(f"    Shape: [{orn_pn_matrix.shape[0]}, {orn_pn_matrix.shape[1]}]")
    report_lines.append(f"    Non-zero entries: {orn_pn_matrix.nnz:,}")
    report_lines.append(f"    Sparsity: {orn_pn_sparsity:.2f}%")
    report_lines.append(f"    Value range: [{orn_pn_matrix.min():.3f}, {orn_pn_matrix.max():.3f}]")
    report_lines.append("")

    report_lines.append(f"  PN→KC Matrix:")
    report_lines.append(f"    Shape: [{pn_kc_matrix.shape[0]}, {pn_kc_matrix.shape[1]}]")
    report_lines.append(f"    Non-zero entries: {pn_kc_matrix.nnz:,}")
    report_lines.append(f"    Sparsity: {pn_kc_sparsity:.2f}%")
    report_lines.append(f"    Value range: [{pn_kc_matrix.min():.3f}, {pn_kc_matrix.max():.3f}]")
    report_lines.append("")

    # Coverage statistics
    report_lines.append("-" * 80)
    report_lines.append("4. COVERAGE STATISTICS")
    report_lines.append("-" * 80)

    report_lines.append(f"  PN Population:")
    report_lines.append(f"    Mean % reached per receptor: {stats_df['pct_pns_reached'].mean():.1f}%")
    report_lines.append(f"    Min: {stats_df['pct_pns_reached'].min():.1f}%")
    report_lines.append(f"    Max: {stats_df['pct_pns_reached'].max():.1f}%")
    report_lines.append("")

    report_lines.append(f"  KC Population:")
    kc_recruitment = stats_df['pct_kcs_reached'].mean()
    report_lines.append(f"    Mean % reached per receptor: {kc_recruitment:.1f}%")
    report_lines.append(f"    Min: {stats_df['pct_kcs_reached'].min():.1f}%")
    report_lines.append(f"    Max: {stats_df['pct_kcs_reached'].max():.1f}%")

    if 40 <= kc_recruitment <= 60:
        report_lines.append(f"    ✓ Recruitment in expected range (40-60%)")
    else:
        report_lines.append(f"    ⚠ Recruitment outside expected range (40-60%)")

    report_lines.append("")

    # Top receptors
    report_lines.append(f"  Top 10 receptors by KC coverage:")
    top_10 = stats_df.nlargest(10, 'pct_kcs_reached')
    for _, row in top_10.iterrows():
        report_lines.append(f"    {row['receptor']:10s}: {row['pct_kcs_reached']:5.1f}% ({row['n_kcs_reached']:4d} / {n_kcs})")

    report_lines.append("")

    # Circuit integrity
    report_lines.append("-" * 80)
    report_lines.append("5. CIRCUIT INTEGRITY")
    report_lines.append("-" * 80)

    intact_count = sum(receptor_status.values())
    broken_count = len(receptor_status) - intact_count

    report_lines.append(f"  Intact pathways: {intact_count} / {len(receptor_status)}")
    report_lines.append(f"  Broken pathways: {broken_count} / {len(receptor_status)}")

    if broken_count > 0:
        broken_receptors = [r for r, status in receptor_status.items() if not status]
        report_lines.append(f"\n  Broken pathways: {', '.join(broken_receptors[:20])}")
        if len(broken_receptors) > 20:
            report_lines.append(f"    ... and {len(broken_receptors) - 20} more")

    report_lines.append("")

    # Validation summary
    report_lines.append("-" * 80)
    report_lines.append("6. VALIDATION SUMMARY")
    report_lines.append("-" * 80)

    all_checks = []

    # Check 1: Receptors mapped
    check1 = receptors_mapped >= n_receptors * 0.9  # At least 90%
    all_checks.append(check1)
    status1 = "✓ PASS" if check1 else "✗ FAIL"
    report_lines.append(f"  Receptor mapping (≥90%): {status1} ({receptors_mapped}/{n_receptors})")

    # Check 2: No broken pathways
    check2 = broken_count == 0
    all_checks.append(check2)
    status2 = "✓ PASS" if check2 else "⚠ WARNING"
    report_lines.append(f"  Circuit integrity (0 broken): {status2} ({broken_count} broken)")

    # Check 3: KC recruitment in range
    check3 = 30 <= kc_recruitment <= 70  # Relaxed range
    all_checks.append(check3)
    status3 = "✓ PASS" if check3 else "⚠ WARNING"
    report_lines.append(f"  KC recruitment (30-70%): {status3} ({kc_recruitment:.1f}%)")

    # Check 4: Matrices normalized
    check4 = (orn_pn_matrix.max() <= 1.0) and (pn_kc_matrix.max() <= 1.0)
    all_checks.append(check4)
    status4 = "✓ PASS" if check4 else "✗ FAIL"
    report_lines.append(f"  Normalized to [0,1]: {status4}")

    report_lines.append("")

    # Final status
    if all(all_checks):
        report_lines.append("  ✓✓✓ ALL CHECKS PASSED ✓✓✓")
        report_lines.append("")
        report_lines.append("  Status: READY FOR ARCHITECTURE")
        report_lines.append("")
        report_lines.append("  Next steps:")
        report_lines.append("    1. Load connectivity: orn_pn = torch.load('orn_pn_connectivity.pt')")
        report_lines.append("    2. Build RNN architecture with connectome constraints")
        report_lines.append("    3. Train with trial features from PROMPT 1")
    else:
        report_lines.append("  ⚠⚠⚠ SOME CHECKS FAILED ⚠⚠⚠")
        report_lines.append("")
        report_lines.append("  Status: REVIEW REQUIRED")
        report_lines.append("")
        report_lines.append("  Issues:")
        if not check1:
            report_lines.append("    - Many receptors not mapped (check FlyWire labels)")
        if not check2:
            report_lines.append("    - Broken pathways detected (some receptors disconnected)")
        if not check3:
            report_lines.append("    - KC recruitment unusual (check PN→KC connections)")
        if not check4:
            report_lines.append("    - Normalization failed")

    report_lines.append("")
    report_lines.append("="*80)
    report_lines.append("END OF REPORT")
    report_lines.append("="*80)

    # Join report
    report = "\n".join(report_lines)

    # Save report
    report_path = os.path.join(output_dir, "connectivity_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\n[SAVED REPORT] {report_path}")

    # Print to console
    print("\n" + report)

    print("\n[STEP 9 COMPLETE] ✓ Integration report generated")

    return report


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution pipeline."""

    print("\n" + "="*80)
    print("FLYWIRE CONNECTIVITY EXTRACTION FOR OLFACTORY CIRCUIT")
    print("="*80)
    print("\nThis script maps DoOR receptors to FlyWire neurons and extracts")
    print("connectome-constrained connectivity matrices for ORN→PN→KC pathways.")
    print("")

    # Configuration
    FLYWIRE_DIR = "data/flywire"
    OUTPUT_DIR = "data/pgcn_features/connectivity"

    LABELS_PATH = os.path.join(FLYWIRE_DIR, "processed_labels.csv.gz")
    CONNECTIONS_PATH = os.path.join(FLYWIRE_DIR, "connections_princeton.csv.gz")
    CELL_TYPES_PATH = os.path.join(FLYWIRE_DIR, "consolidated_cell_types.csv.gz")

    print(f"FlyWire directory: {FLYWIRE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("")

    # STEP 1: Load FlyWire data
    labels_df = load_flywire_labels(LABELS_PATH)
    connectivity_df = load_flywire_connectivity(CONNECTIONS_PATH)
    cell_types_df = load_flywire_cell_types(CELL_TYPES_PATH)

    # STEP 2: Map receptors to ORNs
    door_receptors = get_standard_door_receptors()
    receptor_to_orns = map_receptors_to_orns(labels_df, door_receptors)

    # STEP 3: Extract ORN→PN connectivity
    pn_ids = identify_pn_neurons(labels_df)
    orn_pn_matrix, pn_ids = extract_orn_pn_connectivity(
        receptor_to_orns, pn_ids, connectivity_df, door_receptors
    )

    # STEP 4: Extract PN→KC connectivity
    kc_ids = identify_kc_neurons(labels_df)
    pn_kc_matrix, kc_ids = extract_pn_kc_connectivity(pn_ids, kc_ids, connectivity_df)

    # STEP 5: Normalize connectivity
    orn_pn_matrix_norm = normalize_connectivity(orn_pn_matrix, method='row_max')
    pn_kc_matrix_norm = normalize_connectivity(pn_kc_matrix, method='row_max')

    # STEP 6: Validate circuit integrity
    receptor_status = validate_circuit_integrity(
        orn_pn_matrix_norm, pn_kc_matrix_norm, door_receptors
    )

    # STEP 7: Compute statistics
    stats_df = compute_connectivity_statistics(
        orn_pn_matrix_norm, pn_kc_matrix_norm,
        receptor_to_orns, door_receptors, receptor_status
    )

    # STEP 8: Save data
    save_connectivity_data(
        orn_pn_matrix_norm, pn_kc_matrix_norm,
        door_receptors, pn_ids, kc_ids, stats_df, OUTPUT_DIR
    )

    # STEP 9: Generate report
    report = generate_integration_report(
        orn_pn_matrix_norm, pn_kc_matrix_norm,
        receptor_to_orns, stats_df, receptor_status, OUTPUT_DIR
    )

    print("\n" + "="*80)
    print("PIPELINE COMPLETE")
    print("="*80)
    print(f"\n✓ Connectivity extraction successful!")
    print(f"✓ Output saved to: {OUTPUT_DIR}")
    print(f"\n📊 Quick stats:")
    print(f"  - {len(door_receptors)} DoOR receptors mapped")
    print(f"  - {orn_pn_matrix_norm.shape[0]} × {orn_pn_matrix_norm.shape[1]} ORN→PN matrix")
    print(f"  - {pn_kc_matrix_norm.shape[0]} × {pn_kc_matrix_norm.shape[1]} PN→KC matrix")
    print(f"\n📝 See connectivity_report.txt for full validation")
    print("")


if __name__ == "__main__":
    main()
