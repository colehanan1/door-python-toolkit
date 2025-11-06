"""
DoOR Toolkit Utilities
=======================

Helper functions for working with DoOR data.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_response_matrix(
    cache_path: str,
    format: str = "parquet"
) -> pd.DataFrame:
    """
    Load DoOR response matrix from cache.
    
    Args:
        cache_path: Path to DoOR cache directory
        format: File format - 'parquet', 'csv', or 'numpy'
        
    Returns:
        DataFrame with odorants as rows, receptors as columns
        
    Example:
        >>> df = load_response_matrix("door_cache")
        >>> print(df.shape)
        (693, 78)
    """
    cache_dir = Path(cache_path)
    
    if format == "parquet":
        return pd.read_parquet(cache_dir / "response_matrix_norm.parquet")
    elif format == "csv":
        return pd.read_csv(cache_dir / "response_matrix_norm.csv", index_col=0)
    elif format == "numpy":
        arr = np.load(cache_dir / "response_matrix_norm.npy")
        receptors = pd.read_csv(cache_dir / "receptor_index.csv")['receptor']
        odorants = pd.read_csv(cache_dir / "odorant_index.csv")['odorant']
        return pd.DataFrame(arr, index=odorants, columns=receptors)
    else:
        raise ValueError(f"Unknown format: {format}. Use 'parquet', 'csv', or 'numpy'")


def load_odor_metadata(cache_path: str) -> pd.DataFrame:
    """
    Load odorant chemical metadata.
    
    Args:
        cache_path: Path to DoOR cache directory
        
    Returns:
        DataFrame with odorant metadata (CAS, SMILES, MW, etc.)
    """
    cache_dir = Path(cache_path)
    return pd.read_parquet(cache_dir / "odor_metadata.parquet")


def list_odorants(
    cache_path: str,
    pattern: Optional[str] = None
) -> List[str]:
    """
    List available odorant names in cache.
    
    Args:
        cache_path: Path to DoOR cache directory
        pattern: Optional substring filter
        
    Returns:
        List of odorant names
        
    Example:
        >>> acetates = list_odorants("door_cache", pattern="acetate")
        >>> print(len(acetates))
        36
    """
    meta = load_odor_metadata(cache_path)
    names = meta['Name'].dropna().tolist()
    
    if pattern:
        pattern_lower = pattern.lower()
        names = [n for n in names if pattern_lower in str(n).lower()]
    
    return sorted(names)


def get_receptor_info(cache_path: str) -> pd.DataFrame:
    """
    Get receptor coverage statistics.
    
    Args:
        cache_path: Path to DoOR cache directory
        
    Returns:
        DataFrame with receptor names and coverage counts
    """
    response_df = load_response_matrix(cache_path)
    
    coverage = response_df.notna().sum()
    pct_coverage = (coverage / len(response_df)) * 100
    
    return pd.DataFrame({
        'receptor': coverage.index,
        'n_odorants_tested': coverage.values,
        'coverage_pct': pct_coverage.values
    }).sort_values('n_odorants_tested', ascending=False)


def find_similar_odorants(
    target_odor: str,
    cache_path: str,
    top_k: int = 10,
    method: str = "correlation"
) -> List[Tuple[str, float]]:
    """
    Find odorants with similar receptor response patterns.
    
    Args:
        target_odor: Odorant name to find similar compounds for
        cache_path: Path to DoOR cache directory
        top_k: Number of similar odorants to return
        method: Similarity metric - 'correlation' or 'euclidean'
        
    Returns:
        List of (odorant_name, similarity_score) tuples
        
    Example:
        >>> similar = find_similar_odorants("acetic acid", "door_cache", top_k=5)
        >>> for name, score in similar:
        ...     print(f"{name}: {score:.3f}")
    """
    response_df = load_response_matrix(cache_path)
    meta = load_odor_metadata(cache_path)
    
    # Find target odor InChIKey
    target_key = meta[meta['Name'].str.lower() == target_odor.lower()].index
    if len(target_key) == 0:
        raise KeyError(f"Odorant '{target_odor}' not found")
    
    target_key = target_key[0]
    target_response = response_df.loc[target_key]
    
    # Compute similarity
    if method == "correlation":
        similarities = response_df.corrwith(target_response, axis=1)
    elif method == "euclidean":
        # Negative distance (higher = more similar)
        distances = np.sqrt(((response_df - target_response) ** 2).sum(axis=1))
        similarities = -distances
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get top-k (excluding self)
    top_indices = similarities.nlargest(top_k + 1).index[1:]  # Skip self
    
    # Map back to names
    results = []
    for idx in top_indices:
        if idx in meta.index:
            name = meta.loc[idx, 'Name']
            if pd.notna(name):
                results.append((name, float(similarities[idx])))
    
    return results[:top_k]


def export_subset(
    cache_path: str,
    output_path: str,
    odorants: Optional[List[str]] = None,
    receptors: Optional[List[str]] = None
):
    """
    Export a subset of DoOR data.
    
    Args:
        cache_path: Path to DoOR cache directory
        output_path: Output file path (CSV or parquet)
        odorants: Optional list of odorant names to include
        receptors: Optional list of receptor names to include
        
    Example:
        >>> export_subset(
        ...     "door_cache",
        ...     "acetates.csv",
        ...     odorants=list_odorants("door_cache", "acetate")
        ... )
    """
    response_df = load_response_matrix(cache_path)
    meta = load_odor_metadata(cache_path)
    
    # Filter by odorants
    if odorants:
        # Map names to InChIKeys
        name_to_key = {v.lower(): k for k, v in meta['Name'].items()}
        keys = [name_to_key[o.lower()] for o in odorants if o.lower() in name_to_key]
        response_df = response_df.loc[keys]
    
    # Filter by receptors
    if receptors:
        response_df = response_df[receptors]
    
    # Export
    output_path = Path(output_path)
    if output_path.suffix == '.csv':
        response_df.to_csv(output_path)
    elif output_path.suffix == '.parquet':
        response_df.to_parquet(output_path)
    else:
        raise ValueError("Output path must be .csv or .parquet")
    
    logger.info(f"Exported {response_df.shape} subset to {output_path}")


def validate_cache(cache_path: str) -> bool:
    """
    Validate that DoOR cache is complete and usable.
    
    Args:
        cache_path: Path to DoOR cache directory
        
    Returns:
        True if valid, False otherwise
    """
    cache_dir = Path(cache_path)
    
    required_files = [
        "response_matrix_norm.parquet",
        "odor_metadata.parquet",
        "receptor_index.csv",
        "odorant_index.csv",
        "metadata.json"
    ]
    
    for fname in required_files:
        if not (cache_dir / fname).exists():
            logger.error(f"Missing required file: {fname}")
            return False
    
    try:
        # Try loading
        response_df = load_response_matrix(cache_path)
        meta = load_odor_metadata(cache_path)
        
        # Basic validation
        assert response_df.shape[0] > 0, "No odorants in response matrix"
        assert response_df.shape[1] > 0, "No receptors in response matrix"
        assert len(meta) > 0, "No metadata entries"
        
        logger.info(f"Cache validation passed: {response_df.shape} data points")
        return True
        
    except Exception as e:
        logger.error(f"Cache validation failed: {e}")
        return False
