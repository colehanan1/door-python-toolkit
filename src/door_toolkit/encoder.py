"""
DoOR Encoder Module
===================

Encode odorant names to neural activation patterns using DoOR database.
"""

import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class DoOREncoder:
    """
    Encode odorant names to projection neuron (PN) activation vectors.

    Uses empirical DoOR database measurements to map odorant names to
    glomerular response patterns in Drosophila.

    Attributes:
        cache_path: Path to extracted DoOR cache directory
        n_channels: Number of receptor channels (78 for DoOR v2.0)
        receptor_names: List of receptor identifiers
        odorant_names: List of available odorant names

    Example:
        >>> encoder = DoOREncoder("data/door_cache")
        >>> pn_vector = encoder.encode("acetic acid")
        >>> print(pn_vector.shape)
        (78,)
    """

    def __init__(self, cache_path: str = "door_cache", use_torch: bool = True):
        """
        Initialize DoOR encoder.

        Args:
            cache_path: Path to extracted DoOR cache directory
            use_torch: Return torch tensors if available (default: True)

        Raises:
            FileNotFoundError: If cache directory doesn't exist
            ImportError: If torch requested but not installed
        """
        self.cache_path = Path(cache_path)
        self.use_torch = use_torch and TORCH_AVAILABLE

        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"DoOR cache not found at {self.cache_path}. "
                f"Run DoORExtractor first to create cache."
            )

        if use_torch and not TORCH_AVAILABLE:
            logger.warning("PyTorch not available, falling back to NumPy")
            self.use_torch = False

        # Load response matrix (InChIKey → receptor responses)
        self.response_matrix = pd.read_parquet(self.cache_path / "response_matrix_norm.parquet")

        # Load metadata (InChIKey → Name, CAS, etc.)
        self.metadata = pd.read_parquet(self.cache_path / "odor_metadata.parquet")

        # Create name lookup (case-insensitive)
        self.name_to_inchikey = {
            name.lower(): inchikey
            for inchikey, name in self.metadata["Name"].items()
            if pd.notna(name)
        }

        # Expose attributes
        self.n_channels = self.response_matrix.shape[1]
        self.receptor_names = self.response_matrix.columns.tolist()
        self.odorant_names = list(self.name_to_inchikey.keys())

        logger.info(
            f"[DoOREncoder] Loaded {len(self.odorant_names)} odorants, "
            f"{self.n_channels} receptor channels"
        )

    def encode(self, odor_name: str, fill_missing: float = 0.0):
        """
        Encode single odorant to PN activation vector.

        Args:
            odor_name: Odorant name (case-insensitive)
            fill_missing: Value for missing receptor responses (default: 0.0)

        Returns:
            NumPy array or torch.Tensor (if use_torch=True) of shape (n_channels,)
            with PN activations in range [0, 1]

        Raises:
            KeyError: If odorant not found in database

        Example:
            >>> encoder = DoOREncoder()
            >>> pn = encoder.encode("acetic acid")
            >>> print(pn.shape)
            (78,)
        """
        name_lower = odor_name.lower()

        if name_lower not in self.name_to_inchikey:
            raise KeyError(
                f"Odorant '{odor_name}' not found in DoOR database. "
                f"Use list_available_odorants() to see options."
            )

        # Get InChIKey and response
        inchikey = self.name_to_inchikey[name_lower]
        pn_activation = self.response_matrix.loc[inchikey].fillna(fill_missing).values

        if self.use_torch:
            return torch.from_numpy(pn_activation).float()
        return pn_activation.astype(np.float32)

    def batch_encode(self, odor_names: List[str], fill_missing: float = 0.0):
        """
        Encode batch of odorants.

        Args:
            odor_names: List of odorant names
            fill_missing: Value for missing responses

        Returns:
            NumPy array or torch.Tensor of shape (batch_size, n_channels)

        Example:
            >>> encoder = DoOREncoder()
            >>> odors = ["acetic acid", "1-pentanol", "butyric acid"]
            >>> pn_batch = encoder.batch_encode(odors)
            >>> print(pn_batch.shape)
            (3, 78)
        """
        batch = [self.encode(name, fill_missing) for name in odor_names]

        if self.use_torch:
            return torch.stack(batch)
        return np.stack(batch)

    def encode_by_inchikey(self, inchikey: str, fill_missing: float = 0.0):
        """
        Encode odorant by InChIKey directly (faster than name lookup).

        Args:
            inchikey: InChIKey identifier
            fill_missing: Value for missing responses

        Returns:
            NumPy array or torch.Tensor of shape (n_channels,)
        """
        if inchikey not in self.response_matrix.index:
            raise KeyError(f"InChIKey '{inchikey}' not in database")

        pn_activation = self.response_matrix.loc[inchikey].fillna(fill_missing).values

        if self.use_torch:
            return torch.from_numpy(pn_activation).float()
        return pn_activation.astype(np.float32)

    def list_available_odorants(self, pattern: Optional[str] = None) -> List[str]:
        """
        List all available odorant names.

        Args:
            pattern: Optional substring to filter by (case-insensitive)

        Returns:
            Sorted list of odorant names

        Example:
            >>> encoder = DoOREncoder()
            >>> acetates = encoder.list_available_odorants("acetate")
            >>> print(len(acetates))
            36
        """
        names = self.odorant_names

        if pattern:
            pattern_lower = pattern.lower()
            names = [n for n in names if pattern_lower in n]

        return sorted(names)

    def get_receptor_coverage(self, odor_name: str) -> Dict:
        """
        Get coverage statistics for an odorant.

        Args:
            odor_name: Odorant name

        Returns:
            Dictionary with coverage stats:
                - n_tested: Number of receptors tested
                - n_active: Number showing response > 0.1
                - max_response: Maximum response value
                - top_receptors: Top 5 responding receptors

        Example:
            >>> encoder = DoOREncoder()
            >>> stats = encoder.get_receptor_coverage("acetic acid")
            >>> print(stats['n_tested'])
            45
        """
        name_lower = odor_name.lower()
        if name_lower not in self.name_to_inchikey:
            raise KeyError(f"Odorant '{odor_name}' not found")

        inchikey = self.name_to_inchikey[name_lower]
        response = self.response_matrix.loc[inchikey]

        return {
            "n_tested": int(response.notna().sum()),
            "n_active": int((response > 0.1).sum()),
            "max_response": float(response.max()),
            "mean_response": float(response.mean()),
            "top_receptors": response.nlargest(5).to_dict(),
        }

    def get_odor_metadata(self, odor_name: str) -> Dict:
        """
        Get chemical metadata for an odorant.

        Args:
            odor_name: Odorant name

        Returns:
            Dictionary with metadata (CAS, Formula, MW, SMILES, etc.)
        """
        name_lower = odor_name.lower()
        if name_lower not in self.name_to_inchikey:
            raise KeyError(f"Odorant '{odor_name}' not found")

        inchikey = self.name_to_inchikey[name_lower]
        meta = self.metadata.loc[inchikey].to_dict()

        # Convert NaN to None for cleaner output
        return {k: (v if pd.notna(v) else None) for k, v in meta.items()}
