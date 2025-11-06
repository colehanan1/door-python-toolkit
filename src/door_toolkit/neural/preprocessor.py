"""
DoOR Neural Preprocessor
=========================

Main preprocessing interface for neural network training.

This module provides a unified interface for converting DoOR data into
neural network training datasets with sparse encoding, noise augmentation,
and PGCN-compatible export formats.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from door_toolkit.encoder import DoOREncoder
from door_toolkit.neural.concentration_models import ConcentrationResponseModel
from door_toolkit.neural.sparse_encoding import SparseEncoder
from door_toolkit.utils import load_response_matrix

logger = logging.getLogger(__name__)


class DoORNeuralPreprocessor:
    """
    Preprocess DoOR data for neural network training.

    This class provides a complete pipeline for converting DoOR receptor
    response data into formats suitable for neural network training,
    particularly for the PGCN project.

    Attributes:
        door_cache_path: Path to DoOR cache directory
        encoder: DoOREncoder instance
        sparse_encoder: SparseEncoder instance
        concentration_model: ConcentrationResponseModel instance

    Example:
        >>> preprocessor = DoORNeuralPreprocessor("door_cache")
        >>> sparse_data = preprocessor.create_sparse_encoding(sparsity_level=0.05)
        >>> print(f"Sparse shape: {sparse_data.shape}")
        >>> print(f"Sparsity: {(sparse_data > 0).mean():.2%}")
    """

    def __init__(
        self,
        door_cache_path: str,
        n_kc_neurons: int = 2000,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize DoOR neural preprocessor.

        Args:
            door_cache_path: Path to DoOR cache directory
            n_kc_neurons: Number of KC-like neurons for sparse encoding
            random_seed: Random seed for reproducibility

        Raises:
            FileNotFoundError: If cache directory not found
        """
        self.door_cache_path = Path(door_cache_path)
        if not self.door_cache_path.exists():
            raise FileNotFoundError(f"DoOR cache not found: {self.door_cache_path}")

        # Initialize components
        self.encoder = DoOREncoder(str(self.door_cache_path), use_torch=False)
        self.response_matrix = load_response_matrix(str(self.door_cache_path))

        n_receptors = len(self.encoder.receptor_names)
        self.sparse_encoder = SparseEncoder(
            n_input=n_receptors,
            n_output=n_kc_neurons,
            sparsity=0.05,
            random_seed=random_seed,
        )

        self.concentration_model = ConcentrationResponseModel()

        logger.info(
            f"Initialized DoORNeuralPreprocessor: "
            f"{n_receptors} receptors, {n_kc_neurons} KC neurons"
        )

    def create_sparse_encoding(
        self,
        sparsity_level: float = 0.05,
        odorants: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Create sparse KC-like encoding of all odorants.

        Args:
            sparsity_level: Target sparsity (fraction of active neurons)
            odorants: Optional list of odorants (uses all if None)

        Returns:
            Sparse encoding array (n_odorants, n_kc_neurons)

        Example:
            >>> preprocessor = DoORNeuralPreprocessor("door_cache")
            >>> sparse_data = preprocessor.create_sparse_encoding(sparsity_level=0.05)
            >>> print(f"Shape: {sparse_data.shape}")
            >>> print(f"Sparsity: {(sparse_data > 0).mean():.2%}")
        """
        logger.info(f"Creating sparse encoding (sparsity={sparsity_level:.1%})")

        # Update sparsity if different
        if sparsity_level != self.sparse_encoder.sparsity:
            self.sparse_encoder.sparsity = sparsity_level

        # Get odorants to encode
        if odorants is None:
            odorants = self.encoder.odorant_names

        # Encode all odorants to ORN responses
        logger.debug(f"Encoding {len(odorants)} odorants to ORN responses")
        orn_responses = []

        for odorant in odorants:
            try:
                response = self.encoder.encode(odorant)
                # Replace NaN with 0
                response = np.nan_to_num(response, nan=0.0)
                orn_responses.append(response)
            except Exception as e:
                logger.warning(f"Could not encode {odorant}: {e}")
                continue

        orn_responses = np.array(orn_responses)

        # Create sparse encoding
        logger.debug(f"Creating sparse KC encoding from {len(orn_responses)} ORN patterns")
        sparse_encoding = self.sparse_encoder.encode(orn_responses)

        actual_sparsity = (sparse_encoding > 0).mean()
        logger.info(
            f"Created sparse encoding: shape={sparse_encoding.shape}, "
            f"sparsity={actual_sparsity:.2%}"
        )

        return sparse_encoding

    def generate_noise_augmented_responses(
        self,
        noise_types: Optional[List[str]] = None,
        n_augmentations: int = 5,
        noise_level: float = 0.1,
        odorants: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Generate noise-augmented response dataset.

        Args:
            noise_types: Types of noise to apply
            n_augmentations: Number of augmented copies per sample
            noise_level: Noise level
            odorants: Optional list of odorants

        Returns:
            Tuple of (augmented_orn, augmented_kc, odorant_labels)

        Example:
            >>> preprocessor = DoORNeuralPreprocessor("door_cache")
            >>> orn, kc, labels = preprocessor.generate_noise_augmented_responses(
            ...     n_augmentations=3
            ... )
            >>> print(f"Generated {len(orn)} augmented samples")
        """
        logger.info(f"Generating noise-augmented dataset ({n_augmentations}x)")

        if noise_types is None:
            noise_types = ["gaussian", "dropout", "poisson"]

        # Get odorants
        if odorants is None:
            odorants = self.encoder.odorant_names

        # Encode to ORN responses
        orn_responses = []
        valid_odorants = []

        for odorant in odorants:
            try:
                response = self.encoder.encode(odorant)
                response = np.nan_to_num(response, nan=0.0)
                orn_responses.append(response)
                valid_odorants.append(odorant)
            except Exception as e:
                logger.warning(f"Could not encode {odorant}: {e}")
                continue

        orn_responses = np.array(orn_responses)

        # Generate augmented data
        aug_orn, aug_kc = self.sparse_encoder.generate_augmented_dataset(
            orn_responses,
            n_augmentations=n_augmentations,
            noise_types=noise_types,
            noise_level=noise_level,
        )

        # Create labels (repeat each odorant n_augmentations+1 times)
        aug_labels = []
        for odorant in valid_odorants:
            aug_labels.extend([odorant] * (n_augmentations + 1))

        logger.info(
            f"Generated {len(aug_orn)} augmented samples from "
            f"{len(valid_odorants)} odorants"
        )

        return aug_orn, aug_kc, aug_labels

    def export_pgcn_dataset(
        self,
        output_dir: str,
        format: str = "pytorch",
        include_sparse: bool = True,
        include_metadata: bool = True,
    ) -> None:
        """
        Export complete dataset for PGCN training.

        Args:
            output_dir: Output directory path
            format: Export format ('pytorch', 'numpy', or 'h5')
            include_sparse: Include sparse KC encodings
            include_metadata: Include odorant metadata

        Example:
            >>> preprocessor = DoORNeuralPreprocessor("door_cache")
            >>> preprocessor.export_pgcn_dataset(
            ...     "pgcn_data",
            ...     format="pytorch",
            ...     include_sparse=True
            ... )
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting PGCN dataset to {output_dir} (format={format})")

        # Get all odorants
        odorants = self.encoder.odorant_names

        # Encode to ORN responses
        logger.debug("Encoding odorants to ORN responses")
        orn_responses = []
        valid_odorants = []

        for odorant in odorants:
            try:
                response = self.encoder.encode(odorant)
                response = np.nan_to_num(response, nan=0.0)
                orn_responses.append(response)
                valid_odorants.append(odorant)
            except Exception:
                continue

        orn_responses = np.array(orn_responses)

        # Create sparse encoding if requested
        if include_sparse:
            logger.debug("Creating sparse KC encodings")
            kc_responses = self.sparse_encoder.encode(orn_responses)
        else:
            kc_responses = None

        # Export based on format
        if format == "pytorch":
            self._export_pytorch(
                output_dir, orn_responses, kc_responses, valid_odorants
            )

        elif format == "numpy":
            self._export_numpy(
                output_dir, orn_responses, kc_responses, valid_odorants
            )

        elif format == "h5":
            self._export_h5(
                output_dir, orn_responses, kc_responses, valid_odorants
            )

        else:
            raise ValueError(f"Unknown format: {format}")

        # Export metadata if requested
        if include_metadata:
            self._export_metadata(output_dir, valid_odorants)

        logger.info(f"Successfully exported PGCN dataset to {output_dir}")

    def _export_pytorch(
        self,
        output_dir: Path,
        orn_responses: np.ndarray,
        kc_responses: Optional[np.ndarray],
        odorants: List[str],
    ) -> None:
        """Export dataset in PyTorch format."""
        try:
            import torch
        except ImportError:
            raise ImportError(
                "PyTorch required for pytorch export. "
                "Install with: pip install torch"
            )

        # Save as PyTorch tensors
        torch.save(
            {
                "orn_responses": torch.from_numpy(orn_responses).float(),
                "kc_responses": torch.from_numpy(kc_responses).float()
                if kc_responses is not None
                else None,
                "odorants": odorants,
                "receptor_names": self.encoder.receptor_names,
                "n_receptors": len(self.encoder.receptor_names),
                "n_kc_neurons": self.sparse_encoder.n_output if kc_responses is not None else None,
            },
            output_dir / "pgcn_dataset.pt",
        )

        logger.debug(f"Exported PyTorch dataset: {output_dir / 'pgcn_dataset.pt'}")

    def _export_numpy(
        self,
        output_dir: Path,
        orn_responses: np.ndarray,
        kc_responses: Optional[np.ndarray],
        odorants: List[str],
    ) -> None:
        """Export dataset in NumPy format."""
        np.save(output_dir / "orn_responses.npy", orn_responses)

        if kc_responses is not None:
            np.save(output_dir / "kc_responses.npy", kc_responses)

        # Save odorant names
        with open(output_dir / "odorants.txt", "w") as f:
            f.write("\n".join(odorants))

        # Save receptor names
        with open(output_dir / "receptors.txt", "w") as f:
            f.write("\n".join(self.encoder.receptor_names))

        logger.debug(f"Exported NumPy dataset to {output_dir}")

    def _export_h5(
        self,
        output_dir: Path,
        orn_responses: np.ndarray,
        kc_responses: Optional[np.ndarray],
        odorants: List[str],
    ) -> None:
        """Export dataset in HDF5 format."""
        try:
            import h5py
        except ImportError:
            raise ImportError(
                "h5py required for HDF5 export. Install with: pip install h5py"
            )

        with h5py.File(output_dir / "pgcn_dataset.h5", "w") as f:
            f.create_dataset("orn_responses", data=orn_responses)

            if kc_responses is not None:
                f.create_dataset("kc_responses", data=kc_responses)

            # Store strings as fixed-length
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset("odorants", data=odorants, dtype=dt)
            f.create_dataset(
                "receptors", data=self.encoder.receptor_names, dtype=dt
            )

        logger.debug(f"Exported HDF5 dataset: {output_dir / 'pgcn_dataset.h5'}")

    def _export_metadata(self, output_dir: Path, odorants: List[str]) -> None:
        """Export dataset metadata."""
        metadata = {
            "n_odorants": len(odorants),
            "n_receptors": len(self.encoder.receptor_names),
            "n_kc_neurons": self.sparse_encoder.n_output,
            "sparsity": self.sparse_encoder.sparsity,
            "receptor_names": self.encoder.receptor_names,
            "odorant_names": odorants[:100],  # First 100 for brevity
        }

        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.debug(f"Exported metadata: {output_dir / 'metadata.json'}")

    def create_training_validation_split(
        self,
        train_fraction: float = 0.8,
        random_seed: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        Create train/validation split of odorants.

        Args:
            train_fraction: Fraction of data for training
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_odorants, val_odorants)

        Example:
            >>> preprocessor = DoORNeuralPreprocessor("door_cache")
            >>> train, val = preprocessor.create_training_validation_split(0.8)
            >>> print(f"Train: {len(train)}, Val: {len(val)}")
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        odorants = np.array(self.encoder.odorant_names)
        n_train = int(len(odorants) * train_fraction)

        indices = np.random.permutation(len(odorants))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        train_odorants = odorants[train_indices].tolist()
        val_odorants = odorants[val_indices].tolist()

        logger.info(
            f"Created train/val split: {len(train_odorants)} train, "
            f"{len(val_odorants)} val"
        )

        return train_odorants, val_odorants

    def get_dataset_statistics(self) -> Dict:
        """
        Get comprehensive statistics about the dataset.

        Returns:
            Dictionary with dataset statistics

        Example:
            >>> preprocessor = DoORNeuralPreprocessor("door_cache")
            >>> stats = preprocessor.get_dataset_statistics()
            >>> print(f"Coverage: {stats['mean_receptor_coverage']:.1%}")
        """
        # Encode all odorants
        orn_responses = []
        for odorant in self.encoder.odorant_names:
            try:
                response = self.encoder.encode(odorant)
                orn_responses.append(response)
            except Exception:
                continue

        orn_responses = np.array(orn_responses)

        # Calculate statistics
        stats = {
            "n_odorants": len(self.encoder.odorant_names),
            "n_receptors": len(self.encoder.receptor_names),
            "mean_response": float(np.nanmean(orn_responses)),
            "std_response": float(np.nanstd(orn_responses)),
            "mean_receptor_coverage": float((~np.isnan(orn_responses)).mean()),
            "sparsity_at_threshold_0.3": float((orn_responses > 0.3).mean()),
            "max_response": float(np.nanmax(orn_responses)),
            "min_response": float(np.nanmin(orn_responses)),
        }

        return stats


def create_pgcn_training_dataset(
    door_cache_path: str,
    output_dir: str,
    n_kc_neurons: int = 2000,
    sparsity: float = 0.05,
    n_augmentations: int = 5,
    format: str = "pytorch",
) -> None:
    """
    Convenience function to create complete PGCN training dataset.

    Args:
        door_cache_path: Path to DoOR cache
        output_dir: Output directory
        n_kc_neurons: Number of KC neurons
        sparsity: Target sparsity
        n_augmentations: Number of augmentations per sample
        format: Export format

    Example:
        >>> create_pgcn_training_dataset(
        ...     "door_cache",
        ...     "pgcn_training_data",
        ...     n_kc_neurons=2000,
        ...     sparsity=0.05
        ... )
    """
    preprocessor = DoORNeuralPreprocessor(door_cache_path, n_kc_neurons=n_kc_neurons)

    # Generate augmented data
    aug_orn, aug_kc, labels = preprocessor.generate_noise_augmented_responses(
        n_augmentations=n_augmentations
    )

    # Export
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if format == "pytorch":
        import torch

        torch.save(
            {
                "orn_responses": torch.from_numpy(aug_orn).float(),
                "kc_responses": torch.from_numpy(aug_kc).float(),
                "labels": labels,
                "n_augmentations": n_augmentations,
            },
            output_dir / "augmented_dataset.pt",
        )
    else:
        np.save(output_dir / "augmented_orn.npy", aug_orn)
        np.save(output_dir / "augmented_kc.npy", aug_kc)

        with open(output_dir / "labels.txt", "w") as f:
            f.write("\n".join(labels))

    logger.info(f"Created PGCN training dataset: {output_dir}")
