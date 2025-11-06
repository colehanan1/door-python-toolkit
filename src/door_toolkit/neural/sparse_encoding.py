"""
Sparse Encoding Module
=======================

Generate sparse representations mimicking Kenyon Cell (KC) encoding.

This module creates sparse distributed representations inspired by the
Drosophila mushroom body, where KCs exhibit sparse activation patterns
(~5% active neurons).
"""

import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SparseEncoder:
    """
    Create sparse encodings of odorant response patterns.

    This class generates sparse distributed representations that mimic
    Kenyon Cell activation patterns in the Drosophila mushroom body.

    Attributes:
        n_input: Number of input dimensions (ORN responses)
        n_output: Number of output dimensions (KC-like neurons)
        sparsity: Target sparsity level (fraction of active neurons)
        connection_probability: ORN→KC connection probability

    Example:
        >>> encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)
        >>> orn_responses = np.random.randn(78)
        >>> kc_responses = encoder.encode(orn_responses)
        >>> print(f"Sparsity: {(kc_responses > 0).mean():.2%}")
    """

    def __init__(
        self,
        n_input: int = 78,
        n_output: int = 2000,
        sparsity: float = 0.05,
        connection_probability: float = 0.5,
        random_seed: Optional[int] = None,
    ):
        """
        Initialize sparse encoder.

        Args:
            n_input: Number of input neurons (ORNs)
            n_output: Number of output neurons (KCs)
            sparsity: Target sparsity (fraction active)
            connection_probability: ORN→KC connection probability
            random_seed: Random seed for reproducibility

        Example:
            >>> encoder = SparseEncoder(n_input=78, n_output=2000, sparsity=0.05)
        """
        self.n_input = n_input
        self.n_output = n_output
        self.sparsity = sparsity
        self.connection_probability = connection_probability

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize random projection matrix (ORN → KC connections)
        self.weights = self._initialize_weights()

        logger.info(
            f"Initialized SparseEncoder: {n_input} inputs → {n_output} outputs "
            f"(sparsity={sparsity:.1%})"
        )

    def _initialize_weights(self) -> np.ndarray:
        """
        Initialize sparse random projection weights.

        Creates a sparse connection matrix mimicking biological connectivity
        where each KC receives input from ~50% of ORNs.

        Returns:
            Weight matrix (n_output x n_input)
        """
        # Create sparse random connections
        weights = np.zeros((self.n_output, self.n_input))

        for kc_idx in range(self.n_output):
            # Randomly select which ORNs connect to this KC
            connections = np.random.rand(self.n_input) < self.connection_probability

            # Random weights for connected ORNs
            weights[kc_idx, connections] = np.random.randn(connections.sum()) * 0.1

        logger.debug(
            f"Initialized weights: {(weights != 0).sum()} / "
            f"{weights.size} connections "
            f"({(weights != 0).mean():.1%})"
        )

        return weights

    def encode(
        self,
        orn_responses: np.ndarray,
        method: str = "winner-take-all",
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Encode ORN responses to sparse KC representation.

        Args:
            orn_responses: ORN response vector (n_input,) or (batch, n_input)
            method: Sparsification method ('winner-take-all' or 'threshold')
            threshold: Optional threshold for threshold method

        Returns:
            Sparse KC responses (n_output,) or (batch, n_output)

        Example:
            >>> encoder = SparseEncoder(n_input=78, n_output=2000)
            >>> orn = np.random.randn(78)
            >>> kc = encoder.encode(orn, method="winner-take-all")
            >>> print(f"Active KCs: {(kc > 0).sum()} / {len(kc)}")
        """
        # Handle batch dimension
        if orn_responses.ndim == 1:
            orn_responses = orn_responses.reshape(1, -1)
            squeeze_output = True
        else:
            squeeze_output = False

        batch_size = orn_responses.shape[0]

        # Linear projection: KC = W * ORN
        kc_activations = np.dot(orn_responses, self.weights.T)

        # Apply sparsification
        if method == "winner-take-all":
            sparse_kc = self._winner_take_all(kc_activations, self.sparsity)

        elif method == "threshold":
            if threshold is None:
                # Auto-compute threshold to achieve target sparsity
                threshold = np.percentile(kc_activations, (1 - self.sparsity) * 100)
            sparse_kc = np.where(kc_activations > threshold, kc_activations, 0)

        else:
            raise ValueError(f"Unknown sparsification method: {method}")

        if squeeze_output:
            sparse_kc = sparse_kc.squeeze(0)

        return sparse_kc

    def _winner_take_all(self, activations: np.ndarray, sparsity: float) -> np.ndarray:
        """
        Winner-take-all sparsification.

        Keeps only the top-k most active neurons, where k = sparsity * n_output.

        Args:
            activations: Pre-activation values (batch, n_output)
            sparsity: Target sparsity

        Returns:
            Sparse activations
        """
        batch_size = activations.shape[0]
        k = int(self.n_output * sparsity)

        sparse_activations = np.zeros_like(activations)

        for i in range(batch_size):
            # Find top-k indices
            top_k_indices = np.argpartition(activations[i], -k)[-k:]
            sparse_activations[i, top_k_indices] = activations[i, top_k_indices]

        return sparse_activations

    def encode_batch(
        self,
        orn_batch: np.ndarray,
        method: str = "winner-take-all",
    ) -> np.ndarray:
        """
        Encode batch of ORN responses.

        Args:
            orn_batch: Batch of ORN responses (batch_size, n_input)
            method: Sparsification method

        Returns:
            Sparse KC responses (batch_size, n_output)

        Example:
            >>> encoder = SparseEncoder(n_input=78, n_output=2000)
            >>> orn_batch = np.random.randn(100, 78)
            >>> kc_batch = encoder.encode_batch(orn_batch)
            >>> print(f"Batch sparsity: {(kc_batch > 0).mean():.2%}")
        """
        return self.encode(orn_batch, method=method)

    def get_sparsity_stats(self, kc_responses: np.ndarray) -> dict:
        """
        Compute sparsity statistics.

        Args:
            kc_responses: KC response array

        Returns:
            Dictionary with sparsity statistics

        Example:
            >>> encoder = SparseEncoder()
            >>> kc = encoder.encode(np.random.randn(78))
            >>> stats = encoder.get_sparsity_stats(kc)
            >>> print(f"Sparsity: {stats['sparsity']:.2%}")
        """
        if kc_responses.ndim == 1:
            kc_responses = kc_responses.reshape(1, -1)

        active_mask = kc_responses > 0

        stats = {
            "sparsity": float(active_mask.mean()),
            "n_active_mean": float(active_mask.sum(axis=1).mean()),
            "n_active_std": float(active_mask.sum(axis=1).std()),
            "activation_mean": (
                float(kc_responses[active_mask].mean()) if active_mask.any() else 0.0
            ),
            "activation_std": float(kc_responses[active_mask].std()) if active_mask.any() else 0.0,
        }

        return stats

    def add_noise(
        self,
        kc_responses: np.ndarray,
        noise_type: str = "gaussian",
        noise_level: float = 0.1,
    ) -> np.ndarray:
        """
        Add noise to sparse representations for robust training.

        Args:
            kc_responses: KC response array
            noise_type: Type of noise ('gaussian', 'dropout', 'poisson')
            noise_level: Noise level/dropout probability

        Returns:
            Noisy KC responses

        Example:
            >>> encoder = SparseEncoder()
            >>> kc = encoder.encode(np.random.randn(78))
            >>> noisy_kc = encoder.add_noise(kc, noise_type="gaussian", noise_level=0.1)
        """
        noisy_responses = kc_responses.copy()

        if noise_type == "gaussian":
            # Add Gaussian noise to active neurons
            active_mask = kc_responses > 0
            noise = np.random.normal(0, noise_level, size=kc_responses.shape)
            noisy_responses = kc_responses + noise * active_mask

        elif noise_type == "dropout":
            # Randomly zero out some active neurons
            active_mask = kc_responses > 0
            dropout_mask = np.random.rand(*kc_responses.shape) > noise_level
            noisy_responses = kc_responses * dropout_mask

        elif noise_type == "poisson":
            # Poisson noise (biological spiking noise)
            active_mask = kc_responses > 0
            scaled = kc_responses[active_mask] * 100
            noisy_scaled = np.random.poisson(scaled)
            noisy_responses = kc_responses.copy()
            noisy_responses[active_mask] = noisy_scaled / 100.0

        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

        # Keep sparsity (zero out negative values)
        noisy_responses = np.maximum(noisy_responses, 0)

        return noisy_responses

    def generate_augmented_dataset(
        self,
        orn_data: np.ndarray,
        n_augmentations: int = 5,
        noise_types: Optional[list] = None,
        noise_level: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate augmented dataset with multiple noise realizations.

        Args:
            orn_data: ORN response data (n_samples, n_input)
            n_augmentations: Number of augmented copies per sample
            noise_types: List of noise types to use
            noise_level: Noise level

        Returns:
            Tuple of (augmented_orn, augmented_kc)

        Example:
            >>> encoder = SparseEncoder(n_input=78, n_output=2000)
            >>> orn_data = np.random.randn(100, 78)
            >>> aug_orn, aug_kc = encoder.generate_augmented_dataset(orn_data, n_augmentations=3)
            >>> print(f"Augmented from {len(orn_data)} to {len(aug_orn)} samples")
        """
        if noise_types is None:
            noise_types = ["gaussian", "dropout", "poisson"]

        n_samples = orn_data.shape[0]
        augmented_orn = []
        augmented_kc = []

        for i in range(n_samples):
            orn_sample = orn_data[i]

            # Original encoding
            kc_sample = self.encode(orn_sample)
            augmented_orn.append(orn_sample)
            augmented_kc.append(kc_sample)

            # Augmented versions
            for j in range(n_augmentations):
                noise_type = noise_types[j % len(noise_types)]

                # Add noise to ORN input
                orn_noisy = orn_sample + np.random.normal(
                    0, noise_level * 0.5, size=orn_sample.shape
                )

                # Encode and add KC noise
                kc_noisy = self.encode(orn_noisy)
                kc_noisy = self.add_noise(kc_noisy, noise_type=noise_type, noise_level=noise_level)

                augmented_orn.append(orn_noisy)
                augmented_kc.append(kc_noisy)

        augmented_orn = np.array(augmented_orn)
        augmented_kc = np.array(augmented_kc)

        logger.info(
            f"Generated augmented dataset: {n_samples} → "
            f"{len(augmented_orn)} samples ({n_augmentations}x augmentation)"
        )

        return augmented_orn, augmented_kc

    def export_weights(self, output_path: str) -> None:
        """
        Export weight matrix to file.

        Args:
            output_path: Output file path (NPY format)

        Example:
            >>> encoder = SparseEncoder()
            >>> encoder.export_weights("kc_weights.npy")
        """
        from pathlib import Path

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.save(output_path, self.weights)
        logger.info(f"Exported weights to {output_path}")

    def load_weights(self, weights_path: str) -> None:
        """
        Load weight matrix from file.

        Args:
            weights_path: Path to weights file (NPY format)

        Example:
            >>> encoder = SparseEncoder()
            >>> encoder.load_weights("kc_weights.npy")
        """
        self.weights = np.load(weights_path)
        self.n_output, self.n_input = self.weights.shape
        logger.info(f"Loaded weights from {weights_path}: " f"{self.n_input} → {self.n_output}")


def create_kc_like_encoding(
    orn_responses: np.ndarray,
    n_kcs: int = 2000,
    sparsity: float = 0.05,
) -> np.ndarray:
    """
    Convenience function to create KC-like sparse encoding.

    Args:
        orn_responses: ORN response array
        n_kcs: Number of KC neurons
        sparsity: Target sparsity

    Returns:
        Sparse KC responses

    Example:
        >>> orn = np.random.randn(100, 78)
        >>> kc = create_kc_like_encoding(orn, n_kcs=2000, sparsity=0.05)
        >>> print(f"Shape: {kc.shape}, Sparsity: {(kc > 0).mean():.2%}")
    """
    encoder = SparseEncoder(
        n_input=orn_responses.shape[-1],
        n_output=n_kcs,
        sparsity=sparsity,
    )
    return encoder.encode(orn_responses)
