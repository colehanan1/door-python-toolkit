"""
Neural Network Preprocessing Module
====================================

Preprocessing tools for neural network training with DoOR data.

This module provides tools for converting DoOR receptor response data
into formats suitable for neural network training, particularly for the
Plasticity-Guided Connectome Network (PGCN) project.

Features:
- Sparse encoding (KC-like representations)
- Concentration-response modeling
- Noise augmentation for robust training
- PyTorch dataset export

Modules:
    preprocessor: Main preprocessing interface
    sparse_encoding: KC-like sparse representations
    concentration_models: Concentration-response curves

Example:
    >>> from door_toolkit.neural import DoORNeuralPreprocessor
    >>> preprocessor = DoORNeuralPreprocessor("door_cache")
    >>> sparse_data = preprocessor.create_sparse_encoding(sparsity_level=0.05)
    >>> print(f"Sparsity: {(sparse_data > 0).mean():.2%}")
"""

from door_toolkit.neural.preprocessor import DoORNeuralPreprocessor
from door_toolkit.neural.sparse_encoding import SparseEncoder
from door_toolkit.neural.concentration_models import ConcentrationResponseModel

__all__ = [
    "DoORNeuralPreprocessor",
    "SparseEncoder",
    "ConcentrationResponseModel",
]
