"""
DoOR Python Toolkit
===================

Python tools for working with the DoOR (Database of Odorant Responses) database
and FlyWire connectomics data.

Extract, analyze, and integrate Drosophila odorant-receptor response data and
neural connectivity in pure Python. No R installation required.

Basic Usage:
    >>> from door_toolkit import DoORExtractor, DoOREncoder

    >>> # Extract R data to Python formats
    >>> extractor = DoORExtractor(
    ...     input_dir="DoOR.data/data",
    ...     output_dir="door_cache"
    ... )
    >>> extractor.run()

    >>> # Use in machine learning
    >>> encoder = DoOREncoder("door_cache")
    >>> pn_activation = encoder.encode("acetic acid")

Advanced Features:
    >>> # FlyWire integration
    >>> from door_toolkit.flywire import FlyWireMapper
    >>> mapper = FlyWireMapper("community_labels.csv.gz")
    >>> mapper.parse_labels()

    >>> # Pathway analysis
    >>> from door_toolkit.pathways import PathwayAnalyzer
    >>> analyzer = PathwayAnalyzer("door_cache")
    >>> pathway = analyzer.trace_or47b_feeding_pathway()

    >>> # Neural network preprocessing
    >>> from door_toolkit.neural import DoORNeuralPreprocessor
    >>> preprocessor = DoORNeuralPreprocessor("door_cache")
    >>> sparse_data = preprocessor.create_sparse_encoding(sparsity_level=0.05)

    >>> # Connectomics (NEW in v0.3.0)
    >>> from door_toolkit.connectomics import CrossTalkNetwork
    >>> network = CrossTalkNetwork.from_csv("interglomerular_crosstalk_pathways.csv")
    >>> network.set_min_synapse_threshold(10)
    >>> print(network.summary())

Modules:
    extractor: Extract DoOR R data to Python formats
    encoder: Encode odorant names to neural activation patterns
    utils: Helper functions and data loaders
    flywire: FlyWire community labels integration
    pathways: Olfactory pathway analysis and experiment generation
    neural: Neural network preprocessing and sparse encoding
    connectomics: Interglomerular cross-talk network analysis (NEW)

For more information, see: https://github.com/colehanan1/door-python-toolkit
"""

__version__ = "0.3.0"
__author__ = "Cole Hanan"
__license__ = "MIT"

from door_toolkit.extractor import DoORExtractor
from door_toolkit.encoder import DoOREncoder
from door_toolkit.utils import list_odorants, load_response_matrix

__all__ = [
    "DoORExtractor",
    "DoOREncoder",
    "list_odorants",
    "load_response_matrix",
]
