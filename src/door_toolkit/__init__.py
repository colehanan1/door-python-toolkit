"""
DoOR Python Toolkit
===================

Python tools for working with the DoOR (Database of Odorant Responses) database.

Extract, analyze, and integrate Drosophila odorant-receptor response data in pure Python.
No R installation required.

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

Modules:
    extractor: Extract DoOR R data to Python formats
    encoder: Encode odorant names to neural activation patterns
    utils: Helper functions and data loaders

For more information, see: https://github.com/colehanan1/door-python-toolkit
"""

__version__ = "0.1.0"
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
