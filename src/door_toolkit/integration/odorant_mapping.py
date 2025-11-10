"""
Backward-compatible wrapper for `odorant_mapper`.

The OdorantMapper implementation now lives in `odorant_mapper.py`.
This module simply re-exports the class for legacy imports.
"""

from .odorant_mapper import OdorantMapper

__all__ = ["OdorantMapper"]
