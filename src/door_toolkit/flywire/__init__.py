"""
FlyWire Integration Module
===========================

Integration layer for mapping DoOR receptor data to FlyWire community labels
and neural connectivity data.

This module enables:
- Mapping DoOR receptors to FlyWire root IDs
- Parsing and searching 100K+ community labels efficiently
- Generating 3D spatial activation maps
- Exporting FlyWire-DoOR mappings

Modules:
    mapper: Core FlyWire community labels mapping
    community_labels: Community labels parser and search
    spatial_analysis: 3D spatial activation mapping

Example:
    >>> from door_toolkit.flywire import FlyWireMapper
    >>> mapper = FlyWireMapper("processed_labels.csv.gz")
    >>> or42b_cells = mapper.find_receptor_cells("Or42b")
    >>> print(f"Found {len(or42b_cells)} Or42b cells")
"""

from door_toolkit.flywire.mapper import FlyWireMapper
from door_toolkit.flywire.community_labels import CommunityLabelsParser
from door_toolkit.flywire.spatial_analysis import SpatialActivationMap

__all__ = [
    "FlyWireMapper",
    "CommunityLabelsParser",
    "SpatialActivationMap",
]
