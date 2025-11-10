"""
Structure-Function Integration Module
====================================

Tools for integrating DoOR 2.0 olfactory response data with FlyWire connectomics.

This module bridges functional odor responses with network connectivity to reveal
circuit logic principles governing interglomerular processing.

Citation:
    Münch, D. & Galizia, C. G. DoOR 2.0 - Comprehensive Mapping of Drosophila
    melanogaster Odorant Responses. Sci. Rep. 6, 21841 (2016).

Basic Usage:
    >>> from door_toolkit.integration import DoORFlyWireIntegrator
    >>>
    >>> integrator = DoORFlyWireIntegrator(
    ...     door_cache="door_cache",
    ...     connectomics_data="data/interglomerular_crosstalk_pathways.csv"
    ... )
    >>>
    >>> # Analysis 1: Tuning correlation vs connectivity
    >>> results = integrator.analyze_tuning_vs_connectivity()
    >>>
    >>> # Analysis 2: Odor-specific subnetwork
    >>> subnetwork = integrator.extract_odor_network("acetic acid")

Modules:
    door_utils: DoOR data access and manipulation
    connectivity_utils: Connectomics data processing
    analysis: Core structure-function analyses
    visualization: Publication-ready plots
"""

__version__ = "0.1.0"

from door_toolkit.integration.door_utils import (
    load_door_response_matrix,
    load_receptor_mapping,
    calculate_lifetime_kurtosis,
    map_door_to_flywire,
    get_odorant_activated_receptors
)

from door_toolkit.integration.integrator import DoORFlyWireIntegrator
from door_toolkit.integration.odorant_mapper import OdorantMapper

__all__ = [
    "load_door_response_matrix",
    "load_receptor_mapping",
    "calculate_lifetime_kurtosis",
    "map_door_to_flywire",
    "get_odorant_activated_receptors",
    "DoORFlyWireIntegrator",
    "OdorantMapper"
]
