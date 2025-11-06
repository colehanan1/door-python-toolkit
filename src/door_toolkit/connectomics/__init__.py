"""
Connectomics Module
==================

Tools for analyzing interglomerular cross-talk in the Drosophila olfactory system
using FlyWire connectome data.

This module provides:
- Biophysically realistic spiking neural network models
- Four analysis modes: Single ORN, ORN pair comparison, full network, pathway search
- Publication-ready visualizations with hierarchical neuron/glomerulus representation
- Statistical analyses (hub detection, community detection, asymmetry quantification)

Basic Usage:
    >>> from door_toolkit.connectomics import CrossTalkNetwork
    >>>
    >>> # Load network from data
    >>> network = CrossTalkNetwork.from_csv('interglomerular_crosstalk_pathways.csv')
    >>> network.set_min_synapse_threshold(5)
    >>>
    >>> # Analyze single ORN
    >>> results = network.analyze_single_orn('ORN_DL5')
    >>> results.visualize(output='figures/DL5_pathways.png')
    >>>
    >>> # Compare two ORNs
    >>> comparison = network.compare_orn_pair('ORN_DL5', 'ORN_VA1v')
    >>> comparison.plot()
    >>>
    >>> # Full network analysis
    >>> stats = network.analyze_full_network()
    >>> print(stats.hub_neurons)

Classes:
    CrossTalkNetwork: Main class for network analysis
    NetworkConfig: Configuration parameters for network construction
    SingleORNAnalysis: Results from single ORN analysis
    ORNPairComparison: Results from ORN pair comparison
    NetworkStatistics: Full network statistics and metrics

Functions:
    analyze_single_orn: Quick analysis of a single ORN
    compare_orn_pair: Quick comparison of two ORNs
    find_pathways: Find all pathways between two ORNs

For more information, see the CONNECTOMICS_README.md file.
"""

__version__ = "0.1.0"

from door_toolkit.connectomics.config import NetworkConfig
from door_toolkit.connectomics.network_builder import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import (
    analyze_single_orn,
    compare_orn_pair,
    find_pathways,
)

__all__ = [
    "CrossTalkNetwork",
    "NetworkConfig",
    "analyze_single_orn",
    "compare_orn_pair",
    "find_pathways",
]
