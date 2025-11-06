"""
Pathway Analysis Module
========================

Tools for analyzing olfactory pathways and generating experimental protocols
for PGCN (Plasticity-Guided Connectome Network) experiments.

This module provides:
- Quantitative pathway tracing from receptors to behaviors
- Shapley value analysis for neuron importance ranking
- PGCN experiment protocol generation
- Behavioral prediction models

Modules:
    analyzer: Core pathway analysis and tracing
    blocking_experiments: PGCN experiment protocol generators
    behavioral_prediction: Behavior prediction from receptor patterns

Example:
    >>> from door_toolkit.pathways import PathwayAnalyzer
    >>> analyzer = PathwayAnalyzer("door_cache", "flywire_data.csv.gz")
    >>> pathway = analyzer.trace_or47b_feeding_pathway()
    >>> print(f"Pathway strength: {pathway.strength:.3f}")
"""

from door_toolkit.pathways.analyzer import PathwayAnalyzer, PathwayResult
from door_toolkit.pathways.blocking_experiments import (
    BlockingExperimentGenerator,
    ExperimentProtocol,
)
from door_toolkit.pathways.behavioral_prediction import BehavioralPredictor

__all__ = [
    "PathwayAnalyzer",
    "PathwayResult",
    "BlockingExperimentGenerator",
    "ExperimentProtocol",
    "BehavioralPredictor",
]
