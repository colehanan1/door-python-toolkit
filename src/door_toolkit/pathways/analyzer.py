"""
Pathway Analyzer
================

Core pathway analysis and tracing for olfactory circuits.

This module provides quantitative analysis of olfactory pathways from receptors
to behaviors, with special focus on feeding and learning pathways.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from door_toolkit.encoder import DoOREncoder
from door_toolkit.utils import load_response_matrix

logger = logging.getLogger(__name__)


@dataclass
class PathwayResult:
    """
    Result of pathway tracing analysis.

    Attributes:
        pathway_name: Name of the traced pathway
        source_receptors: List of receptors in this pathway
        target_behavior: Target behavior (e.g., "feeding", "avoidance")
        strength: Quantitative pathway strength (0-1)
        receptor_contributions: Mapping of receptor to contribution score
        metadata: Additional pathway metadata
    """

    pathway_name: str
    source_receptors: List[str]
    target_behavior: str
    strength: float
    receptor_contributions: Dict[str, float]
    metadata: Optional[Dict] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pathway_name": self.pathway_name,
            "source_receptors": self.source_receptors,
            "target_behavior": self.target_behavior,
            "strength": self.strength,
            "receptor_contributions": self.receptor_contributions,
            "metadata": self.metadata or {},
        }

    def get_top_receptors(self, n: int = 5) -> List[Tuple[str, float]]:
        """
        Get top N receptors by contribution.

        Args:
            n: Number of top receptors to return

        Returns:
            List of (receptor_name, contribution) tuples
        """
        sorted_receptors = sorted(
            self.receptor_contributions.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_receptors[:n]


class PathwayAnalyzer:
    """
    Analyze olfactory pathways from receptors to behaviors.

    This class enables quantitative analysis of specific olfactory pathways,
    including the Or47b→hexanol→feeding pathway and Or42b pathway.

    Attributes:
        door_cache_path: Path to DoOR cache directory
        flywire_data_path: Optional path to FlyWire community labels
        encoder: DoOREncoder instance
        response_matrix: DoOR response matrix

    Example:
        >>> analyzer = PathwayAnalyzer("door_cache", "flywire_data.csv.gz")
        >>> pathway = analyzer.trace_or47b_feeding_pathway()
        >>> print(f"Pathway strength: {pathway.strength:.3f}")
        >>> top_receptors = pathway.get_top_receptors(3)
        >>> for receptor, contrib in top_receptors:
        ...     print(f"{receptor}: {contrib:.3f}")
    """

    # Known pathway configurations from literature
    KNOWN_PATHWAYS = {
        "or47b_feeding": {
            "receptors": ["Or47b"],
            "key_odorants": ["hexanol", "1-hexanol", "hexan-1-ol"],
            "behavior": "proboscis extension (feeding)",
            "valence": "attractive",
        },
        "or42b": {
            "receptors": ["Or42b"],
            "key_odorants": ["ethyl butyrate", "ethyl acetate"],
            "behavior": "attraction",
            "valence": "attractive",
        },
        "or92a_avoidance": {
            "receptors": ["Or92a"],
            "key_odorants": ["geosmin"],
            "behavior": "avoidance",
            "valence": "aversive",
        },
    }

    def __init__(
        self,
        door_cache_path: str,
        flywire_data_path: Optional[str] = None,
    ):
        """
        Initialize pathway analyzer.

        Args:
            door_cache_path: Path to DoOR cache directory
            flywire_data_path: Optional path to FlyWire community labels

        Raises:
            FileNotFoundError: If cache directory not found
        """
        self.door_cache_path = Path(door_cache_path)
        self.flywire_data_path = (
            Path(flywire_data_path) if flywire_data_path else None
        )

        if not self.door_cache_path.exists():
            raise FileNotFoundError(f"DoOR cache not found: {self.door_cache_path}")

        # Load DoOR data
        self.encoder = DoOREncoder(str(self.door_cache_path), use_torch=False)
        self.response_matrix = load_response_matrix(str(self.door_cache_path))

        logger.info(f"Initialized PathwayAnalyzer with {len(self.response_matrix.columns)} receptors")

    def trace_or47b_feeding_pathway(self) -> PathwayResult:
        """
        Trace the Or47b → hexanol → proboscis extension pathway.

        This pathway is well-characterized: Or47b receptors detect hexanol
        and related alcohols, triggering attractive feeding behavior.

        Returns:
            PathwayResult with quantitative pathway analysis

        Example:
            >>> analyzer = PathwayAnalyzer("door_cache")
            >>> pathway = analyzer.trace_or47b_feeding_pathway()
            >>> print(f"Or47b activation by hexanol: {pathway.strength:.3f}")
        """
        logger.info("Tracing Or47b → hexanol → feeding pathway")

        pathway_config = self.KNOWN_PATHWAYS["or47b_feeding"]
        key_odorants = pathway_config["key_odorants"]

        # Calculate pathway strength based on Or47b responses to key odorants
        or47b_responses = []
        odorant_responses = {}

        # Filter to only odorants that exist in the database
        available_odorants = [o for o in key_odorants if o in self.encoder.odorant_names]
        if not available_odorants:
            logger.warning(f"None of the key odorants {key_odorants} found in DoOR database")
            logger.info(f"Available similar odorants: {[o for o in self.encoder.odorant_names if 'hexanol' in o.lower()][:5]}")

        for odorant in available_odorants:
            try:
                response_vector = self.encoder.encode(odorant)

                # Find Or47b receptor index
                or47b_idx = None
                for i, receptor in enumerate(self.encoder.receptor_names):
                    if receptor.lower() == "or47b":
                        or47b_idx = i
                        break

                if or47b_idx is not None:
                    response = float(response_vector[or47b_idx])
                    if not np.isnan(response):
                        or47b_responses.append(response)
                        odorant_responses[odorant] = response
                        logger.debug(f"Or47b response to {odorant}: {response:.3f}")

            except Exception as e:
                logger.warning(f"Could not encode {odorant}: {e}")
                continue

        # Calculate overall pathway strength
        if or47b_responses:
            pathway_strength = float(np.mean(or47b_responses))
        else:
            pathway_strength = 0.0
            logger.warning("No Or47b responses found for key odorants")

        # Calculate receptor contributions
        receptor_contributions = self._calculate_receptor_contributions(
            key_odorants, ["Or47b"]
        )

        pathway = PathwayResult(
            pathway_name="Or47b → Hexanol → Feeding",
            source_receptors=["Or47b"],
            target_behavior="proboscis extension (feeding)",
            strength=pathway_strength,
            receptor_contributions=receptor_contributions,
            metadata={
                "key_odorants": key_odorants,
                "odorant_responses": odorant_responses,
                "valence": "attractive",
                "literature_reference": "Root et al. (2011)",
            },
        )

        logger.info(f"Or47b feeding pathway strength: {pathway_strength:.3f}")
        return pathway

    def trace_or42b_pathway(self) -> PathwayResult:
        """
        Trace the Or42b pathway for ethyl butyrate and related esters.

        Returns:
            PathwayResult with quantitative pathway analysis

        Example:
            >>> analyzer = PathwayAnalyzer("door_cache")
            >>> pathway = analyzer.trace_or42b_pathway()
            >>> print(f"Or42b pathway strength: {pathway.strength:.3f}")
        """
        logger.info("Tracing Or42b pathway")

        pathway_config = self.KNOWN_PATHWAYS["or42b"]
        key_odorants = pathway_config["key_odorants"]

        or42b_responses = []
        odorant_responses = {}

        for odorant in key_odorants:
            try:
                response_vector = self.encoder.encode(odorant)

                # Find Or42b receptor index
                or42b_idx = None
                for i, receptor in enumerate(self.encoder.receptor_names):
                    if receptor.lower() == "or42b":
                        or42b_idx = i
                        break

                if or42b_idx is not None:
                    response = float(response_vector[or42b_idx])
                    if not np.isnan(response):
                        or42b_responses.append(response)
                        odorant_responses[odorant] = response
                        logger.debug(f"Or42b response to {odorant}: {response:.3f}")

            except Exception as e:
                logger.warning(f"Could not encode {odorant}: {e}")
                continue

        pathway_strength = float(np.mean(or42b_responses)) if or42b_responses else 0.0

        receptor_contributions = self._calculate_receptor_contributions(
            key_odorants, ["Or42b"]
        )

        pathway = PathwayResult(
            pathway_name="Or42b → Fruit Esters → Attraction",
            source_receptors=["Or42b"],
            target_behavior="attraction",
            strength=pathway_strength,
            receptor_contributions=receptor_contributions,
            metadata={
                "key_odorants": key_odorants,
                "odorant_responses": odorant_responses,
                "valence": "attractive",
            },
        )

        logger.info(f"Or42b pathway strength: {pathway_strength:.3f}")
        return pathway

    def trace_custom_pathway(
        self,
        receptors: List[str],
        odorants: List[str],
        behavior: str,
    ) -> PathwayResult:
        """
        Trace a custom pathway for specified receptors and odorants.

        Args:
            receptors: List of receptor names to analyze
            odorants: List of key odorants for this pathway
            behavior: Target behavior description

        Returns:
            PathwayResult with pathway analysis

        Example:
            >>> analyzer = PathwayAnalyzer("door_cache")
            >>> pathway = analyzer.trace_custom_pathway(
            ...     receptors=["Or92a"],
            ...     odorants=["geosmin"],
            ...     behavior="avoidance"
            ... )
        """
        logger.info(f"Tracing custom pathway: {receptors} → {behavior}")

        all_responses = []
        odorant_responses = {}

        for odorant in odorants:
            try:
                response_vector = self.encoder.encode(odorant)

                for receptor in receptors:
                    receptor_idx = None
                    for i, rec in enumerate(self.encoder.receptor_names):
                        if rec.lower() == receptor.lower():
                            receptor_idx = i
                            break

                    if receptor_idx is not None:
                        response = float(response_vector[receptor_idx])
                        if not np.isnan(response):
                            all_responses.append(response)
                            key = f"{receptor}:{odorant}"
                            odorant_responses[key] = response

            except Exception as e:
                logger.warning(f"Could not encode {odorant}: {e}")
                continue

        pathway_strength = float(np.mean(all_responses)) if all_responses else 0.0

        receptor_contributions = self._calculate_receptor_contributions(
            odorants, receptors
        )

        pathway = PathwayResult(
            pathway_name=f"Custom: {', '.join(receptors)} → {behavior}",
            source_receptors=receptors,
            target_behavior=behavior,
            strength=pathway_strength,
            receptor_contributions=receptor_contributions,
            metadata={
                "key_odorants": odorants,
                "odorant_responses": odorant_responses,
            },
        )

        return pathway

    def compute_shapley_importance(
        self, target_behavior: str, odorants: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Compute Shapley values for receptor importance.

        Shapley values quantify each receptor's contribution to a behavior
        by considering all possible receptor combinations.

        Args:
            target_behavior: Behavior to analyze
            odorants: Optional list of odorants to consider

        Returns:
            Dictionary mapping receptor names to Shapley values

        Example:
            >>> analyzer = PathwayAnalyzer("door_cache")
            >>> importance = analyzer.compute_shapley_importance("feeding")
            >>> for receptor, value in sorted(importance.items(), key=lambda x: -x[1])[:5]:
            ...     print(f"{receptor}: {value:.4f}")
        """
        logger.info(f"Computing Shapley values for behavior: {target_behavior}")

        # For simplicity, use variance-based importance as approximation
        # Full Shapley computation would require behavior labels
        if odorants is None:
            odorants = self.encoder.odorant_names[:100]  # Sample for efficiency

        receptor_activations = defaultdict(list)

        for odorant in tqdm(odorants, desc="Computing importance"):
            try:
                response_vector = self.encoder.encode(odorant)
                for i, receptor in enumerate(self.encoder.receptor_names):
                    response = float(response_vector[i])
                    if not np.isnan(response):
                        receptor_activations[receptor].append(response)
            except Exception:
                continue

        # Calculate variance as proxy for importance
        importance_scores = {}
        for receptor, activations in receptor_activations.items():
            if len(activations) > 0:
                # Use variance and mean as combined importance metric
                variance = np.var(activations)
                mean_abs = np.mean(np.abs(activations))
                importance_scores[receptor] = float(variance * mean_abs)
            else:
                importance_scores[receptor] = 0.0

        # Normalize to sum to 1
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {
                k: v / total for k, v in importance_scores.items()
            }

        logger.info(f"Computed importance scores for {len(importance_scores)} receptors")
        return importance_scores

    def find_critical_blocking_targets(
        self, pathway: Optional[PathwayResult] = None, threshold: float = 0.1
    ) -> List[str]:
        """
        Identify critical receptors for blocking experiments.

        Args:
            pathway: Optional pathway to analyze (uses Or47b by default)
            threshold: Minimum contribution threshold

        Returns:
            List of receptor names suitable for blocking experiments

        Example:
            >>> analyzer = PathwayAnalyzer("door_cache")
            >>> pathway = analyzer.trace_or47b_feeding_pathway()
            >>> targets = analyzer.find_critical_blocking_targets(pathway)
            >>> print(f"Blocking targets: {targets}")
        """
        if pathway is None:
            pathway = self.trace_or47b_feeding_pathway()

        critical_receptors = [
            receptor
            for receptor, contribution in pathway.receptor_contributions.items()
            if contribution >= threshold
        ]

        logger.info(
            f"Found {len(critical_receptors)} critical blocking targets "
            f"(threshold={threshold})"
        )

        return critical_receptors

    def _calculate_receptor_contributions(
        self, odorants: List[str], receptors: List[str]
    ) -> Dict[str, float]:
        """
        Calculate receptor contributions to pathway.

        Args:
            odorants: List of key odorants
            receptors: List of receptors to analyze

        Returns:
            Dictionary mapping receptor to contribution score
        """
        contributions = defaultdict(list)

        for odorant in odorants:
            try:
                response_vector = self.encoder.encode(odorant)

                for receptor in receptors:
                    receptor_idx = None
                    for i, rec in enumerate(self.encoder.receptor_names):
                        if rec.lower() == receptor.lower():
                            receptor_idx = i
                            break

                    if receptor_idx is not None:
                        response = float(response_vector[receptor_idx])
                        if not np.isnan(response):
                            contributions[receptor].append(abs(response))

            except Exception:
                continue

        # Average contributions
        result = {}
        for receptor, values in contributions.items():
            result[receptor] = float(np.mean(values)) if values else 0.0

        # Normalize
        total = sum(result.values())
        if total > 0:
            result = {k: v / total for k, v in result.items()}

        return result

    def compare_pathways(
        self, pathways: List[PathwayResult]
    ) -> pd.DataFrame:
        """
        Compare multiple pathways quantitatively.

        Args:
            pathways: List of PathwayResult objects

        Returns:
            DataFrame with pathway comparison

        Example:
            >>> analyzer = PathwayAnalyzer("door_cache")
            >>> p1 = analyzer.trace_or47b_feeding_pathway()
            >>> p2 = analyzer.trace_or42b_pathway()
            >>> comparison = analyzer.compare_pathways([p1, p2])
            >>> print(comparison)
        """
        rows = []
        for pathway in pathways:
            row = {
                "pathway_name": pathway.pathway_name,
                "target_behavior": pathway.target_behavior,
                "strength": pathway.strength,
                "n_receptors": len(pathway.source_receptors),
                "primary_receptor": pathway.source_receptors[0] if pathway.source_receptors else None,
            }
            rows.append(row)

        return pd.DataFrame(rows)
