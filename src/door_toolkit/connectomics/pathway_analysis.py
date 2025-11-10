"""
Pathway Analysis
================

Four analysis modes for investigating interglomerular cross-talk pathways:
1. Single ORN Focus - Analyze all pathways from one ORN/glomerulus
2. ORN Pair Comparison - Compare connectivity between two ORNs/glomeruli
3. Full Network View - Analyze complete network structure
4. Pathway Search - Find specific pathways between neurons

Each mode returns a results object with analysis methods and visualization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Set
from dataclasses import dataclass
from pathlib import Path
import logging

from door_toolkit.connectomics.network_builder import CrossTalkNetwork

logger = logging.getLogger(__name__)


@dataclass
class SingleORNAnalysis:
    """
    Results from single ORN/glomerulus analysis.

    Attributes:
        orn_identifier: ORN root_id or glomerulus name
        is_glomerulus: Whether analyzing full glomerulus or single neuron
        pathways: List of all pathways from this ORN
        intermediate_neurons: Neurons in Layer 1 (LNs, PNs)
        target_neurons: Target neurons (Layer 2)
        pathway_counts: Counts by pathway type
        statistics: Summary statistics
    """

    orn_identifier: str
    is_glomerulus: bool
    pathways: List[Dict]
    intermediate_neurons: Dict[str, List[str]]
    target_neurons: Dict[str, List[str]]
    pathway_counts: Dict[str, int]
    statistics: Dict

    @property
    def num_pathways(self) -> int:
        """Total number of pathways."""
        return len(self.pathways)

    @property
    def num_intermediate_neurons(self) -> int:
        """Total number of intermediate neurons."""
        return sum(len(v) for v in self.intermediate_neurons.values())

    @property
    def num_target_neurons(self) -> int:
        """Total number of target neurons."""
        return sum(len(v) for v in self.target_neurons.values())

    def to_dataframe(self) -> pd.DataFrame:
        """Convert pathways to DataFrame for analysis/export."""
        return pd.DataFrame(self.pathways)

    def get_targets_by_glomerulus(self) -> Dict[str, int]:
        """Count target neurons by glomerulus."""
        df = self.to_dataframe()

        # Handle empty dataframe
        if df.empty or "level2_category" not in df.columns:
            return {}

        target_orns = df[df["level2_category"] == "ORN"]

        if len(target_orns) == 0:
            return {}

        return target_orns["level2_glomerulus"].value_counts().to_dict()

    def get_strongest_pathways(self, n: int = 10) -> pd.DataFrame:
        """Get the N strongest pathways by synapse count."""
        df = self.to_dataframe()

        # Handle empty dataframe or missing column
        if df.empty or "synapse_count_step2" not in df.columns:
            return pd.DataFrame()

        return df.nlargest(n, "synapse_count_step2")

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Single ORN Analysis: {self.orn_identifier}",
            "=" * 60,
            f"Analysis level: {'Glomerulus' if self.is_glomerulus else 'Single neuron'}",
            f"Total pathways: {self.num_pathways:,}",
            "",
            "Intermediate neurons:",
        ]

        for category, neurons in self.intermediate_neurons.items():
            lines.append(f"  {category}: {len(neurons)}")

        lines.extend(
            [
                "",
                "Target neurons:",
            ]
        )

        for category, neurons in self.target_neurons.items():
            lines.append(f"  {category}: {len(neurons)}")

        lines.extend(
            [
                "",
                "Pathway type distribution:",
            ]
        )

        for ptype, count in self.pathway_counts.items():
            lines.append(f"  {ptype}: {count}")

        # Top target glomeruli
        target_gloms = self.get_targets_by_glomerulus()
        if target_gloms:
            lines.extend(
                [
                    "",
                    "Top 5 target glomeruli:",
                ]
            )
            for glom, count in list(target_gloms.items())[:5]:
                lines.append(f"  {glom}: {count} pathways")

        return "\n".join(lines)


@dataclass
class ORNPairComparison:
    """
    Results from comparing two ORNs/glomeruli.

    Attributes:
        orn1: First ORN identifier
        orn2: Second ORN identifier
        is_glomerulus: Whether comparing glomeruli or single neurons
        pathways_1_to_2: Pathways from ORN1 to ORN2
        pathways_2_to_1: Pathways from ORN2 to ORN1
        shared_intermediates: Intermediate neurons shared by both
        cross_talk_strength: Quantified cross-talk metrics
        asymmetry_metrics: Asymmetry analysis
    """

    orn1: str
    orn2: str
    is_glomerulus: bool
    pathways_1_to_2: List[Dict]
    pathways_2_to_1: List[Dict]
    shared_intermediates: Dict[str, List[str]]
    cross_talk_strength: Dict
    asymmetry_metrics: Dict

    @property
    def has_bidirectional_crosstalk(self) -> bool:
        """Check if cross-talk exists in both directions."""
        return len(self.pathways_1_to_2) > 0 and len(self.pathways_2_to_1) > 0

    @property
    def total_shared_intermediates(self) -> int:
        """Total number of shared intermediate neurons."""
        return sum(len(v) for v in self.shared_intermediates.values())

    def get_asymmetry_ratio(self) -> float:
        """
        Calculate asymmetry ratio.

        Returns 0 for symmetric, >0 for asymmetric (1→2 stronger), <0 for (2→1 stronger)
        """
        strength_1_to_2 = self.cross_talk_strength["1_to_2"]
        strength_2_to_1 = self.cross_talk_strength["2_to_1"]

        if strength_1_to_2 + strength_2_to_1 == 0:
            return 0.0

        return (strength_1_to_2 - strength_2_to_1) / (strength_1_to_2 + strength_2_to_1)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"ORN Pair Comparison: {self.orn1} vs {self.orn2}",
            "=" * 60,
            f"Analysis level: {'Glomerulus' if self.is_glomerulus else 'Single neuron'}",
            "",
            "Cross-talk pathways:",
            f"  {self.orn1} → {self.orn2}: {len(self.pathways_1_to_2)} pathways",
            f"  {self.orn2} → {self.orn1}: {len(self.pathways_2_to_1)} pathways",
            f"  Bidirectional: {'Yes' if self.has_bidirectional_crosstalk else 'No'}",
            "",
            "Cross-talk strength (total synapses):",
            f"  {self.orn1} → {self.orn2}: {self.cross_talk_strength['1_to_2']}",
            f"  {self.orn2} → {self.orn1}: {self.cross_talk_strength['2_to_1']}",
            "",
            "Shared intermediate neurons:",
        ]

        for category, neurons in self.shared_intermediates.items():
            lines.append(f"  {category}: {len(neurons)}")

        lines.extend(
            [
                "",
                f"Asymmetry ratio: {self.get_asymmetry_ratio():.3f}",
                f"  (>0: {self.orn1} → {self.orn2} stronger)",
                f"  (<0: {self.orn2} → {self.orn1} stronger)",
            ]
        )

        return "\n".join(lines)


# Mode 1: Single ORN Focus
def analyze_single_orn(
    network: CrossTalkNetwork, orn_identifier: Union[str, int], by_glomerulus: bool = True
) -> SingleORNAnalysis:
    """
    Analyze all pathways from a single ORN or glomerulus.

    Args:
        network: CrossTalkNetwork instance
        orn_identifier: Either ORN root_id or glomerulus name
        by_glomerulus: If True, analyze entire glomerulus

    Returns:
        SingleORNAnalysis results object

    Example:
        >>> network = CrossTalkNetwork.from_csv('pathways.csv')
        >>> results = analyze_single_orn(network, 'ORN_DL5', by_glomerulus=True)
        >>> print(results.summary())
        >>> df = results.to_dataframe()
    """
    logger.info(f"Analyzing {'glomerulus' if by_glomerulus else 'ORN'}: {orn_identifier}")

    # Get all pathways
    pathways = network.get_pathways_from_orn(orn_identifier, by_glomerulus)

    if not pathways:
        logger.warning(f"No pathways found for {orn_identifier}")
        return SingleORNAnalysis(
            orn_identifier=str(orn_identifier),
            is_glomerulus=by_glomerulus,
            pathways=[],
            intermediate_neurons={},
            target_neurons={},
            pathway_counts={},
            statistics={},
        )

    # Organize intermediate neurons by category
    intermediate_neurons = {"LNs": [], "PNs": []}
    target_neurons = {"ORNs": [], "PNs": [], "LNs": []}

    for pathway in pathways:
        # Track intermediate neurons
        if pathway["level1_category"] == "Local_Neuron":
            if pathway["level1_id"] not in intermediate_neurons["LNs"]:
                intermediate_neurons["LNs"].append(pathway["level1_id"])
        elif pathway["level1_category"] == "Projection_Neuron":
            if pathway["level1_id"] not in intermediate_neurons["PNs"]:
                intermediate_neurons["PNs"].append(pathway["level1_id"])

        # Track target neurons
        if pathway["level2_category"] == "ORN":
            if pathway["level2_id"] not in target_neurons["ORNs"]:
                target_neurons["ORNs"].append(pathway["level2_id"])
        elif pathway["level2_category"] == "Projection_Neuron":
            if pathway["level2_id"] not in target_neurons["PNs"]:
                target_neurons["PNs"].append(pathway["level2_id"])
        elif pathway["level2_category"] == "Local_Neuron":
            if pathway["level2_id"] not in target_neurons["LNs"]:
                target_neurons["LNs"].append(pathway["level2_id"])

    # Count pathway types
    pathway_counts = {}
    for pathway in pathways:
        l1_cat = pathway["level1_category"]
        l2_cat = pathway["level2_category"]

        if l1_cat == "Local_Neuron" and l2_cat == "ORN":
            ptype = "ORN→LN→ORN (lateral inhibition)"
        elif l1_cat == "Local_Neuron" and l2_cat == "Projection_Neuron":
            ptype = "ORN→LN→PN (feedforward inhibition)"
        elif l1_cat == "Projection_Neuron":
            ptype = "ORN→PN→feedback"
        else:
            ptype = "Other"

        pathway_counts[ptype] = pathway_counts.get(ptype, 0) + 1

    # Calculate statistics
    synapse_counts_step2 = [p["synapse_count_step2"] for p in pathways]
    statistics = {
        "mean_synapse_count": np.mean(synapse_counts_step2),
        "median_synapse_count": np.median(synapse_counts_step2),
        "total_synapse_count": np.sum(synapse_counts_step2),
        "max_synapse_count": np.max(synapse_counts_step2),
    }

    return SingleORNAnalysis(
        orn_identifier=str(orn_identifier),
        is_glomerulus=by_glomerulus,
        pathways=pathways,
        intermediate_neurons=intermediate_neurons,
        target_neurons=target_neurons,
        pathway_counts=pathway_counts,
        statistics=statistics,
    )


# Mode 2: ORN Pair Comparison
def compare_orn_pair(
    network: CrossTalkNetwork,
    orn1: Union[str, int],
    orn2: Union[str, int],
    by_glomerulus: bool = True,
) -> ORNPairComparison:
    """
    Compare cross-talk between two ORNs or glomeruli.

    Args:
        network: CrossTalkNetwork instance
        orn1: First ORN root_id or glomerulus name
        orn2: Second ORN root_id or glomerulus name
        by_glomerulus: If True, compare glomeruli

    Returns:
        ORNPairComparison results object

    Example:
        >>> network = CrossTalkNetwork.from_csv('pathways.csv')
        >>> comparison = compare_orn_pair(network, 'ORN_DL5', 'ORN_VA1v')
        >>> print(comparison.summary())
        >>> print(f"Asymmetry: {comparison.get_asymmetry_ratio():.3f}")
    """
    logger.info(f"Comparing {orn1} vs {orn2}")

    # Get pathways in both directions
    pathways_1_to_2 = network.get_pathways_between_orns(orn1, orn2, by_glomerulus)
    pathways_2_to_1 = network.get_pathways_between_orns(orn2, orn1, by_glomerulus)

    # Find shared intermediate neurons
    intermediates_1 = set()
    for p in pathways_1_to_2:
        intermediates_1.add((p["level1_id"], p["level1_category"]))

    intermediates_2 = set()
    for p in pathways_2_to_1:
        intermediates_2.add((p["level1_id"], p["level1_category"]))

    shared = intermediates_1 & intermediates_2

    shared_intermediates = {"LNs": [], "PNs": []}
    for neuron_id, category in shared:
        if category == "Local_Neuron":
            shared_intermediates["LNs"].append(neuron_id)
        elif category == "Projection_Neuron":
            shared_intermediates["PNs"].append(neuron_id)

    # Calculate cross-talk strength (total synapses)
    strength_1_to_2 = sum(p["synapse_count_step2"] for p in pathways_1_to_2)
    strength_2_to_1 = sum(p["synapse_count_step2"] for p in pathways_2_to_1)

    cross_talk_strength = {
        "1_to_2": strength_1_to_2,
        "2_to_1": strength_2_to_1,
        "total": strength_1_to_2 + strength_2_to_1,
        "mean_1_to_2": strength_1_to_2 / len(pathways_1_to_2) if pathways_1_to_2 else 0,
        "mean_2_to_1": strength_2_to_1 / len(pathways_2_to_1) if pathways_2_to_1 else 0,
    }

    # Calculate asymmetry metrics
    asymmetry_metrics = {
        "pathway_count_ratio": len(pathways_1_to_2) / max(len(pathways_2_to_1), 1),
        "strength_ratio": strength_1_to_2 / max(strength_2_to_1, 1),
        "shared_intermediate_fraction": len(shared)
        / max(len(intermediates_1 | intermediates_2), 1),
    }

    return ORNPairComparison(
        orn1=str(orn1),
        orn2=str(orn2),
        is_glomerulus=by_glomerulus,
        pathways_1_to_2=pathways_1_to_2,
        pathways_2_to_1=pathways_2_to_1,
        shared_intermediates=shared_intermediates,
        cross_talk_strength=cross_talk_strength,
        asymmetry_metrics=asymmetry_metrics,
    )


# Mode 3: Full Network View (implemented in statistics.py)
# Mode 4: Pathway Search
def find_pathways(
    network: CrossTalkNetwork,
    source: Union[str, int],
    target: Union[str, int],
    by_glomerulus: bool = False,
    max_pathways: Optional[int] = None,
) -> Dict:
    """
    Find all pathways between source and target neurons/glomeruli.

    Args:
        network: CrossTalkNetwork instance
        source: Source ORN root_id or glomerulus name
        target: Target ORN root_id or glomerulus name
        by_glomerulus: If True, search at glomerulus level
        max_pathways: Maximum number of pathways to return (None = all)

    Returns:
        Dictionary with pathway analysis results

    Example:
        >>> network = CrossTalkNetwork.from_csv('pathways.csv')
        >>> results = find_pathways(network, 'ORN_DL5', 'ORN_VA1v', by_glomerulus=True)
        >>> print(f"Found {results['num_pathways']} pathways")
        >>> for p in results['pathways'][:5]:
        ...     print(f"{p['orn_glomerulus']} → {p['level1_type']} → {p['level2_glomerulus']}")
    """
    logger.info(f"Finding pathways from {source} to {target}")

    pathways = network.get_pathways_between_orns(source, target, by_glomerulus)

    if max_pathways is not None:
        # Sort by synapse strength and take top N
        pathways = sorted(pathways, key=lambda p: p["synapse_count_step2"], reverse=True)[
            :max_pathways
        ]

    # Analyze pathways
    if not pathways:
        return {
            "source": str(source),
            "target": str(target),
            "num_pathways": 0,
            "pathways": [],
            "intermediate_neurons": {"LNs": [], "PNs": []},
            "statistics": {
                "total_pathways": 0,
                "total_synapses": 0,
                "mean_synapses_per_pathway": 0,
                "median_synapses_per_pathway": 0,
                "max_synapses": 0,
                "min_synapses": 0,
            },
        }

    # Extract intermediate neurons
    intermediate_neurons = {"LNs": set(), "PNs": set()}
    for pathway in pathways:
        if pathway["level1_category"] == "Local_Neuron":
            intermediate_neurons["LNs"].add(pathway["level1_id"])
        elif pathway["level1_category"] == "Projection_Neuron":
            intermediate_neurons["PNs"].add(pathway["level1_id"])

    # Convert sets to lists
    intermediate_neurons = {k: list(v) for k, v in intermediate_neurons.items()}

    # Calculate statistics
    synapse_counts = [p["synapse_count_step2"] for p in pathways]
    statistics = {
        "total_pathways": len(pathways),
        "total_synapses": sum(synapse_counts),
        "mean_synapses_per_pathway": np.mean(synapse_counts),
        "median_synapses_per_pathway": np.median(synapse_counts),
        "max_synapses": max(synapse_counts),
        "min_synapses": min(synapse_counts),
    }

    # Find shortest path length
    if by_glomerulus:
        # Use first neurons from each glomerulus for shortest path
        source_neurons = network.get_glomerulus_neurons(source)
        target_neurons = network.get_glomerulus_neurons(target)
        if source_neurons and target_neurons:
            shortest_paths = network.find_shortest_paths(
                source_neurons[0], target_neurons[0], max_paths=1
            )
            if shortest_paths:
                statistics["shortest_path_length"] = len(shortest_paths[0]) - 1
    else:
        shortest_paths = network.find_shortest_paths(source, target, max_paths=1)
        if shortest_paths:
            statistics["shortest_path_length"] = len(shortest_paths[0]) - 1

    return {
        "source": str(source),
        "target": str(target),
        "by_glomerulus": by_glomerulus,
        "num_pathways": len(pathways),
        "pathways": pathways,
        "intermediate_neurons": intermediate_neurons,
        "statistics": statistics,
    }


def export_pathways_to_csv(pathways: List[Dict], filepath: Union[str, Path]) -> None:
    """
    Export pathway list to CSV file.

    Args:
        pathways: List of pathway dictionaries
        filepath: Output CSV file path
    """
    df = pd.DataFrame(pathways)
    df.to_csv(filepath, index=False)
    logger.info(f"Exported {len(pathways)} pathways to {filepath}")
