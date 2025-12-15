"""
Mushroom Body Pathway Tracer
=============================

Trace neural pathways from olfactory receptor neurons (ORNs) through the mushroom body
learning circuit: ORN → PN → KC → MBON

This module validates whether LASSO-identified receptors are anatomically positioned
in the appetitive/aversive learning circuits.

Classes:
    PathwayStep: Single step in a neural pathway
    MushroomBodyPathway: Complete pathway from ORN to MBON
    ConnectivityMetrics: Quantitative metrics for circuit validation
    MushroomBodyTracer: Main class for pathway tracing and analysis
"""

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PathwayStep:
    """Single step in a neural pathway."""

    source_id: str
    target_id: str
    source_type: str  # ORN, PN, KC, MBON
    target_type: str
    synapse_count: int
    neuropil: str  # Brain region (AL, MB_CA, MB_ML, etc.)
    neurotransmitter: Optional[str] = None

    def __repr__(self):
        return (
            f"{self.source_type}({self.source_id[:8]}...) "
            f"→[{self.synapse_count} syn]→ "
            f"{self.target_type}({self.target_id[:8]}...) @ {self.neuropil}"
        )


@dataclass
class MushroomBodyPathway:
    """Complete pathway from ORN through mushroom body."""

    receptor_name: str
    orn_ids: List[str]
    pn_connections: List[PathwayStep] = field(default_factory=list)
    kc_connections: List[PathwayStep] = field(default_factory=list)
    mbon_connections: List[PathwayStep] = field(default_factory=list)

    @property
    def total_orn_to_pn_synapses(self) -> int:
        """Total synapses from ORNs to PNs."""
        return sum(step.synapse_count for step in self.pn_connections)

    @property
    def total_pn_to_kc_synapses(self) -> int:
        """Total synapses from PNs to KCs."""
        return sum(step.synapse_count for step in self.kc_connections)

    @property
    def unique_pns(self) -> Set[str]:
        """Unique PN neuron IDs."""
        return {step.target_id for step in self.pn_connections}

    @property
    def unique_kcs(self) -> Set[str]:
        """Unique KC neuron IDs."""
        return {step.target_id for step in self.kc_connections}

    @property
    def kc_compartments(self) -> Dict[str, int]:
        """Count of KCs by compartment type (αβ, γ, α'β')."""
        compartments = defaultdict(int)
        for step in self.kc_connections:
            # Infer compartment from target type
            target_type = step.target_type
            if "KCab" in target_type and "KCapbp" not in target_type:
                compartments["alpha_beta"] += 1
            elif "KCg" in target_type:
                compartments["gamma"] += 1
            elif "KCapbp" in target_type or "KCab-ap" in target_type:
                compartments["alpha_prime_beta_prime"] += 1
            else:
                compartments["unknown"] += 1
        return dict(compartments)

    def pathway_summary(self) -> Dict:
        """Generate summary statistics."""
        return {
            "receptor": self.receptor_name,
            "n_orns": len(self.orn_ids),
            "n_pns": len(self.unique_pns),
            "n_kcs": len(self.unique_kcs),
            "orn_to_pn_synapses": self.total_orn_to_pn_synapses,
            "pn_to_kc_synapses": self.total_pn_to_kc_synapses,
            "kc_compartments": self.kc_compartments,
        }


@dataclass
class ConnectivityMetrics:
    """Quantitative metrics for circuit validation."""

    receptor_name: str
    orn_to_pn_strength: float  # % of ORN output going to PNs
    kc_coverage: float  # % of total KCs contacted
    alpha_beta_fraction: float  # Fraction in αβ lobe (appetitive)
    gamma_fraction: float  # Fraction in γ lobe (aversive)
    mbon_diversity: int  # Number of unique MBON types
    circuit_score: float  # Overall 0-1 score for "in learning circuit"

    def to_dict(self) -> Dict:
        return {
            "receptor": self.receptor_name,
            "orn_to_pn_strength": self.orn_to_pn_strength,
            "kc_coverage": self.kc_coverage,
            "alpha_beta_fraction": self.alpha_beta_fraction,
            "gamma_fraction": self.gamma_fraction,
            "mbon_diversity": self.mbon_diversity,
            "circuit_score": self.circuit_score,
            "circuit_type": (
                "appetitive" if self.alpha_beta_fraction > self.gamma_fraction else "aversive"
            ),
        }


class MushroomBodyTracer:
    """
    Trace pathways from ORNs through mushroom body learning circuits.

    This class loads FlyWire connectivity data and traces multi-step pathways:
    ORN → PN → KC → MBON

    Attributes:
        synapse_data: DataFrame with pre_root_id, post_root_id, syn_count, neuropil
        cell_types: DataFrame mapping root_id to primary_type
        cell_type_cache: Dict for fast cell type lookups

    Example:
        >>> tracer = MushroomBodyTracer(
        ...     synapse_path="connections_princeton.csv.gz",
        ...     cell_types_path="consolidated_cell_types.csv.gz"
        ... )
        >>> pathway = tracer.trace_receptor_pathway(
        ...     receptor_name="Or22b",
        ...     orn_ids=["720575940612345678", ...]
        ... )
        >>> print(pathway.pathway_summary())
    """

    # FlyWire cell type patterns
    ORN_PATTERNS = ["ORN", "Or", "Ir", "Gr"]
    PN_PATTERNS = ["_PN", "_adPN", "_lPN", "_vPN"]
    KC_PATTERNS = ["KC"]
    MBON_PATTERNS = ["MBON"]

    # Mushroom body neuropils (both hemispheres)
    MB_NEUROPILS = [
        "MB_CA_L",
        "MB_CA_R",
        "MB_ML_L",
        "MB_ML_R",
        "MB_VL_L",
        "MB_VL_R",
        "MB_PED_L",
        "MB_PED_R",
    ]
    AL_NEUROPILS = ["AL_L", "AL_R"]

    def __init__(
        self,
        synapse_path: str,
        cell_types_path: str,
        min_synapse_threshold: int = 1,
    ):
        """
        Initialize mushroom body tracer.

        Args:
            synapse_path: Path to connections CSV (pre_root_id, post_root_id, syn_count, neuropil)
            cell_types_path: Path to cell types CSV (root_id, primary_type)
            min_synapse_threshold: Minimum synapses to consider a connection
        """
        self.synapse_path = Path(synapse_path)
        self.cell_types_path = Path(cell_types_path)
        self.min_synapse_threshold = min_synapse_threshold

        logger.info("Initializing MushroomBodyTracer...")

        # Load cell types
        logger.info(f"Loading cell types from {self.cell_types_path}...")
        self.cell_types = pd.read_csv(self.cell_types_path)
        logger.info(f"Loaded {len(self.cell_types):,} cell type annotations")

        # Create fast lookup cache
        self.cell_type_cache = dict(
            zip(self.cell_types["root_id"].astype(str), self.cell_types["primary_type"])
        )

        # Load synapse data (lazy loading - only when needed)
        self.synapse_data = None
        self._synapse_loaded = False

        logger.info("✓ MushroomBodyTracer initialized")

    def _load_synapse_data(self):
        """Lazy load synapse connectivity data."""
        if self._synapse_loaded:
            return

        logger.info(f"Loading synapse data from {self.synapse_path}...")
        self.synapse_data = pd.read_csv(self.synapse_path)

        # Convert IDs to strings for consistency
        self.synapse_data["pre_root_id"] = self.synapse_data["pre_root_id"].astype(str)
        self.synapse_data["post_root_id"] = self.synapse_data["post_root_id"].astype(str)

        # Filter by synapse threshold
        self.synapse_data = self.synapse_data[
            self.synapse_data["syn_count"] >= self.min_synapse_threshold
        ]

        logger.info(
            f"Loaded {len(self.synapse_data):,} connections (≥{self.min_synapse_threshold} synapses)"
        )
        self._synapse_loaded = True

    def get_cell_type(self, root_id: str) -> Optional[str]:
        """
        Get cell type for a neuron ID.

        Args:
            root_id: FlyWire root ID

        Returns:
            Primary cell type, or None if not found
        """
        return self.cell_type_cache.get(str(root_id))

    def classify_neuron(self, root_id: str) -> str:
        """
        Classify neuron into broad category (ORN, PN, KC, MBON, Other).

        Args:
            root_id: FlyWire root ID

        Returns:
            Category string
        """
        cell_type = self.get_cell_type(root_id)
        if cell_type is None:
            return "Unknown"

        # Check patterns
        if any(p in cell_type for p in self.ORN_PATTERNS):
            return "ORN"
        elif any(p in cell_type for p in self.PN_PATTERNS):
            return "PN"
        elif any(p in cell_type for p in self.KC_PATTERNS):
            return "KC"
        elif any(p in cell_type for p in self.MBON_PATTERNS):
            return "MBON"
        else:
            return "Other"

    def get_downstream_connections(
        self,
        source_ids: List[str],
        target_category: Optional[str] = None,
        neuropil_filter: Optional[List[str]] = None,
    ) -> List[PathwayStep]:
        """
        Find all downstream connections from source neurons.

        Args:
            source_ids: List of presynaptic neuron IDs
            target_category: Filter targets by category (PN, KC, MBON, etc.)
            neuropil_filter: Filter by brain region (e.g., ["AL_L", "AL_R"])

        Returns:
            List of PathwayStep objects
        """
        self._load_synapse_data()

        # Filter synapses with source neurons
        connections = self.synapse_data[self.synapse_data["pre_root_id"].isin(source_ids)]

        # Apply neuropil filter if specified
        if neuropil_filter:
            connections = connections[connections["neuropil"].isin(neuropil_filter)]

        # Build pathway steps
        steps = []
        for _, row in connections.iterrows():
            source_id = row["pre_root_id"]
            target_id = row["post_root_id"]
            synapse_count = row["syn_count"]
            neuropil = row["neuropil"]
            nt_type = row.get("nt_type")

            source_type = self.get_cell_type(source_id) or "Unknown"
            target_type = self.get_cell_type(target_id) or "Unknown"

            source_category = self.classify_neuron(source_id)
            target_category_actual = self.classify_neuron(target_id)

            # Filter by target category if specified
            if target_category and target_category_actual != target_category:
                continue

            step = PathwayStep(
                source_id=source_id,
                target_id=target_id,
                source_type=source_type,
                target_type=target_type,
                synapse_count=synapse_count,
                neuropil=neuropil,
                neurotransmitter=nt_type,
            )
            steps.append(step)

        return steps

    def trace_receptor_pathway(
        self, receptor_name: str, orn_ids: List[str]
    ) -> MushroomBodyPathway:
        """
        Trace complete pathway from ORNs expressing a receptor through MB.

        Args:
            receptor_name: Receptor name (e.g., "Or22b")
            orn_ids: List of ORN neuron IDs expressing this receptor

        Returns:
            MushroomBodyPathway with traced connections

        Example:
            >>> pathway = tracer.trace_receptor_pathway("Or22b", orn_ids)
            >>> print(f"Found {len(pathway.unique_pns)} PNs")
        """
        logger.info(f"Tracing pathway for {receptor_name} ({len(orn_ids)} ORNs)...")

        pathway = MushroomBodyPathway(receptor_name=receptor_name, orn_ids=orn_ids)

        # Step 1: ORN → PN (in antennal lobe)
        logger.info("  Step 1: ORN → PN...")
        pn_steps = self.get_downstream_connections(
            source_ids=orn_ids, target_category="PN", neuropil_filter=self.AL_NEUROPILS
        )
        pathway.pn_connections = pn_steps
        logger.info(f"    Found {len(pathway.unique_pns)} unique PNs")

        if not pathway.unique_pns:
            logger.warning(f"    No PNs found for {receptor_name}!")
            return pathway

        # Step 2: PN → KC (in mushroom body calyx)
        logger.info("  Step 2: PN → KC...")
        kc_steps = self.get_downstream_connections(
            source_ids=list(pathway.unique_pns),
            target_category="KC",
            neuropil_filter=self.MB_NEUROPILS,
        )
        pathway.kc_connections = kc_steps
        logger.info(f"    Found {len(pathway.unique_kcs)} unique KCs")
        logger.info(f"    KC compartments: {pathway.kc_compartments}")

        if not pathway.unique_kcs:
            logger.warning(f"    No KCs found for {receptor_name}!")
            return pathway

        # Step 3: KC → MBON (in mushroom body lobes)
        logger.info("  Step 3: KC → MBON...")
        mbon_steps = self.get_downstream_connections(
            source_ids=list(pathway.unique_kcs),
            target_category="MBON",
            neuropil_filter=self.MB_NEUROPILS,
        )
        pathway.mbon_connections = mbon_steps
        unique_mbons = {step.target_id for step in mbon_steps}
        logger.info(f"    Found {len(unique_mbons)} unique MBONs")

        logger.info(f"✓ Pathway traced for {receptor_name}")
        return pathway

    def calculate_connectivity_metrics(
        self, pathway: MushroomBodyPathway, total_kcs_in_brain: int = 2000
    ) -> ConnectivityMetrics:
        """
        Calculate quantitative metrics for circuit validation.

        Args:
            pathway: Traced pathway object
            total_kcs_in_brain: Estimated total KCs for coverage calculation

        Returns:
            ConnectivityMetrics object

        Metrics:
            - orn_to_pn_strength: % of ORN output going to PNs (vs other targets)
            - kc_coverage: % of total KC population contacted
            - alpha_beta_fraction: Fraction in αβ lobe (appetitive learning)
            - gamma_fraction: Fraction in γ lobe (aversive learning)
            - mbon_diversity: Number of unique MBON types
            - circuit_score: Composite 0-1 score
        """
        # Get total ORN outputs to calculate PN fraction
        all_orn_outputs = self.get_downstream_connections(pathway.orn_ids)
        total_orn_synapses = sum(step.synapse_count for step in all_orn_outputs)
        pn_synapses = pathway.total_orn_to_pn_synapses

        orn_to_pn_strength = pn_synapses / total_orn_synapses if total_orn_synapses > 0 else 0.0

        # KC coverage
        kc_coverage = len(pathway.unique_kcs) / total_kcs_in_brain if total_kcs_in_brain > 0 else 0.0

        # MB lobe fractions
        compartments = pathway.kc_compartments
        total_kcs = sum(compartments.values())
        alpha_beta_fraction = compartments.get("alpha_beta", 0) / total_kcs if total_kcs > 0 else 0.0
        gamma_fraction = compartments.get("gamma", 0) / total_kcs if total_kcs > 0 else 0.0

        # MBON diversity
        mbon_diversity = len({step.target_id for step in pathway.mbon_connections})

        # Circuit score (composite)
        # Weights: orn_to_pn (40%), kc_coverage (30%), lobe_fraction (20%), mbon (10%)
        circuit_score = (
            (orn_to_pn_strength * 0.4)
            + (min(kc_coverage / 0.05, 1.0) * 0.3)
            + (max(alpha_beta_fraction, gamma_fraction) * 0.2)
            + (min(mbon_diversity / 3, 1.0) * 0.1)
        )

        metrics = ConnectivityMetrics(
            receptor_name=pathway.receptor_name,
            orn_to_pn_strength=orn_to_pn_strength,
            kc_coverage=kc_coverage,
            alpha_beta_fraction=alpha_beta_fraction,
            gamma_fraction=gamma_fraction,
            mbon_diversity=mbon_diversity,
            circuit_score=circuit_score,
        )

        logger.info(f"Connectivity metrics for {pathway.receptor_name}:")
        logger.info(f"  ORN→PN strength: {orn_to_pn_strength:.2%}")
        logger.info(f"  KC coverage: {kc_coverage:.2%}")
        logger.info(f"  α/β fraction: {alpha_beta_fraction:.2%}")
        logger.info(f"  γ fraction: {gamma_fraction:.2%}")
        logger.info(f"  Circuit score: {circuit_score:.3f}")

        return metrics

    def export_pathway_csv(self, pathways: List[MushroomBodyPathway], output_path: str):
        """Export pathway summaries to CSV."""
        rows = []
        for pathway in pathways:
            summary = pathway.pathway_summary()
            rows.append(summary)

        df = pd.DataFrame(rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported pathway summaries to {output_path}")

    def export_metrics_csv(self, metrics: List[ConnectivityMetrics], output_path: str):
        """Export connectivity metrics to CSV."""
        rows = [m.to_dict() for m in metrics]
        df = pd.DataFrame(rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported connectivity metrics to {output_path}")
