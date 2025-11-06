"""
Blocking Experiments Module
============================

Generate experimental protocols for PGCN blocking experiments.

This module creates structured experimental protocols for testing causal
relationships in neural circuits using optogenetic silencing, microsurgery,
and other intervention techniques.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from door_toolkit.pathways.analyzer import PathwayAnalyzer, PathwayResult

logger = logging.getLogger(__name__)


@dataclass
class ExperimentStep:
    """Single step in an experiment protocol."""

    step_number: int
    action: str
    target: str
    method: str
    parameters: Dict
    expected_outcome: str
    measurement: str

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "step_number": self.step_number,
            "action": self.action,
            "target": self.target,
            "method": self.method,
            "parameters": self.parameters,
            "expected_outcome": self.expected_outcome,
            "measurement": self.measurement,
        }


@dataclass
class ExperimentProtocol:
    """
    Complete experimental protocol for PGCN blocking experiments.

    Attributes:
        experiment_id: Unique experiment identifier
        experiment_name: Descriptive name
        hypothesis: Testable hypothesis
        steps: List of experimental steps
        controls: Control conditions
        metadata: Additional metadata
    """

    experiment_id: str
    experiment_name: str
    hypothesis: str
    steps: List[ExperimentStep]
    controls: List[str]
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "hypothesis": self.hypothesis,
            "steps": [step.to_dict() for step in self.steps],
            "controls": self.controls,
            "metadata": self.metadata,
            "generated_at": datetime.now().isoformat(),
        }

    def export_json(self, output_path: str) -> None:
        """
        Export protocol to JSON file.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

        logger.info(f"Exported experiment protocol to {output_path}")

    def export_markdown(self, output_path: str) -> None:
        """
        Export protocol to Markdown file for human readability.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            f"# Experiment Protocol: {self.experiment_name}",
            f"\n**Experiment ID:** {self.experiment_id}\n",
            f"**Hypothesis:** {self.hypothesis}\n",
            "## Experimental Steps\n",
        ]

        for step in self.steps:
            lines.append(f"### Step {step.step_number}: {step.action}")
            lines.append(f"- **Target:** {step.target}")
            lines.append(f"- **Method:** {step.method}")
            lines.append(f"- **Parameters:** {', '.join(f'{k}={v}' for k, v in step.parameters.items())}")
            lines.append(f"- **Expected Outcome:** {step.expected_outcome}")
            lines.append(f"- **Measurement:** {step.measurement}\n")

        lines.append("## Controls\n")
        for i, control in enumerate(self.controls, 1):
            lines.append(f"{i}. {control}")

        if self.metadata:
            lines.append("\n## Metadata\n")
            for key, value in self.metadata.items():
                lines.append(f"- **{key}:** {value}")

        content = "\n".join(lines)
        output_path.write_text(content)

        logger.info(f"Exported experiment protocol to {output_path}")


class BlockingExperimentGenerator:
    """
    Generate experimental protocols for PGCN blocking experiments.

    This class creates structured protocols for different types of neural
    circuit interventions based on DoOR and FlyWire data.

    Example:
        >>> generator = BlockingExperimentGenerator("door_cache", "flywire_data.csv.gz")
        >>> protocol = generator.generate_experiment_1_protocol()
        >>> protocol.export_json("experiments/exp1_protocol.json")
    """

    def __init__(
        self,
        door_cache_path: str,
        flywire_data_path: Optional[str] = None,
    ):
        """
        Initialize experiment generator.

        Args:
            door_cache_path: Path to DoOR cache directory
            flywire_data_path: Optional path to FlyWire community labels
        """
        self.analyzer = PathwayAnalyzer(door_cache_path, flywire_data_path)
        logger.info("Initialized BlockingExperimentGenerator")

    def generate_experiment_1_protocol(self) -> ExperimentProtocol:
        """
        Generate Experiment 1: Single-unit veto blocking.

        Tests whether silencing individual Or47b neurons can block feeding behavior.

        Returns:
            ExperimentProtocol for single-unit veto experiment

        Example:
            >>> generator = BlockingExperimentGenerator("door_cache")
            >>> protocol = generator.generate_experiment_1_protocol()
            >>> print(protocol.hypothesis)
        """
        logger.info("Generating Experiment 1 (Single-unit veto) protocol")

        # Analyze Or47b pathway
        pathway = self.analyzer.trace_or47b_feeding_pathway()

        steps = [
            ExperimentStep(
                step_number=1,
                action="Identify Or47b neurons",
                target="Or47b ORNs",
                method="FlyWire community labels + immunohistochemistry",
                parameters={"receptor": "Or47b", "sensillum_type": "ab1A"},
                expected_outcome="~20 Or47b neurons identified bilaterally",
                measurement="Cell count and root IDs",
            ),
            ExperimentStep(
                step_number=2,
                action="Express optogenetic silencer",
                target="Individual Or47b neurons",
                method="Split-GAL4 driver line + UAS-GtACR1",
                parameters={
                    "driver": "Or47b-GAL4",
                    "effector": "UAS-GtACR1",
                    "wavelength": "470nm",
                },
                expected_outcome="Selective expression in single Or47b neuron",
                measurement="Fluorescence imaging",
            ),
            ExperimentStep(
                step_number=3,
                action="Present hexanol stimulus",
                target="Fly antenna",
                method="Olfactory delivery system",
                parameters={
                    "odorant": "1-hexanol",
                    "concentration": "10^-3 dilution",
                    "duration": "2s",
                },
                expected_outcome="Baseline proboscis extension response",
                measurement="Video tracking of proboscis",
            ),
            ExperimentStep(
                step_number=4,
                action="Silence single Or47b neuron",
                target="Single identified Or47b ORN",
                method="Optogenetic silencing (GtACR1)",
                parameters={
                    "light_intensity": "1 mW/mm^2",
                    "duration": "5s",
                    "timing": "1s before odor onset",
                },
                expected_outcome="Reduced or abolished proboscis extension",
                measurement="Proboscis extension probability (% trials)",
            ),
            ExperimentStep(
                step_number=5,
                action="Quantify behavioral deficit",
                target="Proboscis extension behavior",
                method="Statistical analysis (n=30 flies, 10 trials each)",
                parameters={
                    "n_flies": 30,
                    "trials_per_fly": 10,
                    "statistics": "paired t-test",
                },
                expected_outcome="Significant reduction in feeding response",
                measurement="Response probability, latency",
            ),
        ]

        controls = [
            "Empty GAL4 control (no GtACR1)",
            "GtACR1 with no light stimulation",
            "Light stimulation of non-Or47b neurons",
            "Response to control odorant (non-hexanol)",
        ]

        metadata = {
            "pathway_strength": pathway.strength,
            "primary_odorant": "1-hexanol",
            "expected_duration": "2 weeks",
            "difficulty": "high",
            "literature_reference": "Jeanne & Wilson (2015) - sparse code concept",
        }

        protocol = ExperimentProtocol(
            experiment_id="PGCN-EXP-001",
            experiment_name="Single-Unit Veto of Or47b Feeding Response",
            hypothesis=(
                "Silencing individual Or47b ORNs is sufficient to disrupt "
                "hexanol-evoked proboscis extension, demonstrating single-unit veto power."
            ),
            steps=steps,
            controls=controls,
            metadata=metadata,
        )

        return protocol

    def generate_experiment_2_protocol(self) -> ExperimentProtocol:
        """
        Generate Experiment 2: Counterfactual microsurgery.

        Tests causal necessity of specific Or47b neurons by ablation.

        Returns:
            ExperimentProtocol for counterfactual microsurgery

        Example:
            >>> generator = BlockingExperimentGenerator("door_cache")
            >>> protocol = generator.generate_experiment_2_protocol()
        """
        logger.info("Generating Experiment 2 (Counterfactual microsurgery) protocol")

        pathway = self.analyzer.trace_or47b_feeding_pathway()

        steps = [
            ExperimentStep(
                step_number=1,
                action="Baseline behavioral testing",
                target="Wild-type flies",
                method="Olfactory preference assay",
                parameters={
                    "odorant": "1-hexanol",
                    "concentration": "10^-3",
                    "n_flies": 50,
                },
                expected_outcome="~80% attraction to hexanol",
                measurement="Preference index (-1 to +1)",
            ),
            ExperimentStep(
                step_number=2,
                action="Targeted neuronal ablation",
                target="Or47b ORNs",
                method="Laser ablation or cell-specific apoptosis",
                parameters={
                    "method": "UAS-reaper",
                    "driver": "Or47b-GAL4",
                    "timing": "developmental",
                },
                expected_outcome="Complete ablation of Or47b neurons",
                measurement="Immunostaining verification",
            ),
            ExperimentStep(
                step_number=3,
                action="Post-ablation behavioral testing",
                target="Or47b-ablated flies",
                method="Olfactory preference assay (identical to step 1)",
                parameters={
                    "odorant": "1-hexanol",
                    "concentration": "10^-3",
                    "n_flies": 50,
                },
                expected_outcome="Abolished attraction to hexanol",
                measurement="Preference index (-1 to +1)",
            ),
            ExperimentStep(
                step_number=4,
                action="Test odorant specificity",
                target="Or47b-ablated flies",
                method="Panel of odorants",
                parameters={
                    "odorants": ["ethyl acetate", "acetic acid", "apple cider vinegar"],
                    "n_odorants": 10,
                },
                expected_outcome="Normal responses to non-hexanol odorants",
                measurement="Preference index for each odorant",
            ),
        ]

        controls = [
            "Ablation of non-Or47b control neurons",
            "GAL4-only control (no UAS-reaper)",
            "Mechanical injury control",
        ]

        metadata = {
            "pathway_strength": pathway.strength,
            "intervention_type": "genetic ablation",
            "expected_duration": "4 weeks",
            "difficulty": "medium",
        }

        protocol = ExperimentProtocol(
            experiment_id="PGCN-EXP-002",
            experiment_name="Counterfactual Microsurgery of Or47b Pathway",
            hypothesis=(
                "Genetic ablation of Or47b neurons eliminates hexanol attraction "
                "while preserving responses to other odorants."
            ),
            steps=steps,
            controls=controls,
            metadata=metadata,
        )

        return protocol

    def generate_experiment_3_protocol(self) -> ExperimentProtocol:
        """
        Generate Experiment 3: Synaptic tagging and plasticity tracking.

        Monitors synaptic changes in Or47b pathway during learning.

        Returns:
            ExperimentProtocol for synaptic tagging experiment

        Example:
            >>> generator = BlockingExperimentGenerator("door_cache")
            >>> protocol = generator.generate_experiment_3_protocol()
        """
        logger.info("Generating Experiment 3 (Synaptic tagging) protocol")

        steps = [
            ExperimentStep(
                step_number=1,
                action="Express synaptic markers",
                target="Or47b → PN synapses",
                method="Trans-synaptic labeling",
                parameters={
                    "presynaptic": "Or47b-GAL4 > UAS-syt-GFP",
                    "postsynaptic": "GH146-LexA > LexAop-mCherry",
                },
                expected_outcome="Visualization of Or47b synapses in antennal lobe",
                measurement="Confocal imaging of glomerulus DM1",
            ),
            ExperimentStep(
                step_number=2,
                action="Baseline imaging",
                target="Or47b synaptic boutons",
                method="Two-photon microscopy",
                parameters={
                    "imaging_depth": "50-100μm",
                    "resolution": "0.5μm/pixel",
                    "time_series": "before conditioning",
                },
                expected_outcome="Quantified baseline synapse density",
                measurement="Synapse count per 100μm^3",
            ),
            ExperimentStep(
                step_number=3,
                action="Olfactory conditioning",
                target="Fly behavior",
                method="Hexanol + sugar reward pairing",
                parameters={
                    "odorant": "1-hexanol",
                    "reward": "2M sucrose",
                    "trials": 10,
                    "inter_trial_interval": "5min",
                },
                expected_outcome="Enhanced hexanol attraction (learning)",
                measurement="Pre/post preference index",
            ),
            ExperimentStep(
                step_number=4,
                action="Post-learning imaging",
                target="Or47b synaptic boutons",
                method="Two-photon microscopy (same parameters)",
                parameters={
                    "timing": "2h and 24h after conditioning",
                    "same_neurons": True,
                },
                expected_outcome="Increased synapse density or bouton size",
                measurement="Synapse count, bouton volume",
            ),
            ExperimentStep(
                step_number=5,
                action="Plasticity quantification",
                target="Synaptic changes",
                method="Statistical comparison",
                parameters={
                    "n_flies": 15,
                    "n_glomeruli": "bilateral DM1",
                    "statistics": "repeated-measures ANOVA",
                },
                expected_outcome="Significant synapse potentiation",
                measurement="% change from baseline",
            ),
        ]

        controls = [
            "Unpaired odorant and reward presentation",
            "Conditioning with control odorant",
            "Imaging without conditioning",
        ]

        metadata = {
            "target_glomerulus": "DM1",
            "plasticity_timepoints": ["2h", "24h", "48h"],
            "expected_duration": "6 weeks",
            "difficulty": "very high",
        }

        protocol = ExperimentProtocol(
            experiment_id="PGCN-EXP-003",
            experiment_name="Synaptic Plasticity Tracking in Or47b Pathway",
            hypothesis=(
                "Appetitive conditioning with hexanol induces structural synaptic "
                "plasticity at Or47b → PN synapses in glomerulus DM1."
            ),
            steps=steps,
            controls=controls,
            metadata=metadata,
        )

        return protocol

    def generate_experiment_6_protocol(self) -> ExperimentProtocol:
        """
        Generate Experiment 6: Causal blocker map.

        Systematically map which neurons can block specific behaviors.

        Returns:
            ExperimentProtocol for causal blocker mapping

        Example:
            >>> generator = BlockingExperimentGenerator("door_cache")
            >>> protocol = generator.generate_experiment_6_protocol()
        """
        logger.info("Generating Experiment 6 (Causal blocker map) protocol")

        # Get list of all critical receptors
        pathway = self.analyzer.trace_or47b_feeding_pathway()
        importance = self.analyzer.compute_shapley_importance("feeding")
        top_receptors = sorted(importance.items(), key=lambda x: -x[1])[:20]

        steps = [
            ExperimentStep(
                step_number=1,
                action="Generate receptor-specific lines",
                target="Top 20 olfactory receptors",
                method="GAL4 driver collection",
                parameters={
                    "receptors": [r[0] for r in top_receptors],
                    "effector": "UAS-Kir2.1 (silencer)",
                },
                expected_outcome="20 receptor-specific silencing lines",
                measurement="Line validation by expression pattern",
            ),
            ExperimentStep(
                step_number=2,
                action="Systematic behavioral screening",
                target="Each receptor line",
                method="High-throughput olfactory assay",
                parameters={
                    "odorants": ["hexanol", "ethyl butyrate", "acetic acid"],
                    "n_flies_per_line": 30,
                    "behaviors": ["attraction", "avoidance", "feeding"],
                },
                expected_outcome="Behavioral profile for each receptor",
                measurement="Preference index matrix (20 x 3 odorants)",
            ),
            ExperimentStep(
                step_number=3,
                action="Identify causal blockers",
                target="Behavioral deficits",
                method="Statistical threshold",
                parameters={
                    "threshold": "50% reduction",
                    "significance": "p < 0.01, Bonferroni corrected",
                },
                expected_outcome="~5-10 causal blockers per behavior",
                measurement="List of receptor-behavior pairs",
            ),
            ExperimentStep(
                step_number=4,
                action="Network visualization",
                target="Causal blocker network",
                method="Graph construction",
                parameters={
                    "nodes": "receptors + behaviors",
                    "edges": "causal blocking relationships",
                    "weights": "blocking strength",
                },
                expected_outcome="Receptor-behavior causal network",
                measurement="Network graph with edge weights",
            ),
        ]

        controls = [
            "Empty GAL4 control for each line",
            "Non-silenced UAS-GFP control",
            "Wild-type behavioral baseline",
        ]

        metadata = {
            "n_receptors": len(top_receptors),
            "n_odorants": 3,
            "total_conditions": len(top_receptors) * 3,
            "expected_duration": "12 weeks",
            "difficulty": "high",
            "output": "receptor-behavior causal map",
        }

        protocol = ExperimentProtocol(
            experiment_id="PGCN-EXP-006",
            experiment_name="Systematic Causal Blocker Mapping",
            hypothesis=(
                "Systematic silencing of individual receptor types reveals "
                "a sparse causal map where each behavior depends on a small "
                "subset of critical receptors."
            ),
            steps=steps,
            controls=controls,
            metadata=metadata,
        )

        return protocol

    def generate_custom_protocol(
        self,
        experiment_name: str,
        hypothesis: str,
        target_receptors: List[str],
        intervention_type: str = "optogenetic silencing",
    ) -> ExperimentProtocol:
        """
        Generate custom experimental protocol.

        Args:
            experiment_name: Name of the experiment
            hypothesis: Testable hypothesis
            target_receptors: List of receptors to target
            intervention_type: Type of intervention

        Returns:
            ExperimentProtocol with custom configuration

        Example:
            >>> generator = BlockingExperimentGenerator("door_cache")
            >>> protocol = generator.generate_custom_protocol(
            ...     experiment_name="Or92a Aversion Test",
            ...     hypothesis="Or92a is necessary for geosmin avoidance",
            ...     target_receptors=["Or92a"],
            ...     intervention_type="optogenetic silencing"
            ... )
        """
        logger.info(f"Generating custom protocol: {experiment_name}")

        steps = [
            ExperimentStep(
                step_number=1,
                action="Setup intervention",
                target=", ".join(target_receptors),
                method=intervention_type,
                parameters={"receptors": target_receptors},
                expected_outcome="Functional silencing",
                measurement="Electrophysiology or imaging",
            ),
            ExperimentStep(
                step_number=2,
                action="Behavioral testing",
                target="Fly behavior",
                method="Olfactory preference assay",
                parameters={"n_flies": 30, "trials": 10},
                expected_outcome="Altered behavior",
                measurement="Preference index",
            ),
        ]

        controls = [
            "Empty vector control",
            "No intervention control",
        ]

        protocol = ExperimentProtocol(
            experiment_id=f"PGCN-CUSTOM-{datetime.now().strftime('%Y%m%d')}",
            experiment_name=experiment_name,
            hypothesis=hypothesis,
            steps=steps,
            controls=controls,
            metadata={
                "target_receptors": target_receptors,
                "intervention_type": intervention_type,
                "custom_protocol": True,
            },
        )

        return protocol
