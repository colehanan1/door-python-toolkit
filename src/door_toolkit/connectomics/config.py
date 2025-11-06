"""
Network Configuration
=====================

Configuration parameters for biophysically realistic spiking neural network models
of the Drosophila antennal lobe.

Parameters are based on experimental data from:
- Wilson & Laurent (2005) - Role of GABAergic inhibition in shaping odor-evoked patterns
- Olsen & Wilson (2008) - Lateral presynaptic inhibition mediates gain control
- Kazama & Wilson (2009) - Origins of correlated activity in olfactory glomeruli
- Nagel & Wilson (2011) - Biophysical mechanisms underlying olfactory receptor neuron dynamics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json


@dataclass
class BiophysicalParameters:
    """
    Biophysical parameters for different neuron types in the antennal lobe.

    All values are research-based from published literature on Drosophila neurons.
    """

    # ORN (Olfactory Receptor Neuron) parameters
    orn_params: Dict[str, float] = field(default_factory=lambda: {
        "tau_m": 20.0,              # Membrane time constant (ms)
        "tau_ref": 2.0,             # Refractory period (ms)
        "v_rest": -65.0,            # Resting potential (mV)
        "v_thresh": -50.0,          # Threshold potential (mV)
        "v_reset": -70.0,           # Reset potential (mV)
        "cm": 0.2,                  # Membrane capacitance (nF)
        "r_m": 100.0,               # Membrane resistance (MOhm)
    })

    # LN (Local Neuron) parameters - primarily GABAergic inhibitory
    ln_params: Dict[str, float] = field(default_factory=lambda: {
        "tau_m": 15.0,              # Faster time constant
        "tau_ref": 1.5,             # Shorter refractory period
        "v_rest": -60.0,
        "v_thresh": -45.0,
        "v_reset": -65.0,
        "cm": 0.15,
        "r_m": 80.0,
    })

    # PN (Projection Neuron) parameters - cholinergic excitatory
    pn_params: Dict[str, float] = field(default_factory=lambda: {
        "tau_m": 25.0,              # Longer time constant
        "tau_ref": 2.0,
        "v_rest": -65.0,
        "v_thresh": -48.0,
        "v_reset": -68.0,
        "cm": 0.25,
        "r_m": 120.0,
    })

    # Synaptic parameters
    synapse_params: Dict[str, float] = field(default_factory=lambda: {
        # Excitatory (acetylcholine) - ORN→PN, PN→downstream
        "tau_ach": 5.0,             # ACh time constant (ms)
        "e_ach": 0.0,               # ACh reversal potential (mV)

        # Inhibitory (GABA) - LN→target
        "tau_gaba": 10.0,           # GABA_A time constant (ms)
        "e_gaba": -80.0,            # GABA reversal potential (mV)

        # Synaptic delays
        "delay_fast": 0.5,          # Fast connections (ms)
        "delay_slow": 2.0,          # Slow connections (ms)
    })


@dataclass
class NetworkConfig:
    """
    Configuration for the interglomerular cross-talk network.

    This class controls all aspects of network construction, simulation,
    and analysis.

    Attributes:
        min_synapse_threshold: Minimum synapse count to include a connection (default: 1)
        weight_scaling_factor: Converts synapse count to conductance (nS) (default: 0.1)
        include_orn_ln_orn: Include ORN→LN→ORN pathways (lateral inhibition)
        include_orn_ln_pn: Include ORN→LN→PN pathways (feedforward inhibition)
        include_orn_pn_feedback: Include ORN→PN→feedback pathways
        simulation_time: Total simulation time (ms)
        dt: Simulation timestep (ms)
        biophysical_params: Container for all biophysical parameters

    Example:
        >>> config = NetworkConfig()
        >>> config.min_synapse_threshold = 10  # Only strong connections
        >>> config.include_orn_pn_feedback = False  # Exclude feedback
    """

    # Connection filtering
    min_synapse_threshold: int = 1
    max_synapse_threshold: Optional[int] = None

    # Pathway selection
    include_orn_ln_orn: bool = True
    include_orn_ln_pn: bool = True
    include_orn_pn_feedback: bool = True

    # Synapse weight conversion
    weight_scaling_factor: float = 0.1  # synapse_count → conductance (nS)

    # Simulation parameters
    simulation_time: float = 1000.0  # ms
    dt: float = 0.1  # ms

    # Biophysical parameters
    biophysical_params: BiophysicalParameters = field(
        default_factory=BiophysicalParameters
    )

    # Visualization parameters
    viz_node_size_range: tuple = (10, 100)
    viz_edge_width_range: tuple = (0.1, 5.0)
    viz_dpi: int = 300
    viz_figure_size: tuple = (12, 10)

    # Statistical analysis parameters
    community_detection_algorithm: str = "louvain"  # or "greedy", "label_propagation"
    hub_threshold_percentile: float = 90.0  # Top 10% are considered hubs

    # Network layout parameters
    layout_algorithm: str = "spring"  # "spring", "kamada_kawai", "circular", "hierarchical"
    layout_iterations: int = 50
    layout_seed: Optional[int] = 42  # For reproducibility

    # Data file paths (can be overridden)
    default_data_paths: Dict[str, str] = field(default_factory=lambda: {
        "full": "interglomerular_crosstalk_pathways.csv",
        "orn_ln_orn": "crosstalk_ORN_LN_ORN.csv",
        "orn_ln_pn": "crosstalk_ORN_LN_PN.csv",
        "orn_pn_feedback": "crosstalk_ORN_PN_feedback.csv",
        "glomerulus_matrix": "crosstalk_matrix_glomerulus.csv",
    })

    def set_min_synapse_threshold(self, threshold: int) -> None:
        """Set minimum synapse threshold for connection inclusion."""
        if threshold < 1:
            raise ValueError("Threshold must be at least 1")
        self.min_synapse_threshold = threshold

    def set_pathway_filters(
        self,
        orn_ln_orn: bool = True,
        orn_ln_pn: bool = True,
        orn_pn_feedback: bool = True
    ) -> None:
        """
        Set which pathway types to include in the network.

        Args:
            orn_ln_orn: Include lateral inhibition pathways
            orn_ln_pn: Include feedforward inhibition to PNs
            orn_pn_feedback: Include PN feedback loops
        """
        self.include_orn_ln_orn = orn_ln_orn
        self.include_orn_ln_pn = orn_ln_pn
        self.include_orn_pn_feedback = orn_pn_feedback

    def get_neuron_params(self, neuron_category: str) -> Dict[str, float]:
        """
        Get biophysical parameters for a specific neuron category.

        Args:
            neuron_category: One of 'ORN', 'Local_Neuron', 'Projection_Neuron'

        Returns:
            Dictionary of biophysical parameters
        """
        category_map = {
            "ORN": self.biophysical_params.orn_params,
            "Local_Neuron": self.biophysical_params.ln_params,
            "Projection_Neuron": self.biophysical_params.pn_params,
        }

        if neuron_category not in category_map:
            raise ValueError(
                f"Unknown neuron category: {neuron_category}. "
                f"Must be one of {list(category_map.keys())}"
            )

        return category_map[neuron_category]

    def get_synapse_params(self, presynaptic_category: str) -> Dict[str, float]:
        """
        Get synaptic parameters based on presynaptic neuron type.

        Implements Dale's law: each neuron type has consistent neurotransmitter.
        - ORNs: Cholinergic (excitatory)
        - PNs: Cholinergic (excitatory)
        - LNs: GABAergic (inhibitory)

        Args:
            presynaptic_category: Category of presynaptic neuron

        Returns:
            Dictionary with tau (time constant), e_rev (reversal potential),
            and delay parameters
        """
        synapse_base = self.biophysical_params.synapse_params

        if presynaptic_category == "Local_Neuron":
            # GABAergic inhibitory
            return {
                "tau": synapse_base["tau_gaba"],
                "e_rev": synapse_base["e_gaba"],
                "delay": synapse_base["delay_fast"],
                "type": "inhibitory",
            }
        elif presynaptic_category in ["ORN", "Projection_Neuron"]:
            # Cholinergic excitatory
            return {
                "tau": synapse_base["tau_ach"],
                "e_rev": synapse_base["e_ach"],
                "delay": synapse_base["delay_slow"],
                "type": "excitatory",
            }
        else:
            raise ValueError(f"Unknown presynaptic category: {presynaptic_category}")

    def to_dict(self) -> Dict:
        """Convert configuration to dictionary for serialization."""
        return {
            "min_synapse_threshold": self.min_synapse_threshold,
            "max_synapse_threshold": self.max_synapse_threshold,
            "include_orn_ln_orn": self.include_orn_ln_orn,
            "include_orn_ln_pn": self.include_orn_ln_pn,
            "include_orn_pn_feedback": self.include_orn_pn_feedback,
            "weight_scaling_factor": self.weight_scaling_factor,
            "simulation_time": self.simulation_time,
            "dt": self.dt,
            "biophysical_params": {
                "orn": self.biophysical_params.orn_params,
                "ln": self.biophysical_params.ln_params,
                "pn": self.biophysical_params.pn_params,
                "synapse": self.biophysical_params.synapse_params,
            },
        }

    def to_json(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, filepath: str) -> 'NetworkConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        config = cls()

        # Update basic parameters
        for key in ['min_synapse_threshold', 'max_synapse_threshold',
                    'include_orn_ln_orn', 'include_orn_ln_pn',
                    'include_orn_pn_feedback', 'weight_scaling_factor',
                    'simulation_time', 'dt']:
            if key in data:
                setattr(config, key, data[key])

        # Update biophysical parameters if present
        if 'biophysical_params' in data:
            bp_data = data['biophysical_params']
            if 'orn' in bp_data:
                config.biophysical_params.orn_params.update(bp_data['orn'])
            if 'ln' in bp_data:
                config.biophysical_params.ln_params.update(bp_data['ln'])
            if 'pn' in bp_data:
                config.biophysical_params.pn_params.update(bp_data['pn'])
            if 'synapse' in bp_data:
                config.biophysical_params.synapse_params.update(bp_data['synapse'])

        return config


# Predefined configurations for common use cases
def get_default_config() -> NetworkConfig:
    """Get default configuration with all pathways included."""
    return NetworkConfig()


def get_strong_connections_config() -> NetworkConfig:
    """Get configuration with only strong connections (>=10 synapses)."""
    config = NetworkConfig()
    config.set_min_synapse_threshold(10)
    return config


def get_lateral_inhibition_config() -> NetworkConfig:
    """Get configuration focused on lateral inhibition (ORN→LN→ORN only)."""
    config = NetworkConfig()
    config.set_pathway_filters(
        orn_ln_orn=True,
        orn_ln_pn=False,
        orn_pn_feedback=False
    )
    return config


def get_feedforward_config() -> NetworkConfig:
    """Get configuration for feedforward pathways only."""
    config = NetworkConfig()
    config.set_pathway_filters(
        orn_ln_orn=False,
        orn_ln_pn=True,
        orn_pn_feedback=False
    )
    return config
