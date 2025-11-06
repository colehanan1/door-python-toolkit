"""
Unit tests for the connectomics module.

Tests cover:
- Network construction
- Data loading
- Pathway analysis (all 4 modes)
- Statistical analyses
- Configuration
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from door_toolkit.connectomics.config import NetworkConfig
from door_toolkit.connectomics.data_loader import ConnectivityData, load_connectivity_data
from door_toolkit.connectomics.network_builder import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import (
    analyze_single_orn,
    compare_orn_pair,
    find_pathways
)
from door_toolkit.connectomics.statistics import NetworkStatistics


# Fixture: Create sample connectivity data
@pytest.fixture
def sample_connectivity_data():
    """Create minimal sample connectivity data for testing."""
    data = {
        'orn_root_id': ['orn1', 'orn1', 'orn2', 'orn2', 'orn3'],
        'orn_label': ['Or7a', 'Or7a', 'Or47b', 'Or47b', 'Or69a'],
        'orn_glomerulus': ['ORN_DL5', 'ORN_DL5', 'ORN_VA1v', 'ORN_VA1v', 'ORN_D'],
        'level1_root_id': ['ln1', 'ln2', 'ln1', 'pn1', 'ln3'],
        'level1_cell_type': ['lLN2F_a', 'lLN2P_b', 'lLN2F_a', 'VA1v_vPN', 'lLN1_bc'],
        'level1_category': ['Local_Neuron', 'Local_Neuron', 'Local_Neuron',
                            'Projection_Neuron', 'Local_Neuron'],
        'level2_root_id': ['orn2', 'orn3', 'orn3', 'ln1', 'pn1'],
        'level2_cell_type': ['Or47b', 'Or69a', 'Or69a', 'lLN2F_a', 'VA1v_vPN'],
        'level2_category': ['ORN', 'ORN', 'ORN', 'Local_Neuron', 'Projection_Neuron'],
        'synapse_count_step1': [15, 20, 12, 18, 8],
        'synapse_count_step2': [25, 30, 15, 22, 10],
    }

    df = pd.DataFrame(data)

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    return temp_path


class TestNetworkConfig:
    """Test NetworkConfig class."""

    def test_default_config(self):
        """Test default configuration."""
        config = NetworkConfig()
        assert config.min_synapse_threshold == 1
        assert config.include_orn_ln_orn == True
        assert config.include_orn_ln_pn == True
        assert config.include_orn_pn_feedback == True

    def test_set_threshold(self):
        """Test setting synapse threshold."""
        config = NetworkConfig()
        config.set_min_synapse_threshold(10)
        assert config.min_synapse_threshold == 10

        # Test invalid threshold
        with pytest.raises(ValueError):
            config.set_min_synapse_threshold(0)

    def test_pathway_filters(self):
        """Test pathway filter configuration."""
        config = NetworkConfig()
        config.set_pathway_filters(
            orn_ln_orn=False,
            orn_ln_pn=True,
            orn_pn_feedback=False
        )
        assert config.include_orn_ln_orn == False
        assert config.include_orn_ln_pn == True
        assert config.include_orn_pn_feedback == False

    def test_neuron_params(self):
        """Test neuron parameter retrieval."""
        config = NetworkConfig()

        orn_params = config.get_neuron_params('ORN')
        assert 'tau_m' in orn_params
        assert orn_params['tau_m'] == 20.0

        ln_params = config.get_neuron_params('Local_Neuron')
        assert ln_params['tau_m'] == 15.0

        # Test invalid category
        with pytest.raises(ValueError):
            config.get_neuron_params('InvalidCategory')

    def test_synapse_params(self):
        """Test synaptic parameter retrieval."""
        config = NetworkConfig()

        # Test inhibitory (LN)
        ln_synapse = config.get_synapse_params('Local_Neuron')
        assert ln_synapse['type'] == 'inhibitory'
        assert ln_synapse['e_rev'] == -80.0

        # Test excitatory (ORN)
        orn_synapse = config.get_synapse_params('ORN')
        assert orn_synapse['type'] == 'excitatory'
        assert orn_synapse['e_rev'] == 0.0

    def test_json_serialization(self):
        """Test configuration save/load."""
        config = NetworkConfig()
        config.set_min_synapse_threshold(15)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config.to_json(f.name)
            temp_path = f.name

        loaded_config = NetworkConfig.from_json(temp_path)
        assert loaded_config.min_synapse_threshold == 15


class TestDataLoader:
    """Test data loading functionality."""

    def test_load_connectivity_data(self, sample_connectivity_data):
        """Test loading connectivity data from CSV."""
        data = load_connectivity_data(sample_connectivity_data)

        assert isinstance(data, ConnectivityData)
        assert data.num_pathways == 5
        assert len(data.glomeruli) == 3
        assert 'ORN_DL5' in data.glomeruli

    def test_filter_by_synapse_count(self, sample_connectivity_data):
        """Test filtering by synapse count."""
        data = load_connectivity_data(sample_connectivity_data)

        filtered = data.filter_by_synapse_count(min_count=20)
        assert filtered.num_pathways < data.num_pathways

    def test_get_pathways_for_orn(self, sample_connectivity_data):
        """Test retrieving pathways for specific ORN."""
        data = load_connectivity_data(sample_connectivity_data)

        pathways = data.get_pathways_for_orn('ORN_DL5', by_glomerulus=True)
        assert len(pathways) == 2  # Two pathways from DL5 in sample data


class TestCrossTalkNetwork:
    """Test CrossTalkNetwork class."""

    def test_network_construction(self, sample_connectivity_data):
        """Test basic network construction."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        assert network.graph is not None
        assert network.glomerulus_graph is not None
        assert network.graph.number_of_nodes() > 0
        assert network.graph.number_of_edges() > 0

    def test_set_threshold(self, sample_connectivity_data):
        """Test changing synapse threshold."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        initial_edges = network.graph.number_of_edges()

        network.set_min_synapse_threshold(20)
        new_edges = network.graph.number_of_edges()

        assert new_edges <= initial_edges

    def test_get_neuron_info(self, sample_connectivity_data):
        """Test neuron information retrieval."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        neuron_info = network.get_neuron_info('orn1')
        assert neuron_info is not None
        assert 'category' in neuron_info

    def test_get_glomerulus_neurons(self, sample_connectivity_data):
        """Test glomerulus neuron retrieval."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        dl5_neurons = network.get_glomerulus_neurons('ORN_DL5')
        assert len(dl5_neurons) > 0
        assert 'orn1' in dl5_neurons

    def test_get_pathways_from_orn(self, sample_connectivity_data):
        """Test pathway retrieval from ORN."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        pathways = network.get_pathways_from_orn('ORN_DL5', by_glomerulus=True)
        assert len(pathways) > 0
        assert all('orn_glomerulus' in p for p in pathways)

    def test_get_hub_neurons(self, sample_connectivity_data):
        """Test hub neuron detection."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        hubs = network.get_hub_neurons(top_n=3)
        assert len(hubs) <= 3
        assert all(isinstance(h, tuple) and len(h) == 2 for h in hubs)

    def test_get_network_statistics(self, sample_connectivity_data):
        """Test network statistics calculation."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        stats = network.get_network_statistics()
        assert 'num_nodes' in stats
        assert 'num_edges' in stats
        assert 'num_glomeruli' in stats
        assert stats['num_nodes'] > 0


class TestPathwayAnalysis:
    """Test pathway analysis functions."""

    def test_analyze_single_orn(self, sample_connectivity_data):
        """Test single ORN analysis."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        results = analyze_single_orn(network, 'ORN_DL5', by_glomerulus=True)

        assert results.orn_identifier == 'ORN_DL5'
        assert results.is_glomerulus == True
        assert results.num_pathways > 0
        assert 'LNs' in results.intermediate_neurons
        assert 'ORNs' in results.target_neurons

    def test_single_orn_methods(self, sample_connectivity_data):
        """Test SingleORNAnalysis methods."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        results = analyze_single_orn(network, 'ORN_DL5', by_glomerulus=True)

        # Test to_dataframe
        df = results.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == results.num_pathways

        # Test get_strongest_pathways
        strongest = results.get_strongest_pathways(n=2)
        assert len(strongest) <= 2

        # Test summary
        summary = results.summary()
        assert isinstance(summary, str)
        assert 'ORN_DL5' in summary

    def test_compare_orn_pair(self, sample_connectivity_data):
        """Test ORN pair comparison."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        comparison = compare_orn_pair(
            network, 'ORN_DL5', 'ORN_VA1v', by_glomerulus=True
        )

        assert comparison.orn1 == 'ORN_DL5'
        assert comparison.orn2 == 'ORN_VA1v'
        assert 'LNs' in comparison.shared_intermediates
        assert '1_to_2' in comparison.cross_talk_strength
        assert '2_to_1' in comparison.cross_talk_strength

    def test_orn_pair_asymmetry(self, sample_connectivity_data):
        """Test asymmetry calculation."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        comparison = compare_orn_pair(network, 'ORN_DL5', 'ORN_VA1v', by_glomerulus=True)

        asymmetry = comparison.get_asymmetry_ratio()
        assert isinstance(asymmetry, float)
        assert -1.0 <= asymmetry <= 1.0

    def test_find_pathways(self, sample_connectivity_data):
        """Test pathway search."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        results = find_pathways(
            network,
            'ORN_DL5',
            'ORN_D',
            by_glomerulus=True
        )

        assert 'num_pathways' in results
        assert 'pathways' in results
        assert 'intermediate_neurons' in results
        assert 'statistics' in results


class TestNetworkStatistics:
    """Test statistical analysis."""

    def test_statistics_initialization(self, sample_connectivity_data):
        """Test NetworkStatistics initialization."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        stats = NetworkStatistics(network)

        assert stats.network == network
        assert stats.graph is not None

    def test_detect_hub_neurons(self, sample_connectivity_data):
        """Test hub neuron detection."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        stats = NetworkStatistics(network)

        hubs = stats.detect_hub_neurons(method='degree', threshold_percentile=50)
        assert isinstance(hubs, list)
        assert all(isinstance(h, tuple) for h in hubs)

    def test_detect_communities(self, sample_connectivity_data):
        """Test community detection."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        stats = NetworkStatistics(network)

        communities = stats.detect_communities(algorithm='greedy', level='glomerulus')
        assert isinstance(communities, dict)
        assert len(communities) > 0

    def test_asymmetry_matrix(self, sample_connectivity_data):
        """Test asymmetry matrix calculation."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        stats = NetworkStatistics(network)

        asym_matrix = stats.calculate_asymmetry_matrix()
        assert isinstance(asym_matrix, pd.DataFrame)

        if len(asym_matrix) > 0:
            assert 'asymmetry_ratio' in asym_matrix.columns
            assert 'source_glomerulus' in asym_matrix.columns
            assert 'target_glomerulus' in asym_matrix.columns

    def test_generate_report(self, sample_connectivity_data):
        """Test report generation."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        stats = NetworkStatistics(network)

        report = stats.generate_full_report()
        assert isinstance(report, str)
        assert len(report) > 0
        assert 'Network Statistical Analysis' in report


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_pathways(self, sample_connectivity_data):
        """Test handling of non-existent pathways."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        results = analyze_single_orn(network, 'NonExistentGlom', by_glomerulus=True)
        assert results.num_pathways == 0

    def test_invalid_neuron_id(self, sample_connectivity_data):
        """Test invalid neuron ID handling."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)

        info = network.get_neuron_info('invalid_id')
        assert info is None

    def test_high_threshold(self, sample_connectivity_data):
        """Test very high synapse threshold."""
        network = CrossTalkNetwork.from_csv(sample_connectivity_data)
        network.set_min_synapse_threshold(1000)  # Very high

        # Network should be much smaller or empty
        assert network.graph.number_of_edges() <= 5


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
