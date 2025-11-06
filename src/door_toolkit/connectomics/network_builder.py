"""
Network Builder
===============

Core class for building and analyzing interglomerular cross-talk networks
from FlyWire connectivity data.

This module constructs NetworkX graphs with:
- Individual neurons as nodes
- Hierarchical glomerulus meta-nodes
- Synapse-weighted edges
- Pathway information
"""

import networkx as nx
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import logging
from collections import defaultdict

from door_toolkit.connectomics.config import NetworkConfig
from door_toolkit.connectomics.data_loader import (
    ConnectivityData,
    load_connectivity_data,
    load_glomerulus_matrix,
)

logger = logging.getLogger(__name__)


class CrossTalkNetwork:
    """
    Main class for interglomerular cross-talk network analysis.

    This class builds a multi-layer directed graph representing the
    Drosophila antennal lobe circuitry with:
    - Layer 0: ORNs (individual neurons + glomerulus meta-nodes)
    - Layer 1: Local Neurons (LNs) and Projection Neurons (PNs)
    - Layer 2: Target neurons (ORNs, PNs, LNs)

    Attributes:
        config: NetworkConfig object controlling network construction
        data: ConnectivityData object with raw pathway data
        graph: NetworkX DiGraph with full network
        glomerulus_graph: Aggregated graph at glomerulus level

    Example:
        >>> network = CrossTalkNetwork.from_csv('pathways.csv')
        >>> network.set_min_synapse_threshold(5)
        >>> results = network.analyze_single_orn('ORN_DL5')
    """

    def __init__(
        self,
        data: ConnectivityData,
        config: Optional[NetworkConfig] = None
    ):
        """
        Initialize CrossTalkNetwork.

        Args:
            data: ConnectivityData object with pathway information
            config: NetworkConfig object (uses default if None)
        """
        self.config = config or NetworkConfig()
        self.data = data
        self.graph: Optional[nx.DiGraph] = None
        self.glomerulus_graph: Optional[nx.DiGraph] = None
        self._neuron_to_glomerulus: Dict[str, str] = {}
        self._glomerulus_to_neurons: Dict[str, List[str]] = defaultdict(list)

        # Build the network
        self._build_network()

    @classmethod
    def from_csv(
        cls,
        filepath: Union[str, Path],
        config: Optional[NetworkConfig] = None
    ) -> 'CrossTalkNetwork':
        """
        Create CrossTalkNetwork from a CSV file.

        Args:
            filepath: Path to connectivity CSV file
            config: NetworkConfig object (optional)

        Returns:
            CrossTalkNetwork instance
        """
        if config is None:
            config = NetworkConfig()

        data = load_connectivity_data(filepath, config)
        return cls(data, config)

    @classmethod
    def from_multiple_files(
        cls,
        filepaths: Dict[str, Union[str, Path]],
        config: Optional[NetworkConfig] = None
    ) -> 'CrossTalkNetwork':
        """
        Create CrossTalkNetwork from multiple pathway type files.

        Args:
            filepaths: Dictionary mapping pathway types to file paths
            config: NetworkConfig object (optional)

        Returns:
            CrossTalkNetwork instance
        """
        from door_toolkit.connectomics.data_loader import load_multiple_pathway_types

        if config is None:
            config = NetworkConfig()

        data = load_multiple_pathway_types(filepaths, config)
        return cls(data, config)

    def _build_network(self) -> None:
        """Build the NetworkX graph from connectivity data."""
        logger.info("Building network graph...")

        self.graph = nx.DiGraph()

        # Add all neurons as nodes with their properties
        for neuron_id, props in self.data.neurons.items():
            self.graph.add_node(
                neuron_id,
                **props,
                node_type='neuron'
            )

            # Track ORN-to-glomerulus mapping
            if props['category'] == 'ORN' and 'glomerulus' in props:
                glom = props['glomerulus']
                self._neuron_to_glomerulus[neuron_id] = glom
                self._glomerulus_to_neurons[glom].append(neuron_id)

        # Add glomerulus meta-nodes
        for glomerulus in self.data.glomeruli:
            self.graph.add_node(
                f"GLOM_{glomerulus}",
                glomerulus=glomerulus,
                node_type='glomerulus_meta',
                category='Glomerulus',
                num_neurons=len(self._glomerulus_to_neurons[glomerulus])
            )

        # Add edges from pathways
        for _, pathway in self.data.pathways.iterrows():
            # ORN → Level1 connection
            self.graph.add_edge(
                pathway['orn_root_id'],
                pathway['level1_root_id'],
                weight=pathway['synapse_count_step1'],
                synapse_count=pathway['synapse_count_step1'],
                pathway_step=1,
            )

            # Level1 → Level2 connection
            self.graph.add_edge(
                pathway['level1_root_id'],
                pathway['level2_root_id'],
                weight=pathway['synapse_count_step2'],
                synapse_count=pathway['synapse_count_step2'],
                pathway_step=2,
            )

        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes "
                   f"and {self.graph.number_of_edges()} edges")

        # Build glomerulus-level graph
        self._build_glomerulus_graph()

    def _build_glomerulus_graph(self) -> None:
        """Build aggregated glomerulus-to-glomerulus graph."""
        logger.info("Building glomerulus-level graph...")

        self.glomerulus_graph = nx.DiGraph()

        # Add glomerulus nodes
        for glom in self.data.glomeruli:
            self.glomerulus_graph.add_node(
                glom,
                num_neurons=len(self._glomerulus_to_neurons[glom])
            )

        # Aggregate connections at glomerulus level
        glom_connections = defaultdict(int)

        for _, pathway in self.data.pathways.iterrows():
            source_glom = pathway['orn_glomerulus']

            # Determine target glomerulus
            target_id = pathway['level2_root_id']
            target_glom = self._neuron_to_glomerulus.get(target_id)

            if target_glom is not None and target_glom != source_glom:
                # ORN→LN→ORN pathway
                glom_connections[(source_glom, target_glom)] += \
                    pathway['synapse_count_step2']

        # Add edges
        for (source, target), weight in glom_connections.items():
            self.glomerulus_graph.add_edge(
                source,
                target,
                weight=weight,
                synapse_count=weight
            )

        logger.info(f"Built glomerulus graph with "
                   f"{self.glomerulus_graph.number_of_nodes()} nodes and "
                   f"{self.glomerulus_graph.number_of_edges()} edges")

    def set_min_synapse_threshold(self, threshold: int) -> None:
        """
        Set minimum synapse threshold and rebuild network.

        Args:
            threshold: Minimum synapse count for connections
        """
        self.config.set_min_synapse_threshold(threshold)
        # Filter data
        self.data = self.data.filter_by_synapse_count(threshold)
        # Rebuild network
        self._build_network()

    def get_neuron_info(self, neuron_id: str) -> Optional[Dict]:
        """
        Get information about a specific neuron.

        Args:
            neuron_id: Neuron root ID

        Returns:
            Dictionary with neuron properties or None if not found
        """
        if self.graph.has_node(neuron_id):
            return dict(self.graph.nodes[neuron_id])
        return None

    def get_glomerulus_neurons(self, glomerulus: str) -> List[str]:
        """
        Get all ORN neuron IDs belonging to a glomerulus.

        Args:
            glomerulus: Glomerulus name (e.g., 'ORN_DL5')

        Returns:
            List of neuron IDs
        """
        return self._glomerulus_to_neurons.get(glomerulus, [])

    def get_neuron_glomerulus(self, neuron_id: str) -> Optional[str]:
        """
        Get the glomerulus that a neuron belongs to.

        Args:
            neuron_id: Neuron root ID

        Returns:
            Glomerulus name or None if not an ORN
        """
        return self._neuron_to_glomerulus.get(neuron_id)

    def get_pathways_from_orn(
        self,
        orn_identifier: Union[str, int],
        by_glomerulus: bool = False
    ) -> List[Dict]:
        """
        Get all pathways originating from a specific ORN or glomerulus.

        Args:
            orn_identifier: Either ORN root_id or glomerulus name
            by_glomerulus: If True, get pathways for entire glomerulus

        Returns:
            List of pathway dictionaries
        """
        if by_glomerulus:
            # Get all neurons in glomerulus
            orn_ids = self.get_glomerulus_neurons(orn_identifier)
        else:
            orn_ids = [orn_identifier]

        pathways = []

        for orn_id in orn_ids:
            if not self.graph.has_node(orn_id):
                continue

            # Get Level1 neurons (LNs/PNs) connected to this ORN
            level1_neighbors = list(self.graph.successors(orn_id))

            for level1_id in level1_neighbors:
                step1_data = self.graph[orn_id][level1_id]

                # Get Level2 neurons (targets) connected to Level1
                level2_neighbors = list(self.graph.successors(level1_id))

                for level2_id in level2_neighbors:
                    step2_data = self.graph[level1_id][level2_id]

                    pathway = {
                        'orn_id': orn_id,
                        'orn_glomerulus': self.get_neuron_glomerulus(orn_id),
                        'level1_id': level1_id,
                        'level1_type': self.graph.nodes[level1_id]['type'],
                        'level1_category': self.graph.nodes[level1_id]['category'],
                        'level2_id': level2_id,
                        'level2_type': self.graph.nodes[level2_id]['type'],
                        'level2_category': self.graph.nodes[level2_id]['category'],
                        'level2_glomerulus': self.get_neuron_glomerulus(level2_id),
                        'synapse_count_step1': step1_data['synapse_count'],
                        'synapse_count_step2': step2_data['synapse_count'],
                    }
                    pathways.append(pathway)

        return pathways

    def get_pathways_between_orns(
        self,
        source_orn: str,
        target_orn: str,
        by_glomerulus: bool = False
    ) -> List[Dict]:
        """
        Find all pathways between two ORNs/glomeruli.

        Args:
            source_orn: Source ORN/glomerulus identifier
            target_orn: Target ORN/glomerulus identifier
            by_glomerulus: If True, treat identifiers as glomerulus names

        Returns:
            List of pathway dictionaries
        """
        all_pathways = self.get_pathways_from_orn(source_orn, by_glomerulus)

        if by_glomerulus:
            target_ids = set(self.get_glomerulus_neurons(target_orn))
        else:
            target_ids = {target_orn}

        # Filter for pathways ending at target
        matching_pathways = [
            p for p in all_pathways
            if p['level2_id'] in target_ids
        ]

        return matching_pathways

    def find_shortest_paths(
        self,
        source: str,
        target: str,
        max_paths: int = 10
    ) -> List[List[str]]:
        """
        Find shortest paths between two neurons.

        Args:
            source: Source neuron ID
            target: Target neuron ID
            max_paths: Maximum number of paths to return

        Returns:
            List of paths (each path is a list of neuron IDs)
        """
        try:
            # Find all shortest paths
            paths = list(nx.all_shortest_paths(
                self.graph,
                source,
                target
            ))
            return paths[:max_paths]
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def get_hub_neurons(
        self,
        neuron_category: Optional[str] = None,
        top_n: int = 10,
        by_degree: str = 'out'
    ) -> List[Tuple[str, int]]:
        """
        Identify hub neurons with highest connectivity.

        Args:
            neuron_category: Filter by category ('Local_Neuron', 'ORN', etc.)
            top_n: Number of top hubs to return
            by_degree: 'in', 'out', or 'total' degree

        Returns:
            List of (neuron_id, degree) tuples
        """
        degree_func = {
            'in': self.graph.in_degree,
            'out': self.graph.out_degree,
            'total': self.graph.degree,
        }[by_degree]

        # Filter nodes by category if specified
        if neuron_category:
            nodes = [
                n for n, d in self.graph.nodes(data=True)
                if d.get('category') == neuron_category
                and d.get('node_type') == 'neuron'
            ]
        else:
            nodes = [
                n for n, d in self.graph.nodes(data=True)
                if d.get('node_type') == 'neuron'
            ]

        # Calculate degrees
        degrees = [(n, degree_func(n)) for n in nodes]

        # Sort and return top N
        degrees.sort(key=lambda x: x[1], reverse=True)
        return degrees[:top_n]

    def get_intermediate_neurons(
        self,
        source_glomerulus: str,
        target_glomerulus: str
    ) -> Dict[str, List[str]]:
        """
        Find all intermediate neurons (LNs/PNs) connecting two glomeruli.

        Args:
            source_glomerulus: Source glomerulus name
            target_glomerulus: Target glomerulus name

        Returns:
            Dictionary with 'LNs' and 'PNs' keys, each containing list of neuron IDs
        """
        pathways = self.get_pathways_between_orns(
            source_glomerulus,
            target_glomerulus,
            by_glomerulus=True
        )

        intermediates = {'LNs': [], 'PNs': []}

        for pathway in pathways:
            if pathway['level1_category'] == 'Local_Neuron':
                if pathway['level1_id'] not in intermediates['LNs']:
                    intermediates['LNs'].append(pathway['level1_id'])
            elif pathway['level1_category'] == 'Projection_Neuron':
                if pathway['level1_id'] not in intermediates['PNs']:
                    intermediates['PNs'].append(pathway['level1_id'])

        return intermediates

    def get_network_statistics(self) -> Dict:
        """
        Calculate comprehensive network statistics.

        Returns:
            Dictionary with network metrics
        """
        # Basic statistics
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_glomeruli': len(self.data.glomeruli),
            'num_pathways': self.data.num_pathways,
        }

        # Count by category
        category_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            if data.get('node_type') == 'neuron':
                category_counts[data['category']] += 1
        stats['neurons_by_category'] = dict(category_counts)

        # Connectivity statistics
        neuron_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get('node_type') == 'neuron'
        ]

        in_degrees = [self.graph.in_degree(n) for n in neuron_nodes]
        out_degrees = [self.graph.out_degree(n) for n in neuron_nodes]

        stats['connectivity'] = {
            'mean_in_degree': np.mean(in_degrees),
            'mean_out_degree': np.mean(out_degrees),
            'max_in_degree': np.max(in_degrees),
            'max_out_degree': np.max(out_degrees),
        }

        # Synapse weight statistics
        edge_weights = [d['weight'] for _, _, d in self.graph.edges(data=True)]
        stats['synapse_weights'] = {
            'mean': np.mean(edge_weights),
            'median': np.median(edge_weights),
            'std': np.std(edge_weights),
            'min': np.min(edge_weights),
            'max': np.max(edge_weights),
        }

        # Glomerulus graph statistics
        if self.glomerulus_graph:
            stats['glomerulus_graph'] = {
                'num_connections': self.glomerulus_graph.number_of_edges(),
                'density': nx.density(self.glomerulus_graph),
            }

        return stats

    def to_networkx(self) -> nx.DiGraph:
        """
        Export the network as a NetworkX DiGraph.

        Returns:
            NetworkX DiGraph object
        """
        return self.graph.copy()

    def to_glomerulus_networkx(self) -> nx.DiGraph:
        """
        Export the glomerulus-level network as a NetworkX DiGraph.

        Returns:
            NetworkX DiGraph object
        """
        return self.glomerulus_graph.copy()

    def export_to_graphml(self, filepath: Union[str, Path]) -> None:
        """
        Export network to GraphML format (for Cytoscape, etc.).

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        nx.write_graphml(self.graph, filepath)
        logger.info(f"Exported network to {filepath}")

    def export_to_gexf(self, filepath: Union[str, Path]) -> None:
        """
        Export network to GEXF format (for Gephi, etc.).

        Args:
            filepath: Output file path
        """
        filepath = Path(filepath)
        nx.write_gexf(self.graph, filepath)
        logger.info(f"Exported network to {filepath}")

    def summary(self) -> str:
        """
        Generate summary string of the network.

        Returns:
            Multi-line summary string
        """
        stats = self.get_network_statistics()

        lines = [
            "Cross-Talk Network Summary",
            "=" * 60,
            f"Nodes: {stats['num_nodes']:,}",
            f"Edges: {stats['num_edges']:,}",
            f"Glomeruli: {stats['num_glomeruli']}",
            f"Pathways: {stats['num_pathways']:,}",
            "",
            "Neurons by category:",
        ]

        for category, count in stats['neurons_by_category'].items():
            lines.append(f"  {category}: {count:,}")

        lines.extend([
            "",
            "Connectivity:",
            f"  Mean in-degree: {stats['connectivity']['mean_in_degree']:.2f}",
            f"  Mean out-degree: {stats['connectivity']['mean_out_degree']:.2f}",
            f"  Max in-degree: {stats['connectivity']['max_in_degree']}",
            f"  Max out-degree: {stats['connectivity']['max_out_degree']}",
            "",
            "Synapse weights:",
            f"  Mean: {stats['synapse_weights']['mean']:.2f}",
            f"  Median: {stats['synapse_weights']['median']:.0f}",
            f"  Range: {stats['synapse_weights']['min']} - {stats['synapse_weights']['max']}",
        ])

        if 'glomerulus_graph' in stats:
            lines.extend([
                "",
                "Glomerulus-level network:",
                f"  Connections: {stats['glomerulus_graph']['num_connections']}",
                f"  Density: {stats['glomerulus_graph']['density']:.4f}",
            ])

        return "\n".join(lines)
