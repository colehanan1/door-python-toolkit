"""
Statistical Analysis
====================

Statistical analyses for cross-talk networks including:
- Hub neuron detection
- Community detection (glomerular clustering)
- Asymmetry quantification
- Network motif analysis
- Path length distributions
"""

import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
import logging

from door_toolkit.connectomics.network_builder import CrossTalkNetwork

logger = logging.getLogger(__name__)


class NetworkStatistics:
    """
    Comprehensive statistical analysis of cross-talk networks.

    Provides methods for analyzing network topology, connectivity patterns,
    and functional organization.
    """

    def __init__(self, network: CrossTalkNetwork):
        """
        Initialize statistics analyzer.

        Args:
            network: CrossTalkNetwork instance
        """
        self.network = network
        self.graph = network.graph
        self.glomerulus_graph = network.glomerulus_graph

        # Cache for expensive computations
        self._communities = None
        self._hub_neurons = None

    def detect_hub_neurons(
        self,
        method: str = 'degree',
        threshold_percentile: float = 90.0,
        neuron_category: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Identify hub neurons with high connectivity.

        Args:
            method: Detection method ('degree', 'betweenness', 'closeness', 'eigenvector')
            threshold_percentile: Percentile threshold for hub classification
            neuron_category: Filter by category ('Local_Neuron', 'ORN', etc.)

        Returns:
            List of (neuron_id, centrality_score) tuples sorted by score

        Example:
            >>> stats = NetworkStatistics(network)
            >>> hubs = stats.detect_hub_neurons(method='betweenness', threshold_percentile=95)
            >>> print(f"Found {len(hubs)} hub neurons")
        """
        logger.info(f"Detecting hub neurons using {method} centrality...")

        # Filter nodes
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

        if not nodes:
            return []

        # Create subgraph
        subgraph = self.graph.subgraph(nodes).copy()

        # Calculate centrality
        if method == 'degree':
            centrality = dict(subgraph.degree())
        elif method == 'betweenness':
            centrality = nx.betweenness_centrality(subgraph)
        elif method == 'closeness':
            centrality = nx.closeness_centrality(subgraph)
        elif method == 'eigenvector':
            try:
                centrality = nx.eigenvector_centrality(subgraph, max_iter=1000)
            except nx.PowerIterationFailedConvergence:
                logger.warning("Eigenvector centrality failed to converge, using degree")
                centrality = dict(subgraph.degree())
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate threshold
        scores = list(centrality.values())
        threshold = np.percentile(scores, threshold_percentile)

        # Filter and sort
        hubs = [
            (node, score)
            for node, score in centrality.items()
            if score >= threshold
        ]
        hubs.sort(key=lambda x: x[1], reverse=True)

        logger.info(f"Found {len(hubs)} hub neurons (threshold: {threshold:.4f})")

        return hubs

    def detect_communities(
        self,
        algorithm: str = 'louvain',
        level: str = 'glomerulus'
    ) -> Dict[str, int]:
        """
        Detect communities (functional clusters) in the network.

        Args:
            algorithm: Algorithm to use ('louvain', 'greedy', 'label_propagation')
            level: Analysis level ('glomerulus' or 'neuron')

        Returns:
            Dictionary mapping nodes to community IDs

        Example:
            >>> stats = NetworkStatistics(network)
            >>> communities = stats.detect_communities(algorithm='louvain')
            >>> print(f"Found {max(communities.values()) + 1} communities")
        """
        logger.info(f"Detecting communities using {algorithm}...")

        # Choose graph level
        if level == 'glomerulus':
            G = self.glomerulus_graph
        else:
            G = self.graph

        # Convert to undirected for community detection
        G_undirected = G.to_undirected()

        # Detect communities
        if algorithm == 'louvain':
            try:
                import community as community_louvain
                communities = community_louvain.best_partition(G_undirected)
            except ImportError:
                logger.warning("python-louvain not available, using greedy modularity")
                communities = self._greedy_modularity(G_undirected)
        elif algorithm == 'greedy':
            communities = self._greedy_modularity(G_undirected)
        elif algorithm == 'label_propagation':
            communities_gen = nx.community.label_propagation_communities(G_undirected)
            communities = {}
            for i, comm in enumerate(communities_gen):
                for node in comm:
                    communities[node] = i
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        num_communities = max(communities.values()) + 1
        logger.info(f"Found {num_communities} communities")

        self._communities = communities
        return communities

    def _greedy_modularity(self, G: nx.Graph) -> Dict[str, int]:
        """Greedy modularity maximization."""
        communities_gen = nx.community.greedy_modularity_communities(G)
        communities = {}
        for i, comm in enumerate(communities_gen):
            for node in comm:
                communities[node] = i
        return communities

    def calculate_asymmetry_matrix(self) -> pd.DataFrame:
        """
        Calculate asymmetry in glomerulus-to-glomerulus connections.

        For each glomerulus pair, quantifies the difference in
        cross-talk strength in forward vs reverse directions.

        Returns:
            DataFrame with asymmetry scores

        Example:
            >>> stats = NetworkStatistics(network)
            >>> asym_matrix = stats.calculate_asymmetry_matrix()
            >>> print(asym_matrix.head())
        """
        logger.info("Calculating asymmetry matrix...")

        glomeruli = sorted(self.network.data.glomeruli)
        results = []

        for source_glom in glomeruli:
            for target_glom in glomeruli:
                if source_glom == target_glom:
                    continue

                # Get pathways in both directions
                pathways_forward = self.network.get_pathways_between_orns(
                    source_glom, target_glom, by_glomerulus=True
                )
                pathways_reverse = self.network.get_pathways_between_orns(
                    target_glom, source_glom, by_glomerulus=True
                )

                # Calculate strengths
                strength_forward = sum(p['synapse_count_step2'] for p in pathways_forward)
                strength_reverse = sum(p['synapse_count_step2'] for p in pathways_reverse)

                if strength_forward + strength_reverse == 0:
                    continue

                # Asymmetry ratio: -1 (reverse stronger) to +1 (forward stronger)
                asymmetry = (strength_forward - strength_reverse) / \
                           (strength_forward + strength_reverse)

                results.append({
                    'source_glomerulus': source_glom,
                    'target_glomerulus': target_glom,
                    'strength_forward': strength_forward,
                    'strength_reverse': strength_reverse,
                    'total_strength': strength_forward + strength_reverse,
                    'asymmetry_ratio': asymmetry,
                    'num_pathways_forward': len(pathways_forward),
                    'num_pathways_reverse': len(pathways_reverse),
                })

        df = pd.DataFrame(results)
        logger.info(f"Calculated asymmetry for {len(df)} glomerulus pairs")

        return df

    def analyze_path_lengths(
        self,
        source_glomerulus: Optional[str] = None,
        target_glomerulus: Optional[str] = None
    ) -> Dict:
        """
        Analyze path length distributions in the network.

        Args:
            source_glomerulus: Optional source glomerulus (None = all)
            target_glomerulus: Optional target glomerulus (None = all)

        Returns:
            Dictionary with path length statistics

        Example:
            >>> stats = NetworkStatistics(network)
            >>> path_stats = stats.analyze_path_lengths(source_glomerulus='ORN_DL5')
            >>> print(f"Average path length: {path_stats['mean_path_length']:.2f}")
        """
        logger.info("Analyzing path lengths...")

        path_lengths = []

        if source_glomerulus and target_glomerulus:
            # Specific pair
            source_neurons = self.network.get_glomerulus_neurons(source_glomerulus)
            target_neurons = self.network.get_glomerulus_neurons(target_glomerulus)

            for src in source_neurons[:5]:  # Sample for efficiency
                for tgt in target_neurons[:5]:
                    if src == tgt:
                        continue
                    try:
                        length = nx.shortest_path_length(self.graph, src, tgt)
                        path_lengths.append(length)
                    except nx.NetworkXNoPath:
                        pass

        elif source_glomerulus:
            # From one glomerulus to all others
            source_neurons = self.network.get_glomerulus_neurons(source_glomerulus)

            for src in source_neurons[:5]:
                lengths = nx.single_source_shortest_path_length(self.graph, src)
                path_lengths.extend(lengths.values())

        else:
            # Global path lengths (sample for large networks)
            nodes = list(self.graph.nodes())[:100]  # Sample
            for node in nodes:
                try:
                    lengths = nx.single_source_shortest_path_length(self.graph, node)
                    path_lengths.extend(lengths.values())
                except:
                    pass

        if not path_lengths:
            return {
                'mean_path_length': 0,
                'median_path_length': 0,
                'max_path_length': 0,
                'num_paths': 0,
            }

        return {
            'mean_path_length': np.mean(path_lengths),
            'median_path_length': np.median(path_lengths),
            'max_path_length': np.max(path_lengths),
            'min_path_length': np.min(path_lengths),
            'std_path_length': np.std(path_lengths),
            'num_paths': len(path_lengths),
            'path_length_distribution': Counter(path_lengths),
        }

    def calculate_clustering_coefficients(
        self,
        level: str = 'glomerulus'
    ) -> Dict[str, float]:
        """
        Calculate clustering coefficients for nodes.

        Args:
            level: 'glomerulus' or 'neuron'

        Returns:
            Dictionary mapping nodes to clustering coefficients

        Example:
            >>> stats = NetworkStatistics(network)
            >>> clustering = stats.calculate_clustering_coefficients('glomerulus')
            >>> avg_clustering = np.mean(list(clustering.values()))
        """
        logger.info(f"Calculating clustering coefficients at {level} level...")

        if level == 'glomerulus':
            G = self.glomerulus_graph.to_undirected()
        else:
            G = self.graph.to_undirected()

        clustering = nx.clustering(G)

        return clustering

    def find_network_motifs(
        self,
        motif_size: int = 3
    ) -> Counter:
        """
        Find common network motifs (small subgraph patterns).

        Args:
            motif_size: Size of motifs to search for (3 or 4)

        Returns:
            Counter of motif patterns

        Example:
            >>> stats = NetworkStatistics(network)
            >>> motifs = stats.find_network_motifs(motif_size=3)
            >>> print(f"Found {len(motifs)} distinct motif types")
        """
        logger.info(f"Finding network motifs of size {motif_size}...")

        if motif_size != 3:
            logger.warning("Only 3-node motifs currently supported")
            motif_size = 3

        # Sample subgraphs
        motif_counter = Counter()

        nodes = list(self.graph.nodes())[:100]  # Sample for efficiency

        for i, node1 in enumerate(nodes):
            neighbors = list(self.graph.neighbors(node1))
            for j, node2 in enumerate(neighbors):
                for node3 in list(self.graph.neighbors(node2))[:5]:
                    if node3 == node1 or node3 in neighbors:
                        continue

                    # Extract subgraph
                    subgraph = self.graph.subgraph([node1, node2, node3])

                    # Classify motif type by number and direction of edges
                    num_edges = subgraph.number_of_edges()
                    motif_counter[f"{num_edges}_edges"] += 1

        return motif_counter

    def generate_full_report(self) -> str:
        """
        Generate comprehensive statistical report.

        Returns:
            Multi-line string with all statistics

        Example:
            >>> stats = NetworkStatistics(network)
            >>> report = stats.generate_full_report()
            >>> print(report)
        """
        lines = [
            "Network Statistical Analysis Report",
            "=" * 70,
            ""
        ]

        # Basic statistics
        basic_stats = self.network.get_network_statistics()
        lines.append("Basic Network Statistics:")
        lines.append(f"  Nodes: {basic_stats['num_nodes']:,}")
        lines.append(f"  Edges: {basic_stats['num_edges']:,}")
        lines.append(f"  Glomeruli: {basic_stats['num_glomeruli']}")
        lines.append("")

        # Hub neurons
        hubs_degree = self.detect_hub_neurons(method='degree', threshold_percentile=95)
        lines.append(f"Hub Neurons (top 5% by degree):")
        for i, (node, degree) in enumerate(hubs_degree[:10]):
            node_info = self.graph.nodes[node]
            cell_type = node_info.get('type', 'Unknown')
            lines.append(f"  {i+1}. {cell_type} (degree: {degree})")
        lines.append("")

        # Communities
        communities = self.detect_communities(level='glomerulus')
        num_communities = max(communities.values()) + 1
        lines.append(f"Community Detection (glomerulus level):")
        lines.append(f"  Number of communities: {num_communities}")

        # Count glomeruli per community
        comm_sizes = Counter(communities.values())
        lines.append("  Community sizes:")
        for comm_id, size in sorted(comm_sizes.items()):
            lines.append(f"    Community {comm_id}: {size} glomeruli")
        lines.append("")

        # Path lengths
        path_stats = self.analyze_path_lengths()
        lines.append("Path Length Statistics:")
        lines.append(f"  Mean path length: {path_stats['mean_path_length']:.2f}")
        lines.append(f"  Median path length: {path_stats['median_path_length']:.1f}")
        lines.append(f"  Max path length: {path_stats['max_path_length']}")
        lines.append("")

        # Clustering
        clustering = self.calculate_clustering_coefficients('glomerulus')
        if clustering:
            avg_clustering = np.mean(list(clustering.values()))
            lines.append("Clustering Coefficients (glomerulus level):")
            lines.append(f"  Average clustering: {avg_clustering:.4f}")
            lines.append("")

        # Asymmetry
        asym_matrix = self.calculate_asymmetry_matrix()
        if len(asym_matrix) > 0:
            lines.append("Asymmetry Analysis:")
            lines.append(f"  Mean asymmetry ratio: {asym_matrix['asymmetry_ratio'].mean():.3f}")
            lines.append(f"  Std asymmetry ratio: {asym_matrix['asymmetry_ratio'].std():.3f}")

            # Most asymmetric pairs
            most_asym = asym_matrix.nlargest(5, 'asymmetry_ratio')
            lines.append("  Most asymmetric connections (forward >> reverse):")
            for _, row in most_asym.iterrows():
                lines.append(
                    f"    {row['source_glomerulus']} → {row['target_glomerulus']}: "
                    f"{row['asymmetry_ratio']:.3f}"
                )

        return "\n".join(lines)


# Convenience functions
def analyze_network_statistics(network: CrossTalkNetwork) -> NetworkStatistics:
    """
    Create NetworkStatistics analyzer and generate report.

    Args:
        network: CrossTalkNetwork instance

    Returns:
        NetworkStatistics instance

    Example:
        >>> network = CrossTalkNetwork.from_csv('pathways.csv')
        >>> stats = analyze_network_statistics(network)
        >>> print(stats.generate_full_report())
    """
    return NetworkStatistics(network)
