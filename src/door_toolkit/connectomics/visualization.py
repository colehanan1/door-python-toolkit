"""
Visualization
=============

Publication-ready visualization tools for interglomerular cross-talk networks.

Features:
- Hierarchical neuron/glomerulus representation
- Customizable node colors, sizes, and layouts
- Edge thickness by synapse count
- Multiple export formats (PNG, PDF, SVG)
- Interactive and static plots
- Heatmaps for glomerulus-level connectivity
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set
import logging

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from door_toolkit.connectomics.network_builder import CrossTalkNetwork
from door_toolkit.connectomics.config import NetworkConfig

logger = logging.getLogger(__name__)


class NetworkVisualizer:
    """
    Visualization engine for cross-talk networks.

    Handles all plotting, layout, and export functionality with
    publication-ready defaults.
    """

    def __init__(
        self,
        network: CrossTalkNetwork,
        config: Optional[NetworkConfig] = None
    ):
        """
        Initialize visualizer.

        Args:
            network: CrossTalkNetwork instance to visualize
            config: NetworkConfig for visualization parameters
        """
        self.network = network
        self.config = config or network.config

        # Color schemes
        self.color_schemes = {
            'ORN': '#4A90E2',           # Blue
            'Local_Neuron': '#E74C3C',  # Red (inhibitory)
            'Projection_Neuron': '#2ECC71',  # Green (excitatory)
            'Glomerulus': '#9B59B6',    # Purple
        }

    def plot_full_network(
        self,
        output_path: Optional[Union[str, Path]] = None,
        show_glomeruli: bool = True,
        show_individual_neurons: bool = False,
        layout: str = 'spring',
        figsize: Tuple[float, float] = (16, 12),
        dpi: int = 300,
        min_synapse_display: int = 1,
    ) -> None:
        """
        Plot the complete network.

        Args:
            output_path: File path to save (None = display only)
            show_glomeruli: Show glomerulus meta-nodes
            show_individual_neurons: Show individual neurons
            layout: Layout algorithm ('spring', 'kamada_kawai', 'circular', 'hierarchical')
            figsize: Figure size in inches
            dpi: Resolution for raster outputs
            min_synapse_display: Minimum synapse count to display edge
        """
        logger.info("Plotting full network...")

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Filter graph by synapse threshold
        G = self._filter_graph_by_synapses(
            self.network.graph.copy(),
            min_synapse_display
        )

        # Optionally filter nodes
        if not show_individual_neurons:
            # Remove individual neurons, keep only glomeruli
            nodes_to_remove = [
                n for n, d in G.nodes(data=True)
                if d.get('node_type') == 'neuron'
            ]
            G.remove_nodes_from(nodes_to_remove)
        elif not show_glomeruli:
            # Remove glomerulus meta-nodes
            nodes_to_remove = [
                n for n, d in G.nodes(data=True)
                if d.get('node_type') == 'glomerulus_meta'
            ]
            G.remove_nodes_from(nodes_to_remove)

        # Calculate layout
        pos = self._calculate_layout(G, layout)

        # Prepare node attributes
        node_colors = []
        node_sizes = []
        for node in G.nodes():
            category = G.nodes[node].get('category', 'Unknown')
            node_colors.append(self.color_schemes.get(category, '#95A5A6'))

            # Size by node type
            if G.nodes[node].get('node_type') == 'glomerulus_meta':
                node_sizes.append(300)
            else:
                node_sizes.append(50)

        # Prepare edge attributes
        edge_widths = []
        edge_colors = []
        for u, v, data in G.edges(data=True):
            # Width proportional to synapse count
            weight = data.get('synapse_count', 1)
            edge_widths.append(np.log1p(weight) * 0.5)

            # Color by presynaptic neuron type
            presynaptic_category = G.nodes[u].get('category', '')
            if presynaptic_category == 'Local_Neuron':
                edge_colors.append('#E74C3C')  # Red for inhibitory
            else:
                edge_colors.append('#2ECC71')  # Green for excitatory

        # Draw network
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.7,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            width=edge_widths,
            edge_color=edge_colors,
            alpha=0.4,
            arrows=True,
            arrowsize=10,
            arrowstyle='->',
            ax=ax,
            connectionstyle='arc3,rad=0.1'
        )

        # Add labels for glomeruli only
        if show_glomeruli:
            glom_labels = {
                n: G.nodes[n].get('glomerulus', '')
                for n in G.nodes()
                if G.nodes[n].get('node_type') == 'glomerulus_meta'
            }
            nx.draw_networkx_labels(
                G, pos,
                labels=glom_labels,
                font_size=8,
                font_weight='bold',
                ax=ax
            )

        # Create legend
        legend_elements = [
            mpatches.Patch(color=self.color_schemes['ORN'], label='ORNs'),
            mpatches.Patch(color=self.color_schemes['Local_Neuron'], label='Local Neurons (inhibitory)'),
            mpatches.Patch(color=self.color_schemes['Projection_Neuron'], label='Projection Neurons (excitatory)'),
        ]
        if show_glomeruli:
            legend_elements.append(
                mpatches.Patch(color=self.color_schemes['Glomerulus'], label='Glomeruli')
            )

        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        ax.set_title(
            f'Interglomerular Cross-Talk Network\n'
            f'{G.number_of_nodes()} nodes, {G.number_of_edges()} edges',
            fontsize=14,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved network plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_single_orn_pathways(
        self,
        orn_identifier: str,
        by_glomerulus: bool = True,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (14, 10),
        dpi: int = 300,
    ) -> None:
        """
        Visualize pathways from a single ORN/glomerulus.

        Args:
            orn_identifier: ORN root_id or glomerulus name
            by_glomerulus: If True, show all neurons in glomerulus
            output_path: File path to save
            figsize: Figure size
            dpi: Resolution
        """
        logger.info(f"Plotting pathways for {orn_identifier}...")

        from door_toolkit.connectomics.pathway_analysis import analyze_single_orn

        analysis = analyze_single_orn(self.network, orn_identifier, by_glomerulus)

        if analysis.num_pathways == 0:
            logger.warning(f"No pathways found for {orn_identifier}")
            return

        # Build subgraph
        G = nx.DiGraph()

        # Add source node(s)
        if by_glomerulus:
            source_nodes = self.network.get_glomerulus_neurons(orn_identifier)
            # Add glomerulus meta-node
            G.add_node(f"GLOM_{orn_identifier}", node_type='glomerulus', category='Glomerulus')
        else:
            source_nodes = [orn_identifier]
            G.add_node(orn_identifier, node_type='neuron', category='ORN')

        # Add all pathway nodes and edges
        for pathway in analysis.pathways:
            # Add intermediate neuron
            G.add_node(
                pathway['level1_id'],
                node_type='neuron',
                category=pathway['level1_category'],
                cell_type=pathway['level1_type']
            )

            # Add target neuron
            G.add_node(
                pathway['level2_id'],
                node_type='neuron',
                category=pathway['level2_category'],
                cell_type=pathway['level2_type']
            )

            # Add edges
            if by_glomerulus:
                G.add_edge(
                    f"GLOM_{orn_identifier}",
                    pathway['level1_id'],
                    weight=pathway['synapse_count_step1']
                )
            else:
                G.add_edge(
                    pathway['orn_id'],
                    pathway['level1_id'],
                    weight=pathway['synapse_count_step1']
                )

            G.add_edge(
                pathway['level1_id'],
                pathway['level2_id'],
                weight=pathway['synapse_count_step2']
            )

        # Create hierarchical layout
        pos = self._hierarchical_layout(G, orn_identifier, by_glomerulus)

        # Plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Node colors and sizes
        node_colors = [
            self.color_schemes.get(G.nodes[n].get('category', 'Unknown'), '#95A5A6')
            for n in G.nodes()
        ]
        node_sizes = [
            400 if G.nodes[n].get('node_type') == 'glomerulus' else 200
            for n in G.nodes()
        ]

        # Edge colors and widths
        edge_colors = []
        edge_widths = []
        for u, v, data in G.edges(data=True):
            presynaptic_category = G.nodes[u].get('category', '')
            if presynaptic_category == 'Local_Neuron':
                edge_colors.append('#E74C3C')
            else:
                edge_colors.append('#2ECC71')

            edge_widths.append(np.log1p(data['weight']) * 0.5)

        # Draw
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=node_sizes,
            alpha=0.8,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            edge_color=edge_colors,
            width=edge_widths,
            alpha=0.6,
            arrows=True,
            arrowsize=15,
            arrowstyle='->',
            ax=ax
        )

        # Labels
        labels = {}
        for n in G.nodes():
            if G.nodes[n].get('node_type') == 'glomerulus':
                labels[n] = orn_identifier
            else:
                cell_type = G.nodes[n].get('cell_type', n)
                # Truncate long names
                if len(cell_type) > 15:
                    cell_type = cell_type[:12] + '...'
                labels[n] = cell_type

        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

        # Legend and title
        legend_elements = [
            mpatches.Patch(color=self.color_schemes['ORN'], label='Source ORN'),
            mpatches.Patch(color=self.color_schemes['Local_Neuron'], label='Local Neurons'),
            mpatches.Patch(color=self.color_schemes['Projection_Neuron'], label='Projection Neurons'),
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(
            f'Pathways from {orn_identifier}\n'
            f'{analysis.num_pathways} pathways, '
            f'{analysis.num_intermediate_neurons} intermediate neurons',
            fontsize=12,
            fontweight='bold'
        )
        ax.axis('off')

        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved pathway plot to {output_path}")
        else:
            plt.show()

        plt.close()

    def plot_glomerulus_heatmap(
        self,
        output_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[float, float] = (12, 10),
        dpi: int = 300,
        cmap: str = 'YlOrRd',
        log_scale: bool = True,
    ) -> None:
        """
        Plot heatmap of glomerulus-to-glomerulus connectivity.

        Args:
            output_path: File path to save
            figsize: Figure size
            dpi: Resolution
            cmap: Colormap name
            log_scale: Use log scale for colors
        """
        if not HAS_SEABORN:
            logger.warning("Seaborn not available, skipping heatmap")
            return

        logger.info("Plotting glomerulus connectivity heatmap...")

        # Build connectivity matrix
        glomeruli = sorted(self.network.data.glomeruli)
        n_glom = len(glomeruli)
        matrix = np.zeros((n_glom, n_glom))

        glom_to_idx = {g: i for i, g in enumerate(glomeruli)}

        for _, pathway in self.network.data.pathways.iterrows():
            source_glom = pathway['orn_glomerulus']
            target_id = pathway['level2_root_id']
            target_glom = self.network.get_neuron_glomerulus(target_id)

            if target_glom and target_glom in glom_to_idx:
                i = glom_to_idx[source_glom]
                j = glom_to_idx[target_glom]
                matrix[i, j] += pathway['synapse_count_step2']

        # Apply log scale if requested
        if log_scale:
            matrix = np.log1p(matrix)

        # Plot
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        sns.heatmap(
            matrix,
            xticklabels=glomeruli,
            yticklabels=glomeruli,
            cmap=cmap,
            ax=ax,
            cbar_kws={'label': 'Log(Synapse Count + 1)' if log_scale else 'Synapse Count'},
            square=True,
        )

        ax.set_title(
            'Interglomerular Cross-Talk Strength Matrix',
            fontsize=14,
            fontweight='bold'
        )
        ax.set_xlabel('Target Glomerulus', fontsize=12)
        ax.set_ylabel('Source Glomerulus', fontsize=12)

        plt.xticks(rotation=45, ha='right', fontsize=8)
        plt.yticks(rotation=0, fontsize=8)
        plt.tight_layout()

        if output_path:
            output_path = Path(output_path)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Saved heatmap to {output_path}")
        else:
            plt.show()

        plt.close()

    def _filter_graph_by_synapses(
        self,
        G: nx.DiGraph,
        min_count: int
    ) -> nx.DiGraph:
        """Filter graph edges by minimum synapse count."""
        edges_to_remove = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get('synapse_count', 0) < min_count
        ]
        G.remove_edges_from(edges_to_remove)

        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))

        return G

    def _calculate_layout(
        self,
        G: nx.DiGraph,
        layout: str
    ) -> Dict:
        """Calculate node positions using specified layout algorithm."""
        if layout == 'spring':
            pos = nx.spring_layout(
                G,
                k=2.0,
                iterations=self.config.layout_iterations,
                seed=self.config.layout_seed
            )
        elif layout == 'kamada_kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'hierarchical':
            pos = nx.multipartite_layout(G)
        else:
            logger.warning(f"Unknown layout '{layout}', using spring")
            pos = nx.spring_layout(G)

        return pos

    def _hierarchical_layout(
        self,
        G: nx.DiGraph,
        source_identifier: str,
        by_glomerulus: bool
    ) -> Dict:
        """Create hierarchical layout for pathway visualization."""
        pos = {}

        # Layer 0: Source
        if by_glomerulus:
            source_node = f"GLOM_{source_identifier}"
        else:
            source_node = source_identifier
        pos[source_node] = (0, 2)

        # Layer 1: Intermediate neurons
        intermediates = [
            n for n in G.nodes()
            if n != source_node and
            G.nodes[n].get('category') in ['Local_Neuron', 'Projection_Neuron']
        ]

        n_inter = len(intermediates)
        for i, node in enumerate(intermediates):
            x = -2 + (4 * i / max(n_inter - 1, 1))
            pos[node] = (x, 1)

        # Layer 2: Targets
        targets = [
            n for n in G.nodes()
            if n not in intermediates and n != source_node
        ]

        n_targets = len(targets)
        for i, node in enumerate(targets):
            x = -2 + (4 * i / max(n_targets - 1, 1))
            pos[node] = (x, 0)

        return pos


# Convenience functions
def plot_network(
    network: CrossTalkNetwork,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Quick function to plot full network.

    Args:
        network: CrossTalkNetwork instance
        output_path: Save path
        **kwargs: Additional arguments for plot_full_network
    """
    viz = NetworkVisualizer(network)
    viz.plot_full_network(output_path=output_path, **kwargs)


def plot_orn_pathways(
    network: CrossTalkNetwork,
    orn_identifier: str,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Quick function to plot single ORN pathways.

    Args:
        network: CrossTalkNetwork instance
        orn_identifier: ORN/glomerulus identifier
        output_path: Save path
        **kwargs: Additional arguments for plot_single_orn_pathways
    """
    viz = NetworkVisualizer(network)
    viz.plot_single_orn_pathways(orn_identifier, output_path=output_path, **kwargs)


def plot_heatmap(
    network: CrossTalkNetwork,
    output_path: Optional[Union[str, Path]] = None,
    **kwargs
) -> None:
    """
    Quick function to plot glomerulus heatmap.

    Args:
        network: CrossTalkNetwork instance
        output_path: Save path
        **kwargs: Additional arguments for plot_glomerulus_heatmap
    """
    viz = NetworkVisualizer(network)
    viz.plot_glomerulus_heatmap(output_path=output_path, **kwargs)
