# Connectomics Module - Interglomerular Cross-Talk Analysis

**Version:** 0.1.0
**Author:** DoOR Python Toolkit Team
**License:** MIT

---

## 📚 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Four Analysis Modes](#four-analysis-modes)
- [Data Files](#data-files)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Biological Context](#biological-context)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## Overview

The **Connectomics Module** provides comprehensive tools for analyzing interglomerular cross-talk in the *Drosophila melanogaster* olfactory system using FlyWire connectome data. It enables researchers to:

- Build and analyze multi-layer neural networks (ORN → LN/PN → Target)
- Quantify lateral inhibition and feedforward pathways
- Detect hub neurons and functional communities
- Visualize connectivity patterns with publication-ready figures
- Export data in multiple formats for further analysis

**Key Innovation:** Hierarchical representation of individual neurons AND glomerulus meta-nodes, enabling analysis at multiple biological scales.

---

## Features

### Core Functionality

✅ **Network Construction**
- NetworkX-based directed graph representation
- Individual neurons as nodes with hierarchical glomerulus grouping
- Synapse-weighted edges
- Configurable filtering by connection strength

✅ **Four Analysis Modes**
1. **Single ORN Focus** - All pathways from one ORN/glomerulus
2. **ORN Pair Comparison** - Bidirectional cross-talk quantification
3. **Full Network View** - Global topology and statistics
4. **Pathway Search** - Find specific connections

✅ **Statistical Analyses**
- Hub neuron detection (degree, betweenness, closeness, eigenvector centrality)
- Community detection (Louvain, greedy modularity, label propagation)
- Asymmetry quantification
- Path length distributions
- Clustering coefficients

✅ **Visualization**
- Hierarchical network plots (neuron + glomerulus levels)
- Glomerulus connectivity heatmaps
- Single ORN pathway diagrams
- Publication-ready output (PNG, PDF, SVG)
- Export to Cytoscape (GraphML) and Gephi (GEXF)

✅ **Biophysical Parameters**
- Research-based neuron parameters (from Wilson, Olsen, Kazama labs)
- Dale's law enforcement (consistent neurotransmitter per neuron type)
- Synaptic time constants for ACh and GABA

---

## Installation

### Prerequisites

```bash
# Ensure door-python-toolkit is installed
pip install door-python-toolkit

# Or install from source with connectomics dependencies
pip install -e ".[flywire]"
```

### Required Dependencies

```
pandas>=1.5.0
numpy>=1.21.0
networkx>=2.8
matplotlib>=3.5.0
scipy>=1.9.0
```

### Optional Dependencies

```
seaborn>=0.11.0  # For heatmaps
python-louvain   # For Louvain community detection
```

---

## Quick Start

### 1. Load and Explore Network

```python
from door_toolkit.connectomics import CrossTalkNetwork

# Load from CSV file
network = CrossTalkNetwork.from_csv('interglomerular_crosstalk_pathways.csv')

# Print summary
print(network.summary())

# Set minimum synapse threshold
network.set_min_synapse_threshold(10)

# Get basic statistics
stats = network.get_network_statistics()
print(f"Nodes: {stats['num_nodes']}, Edges: {stats['num_edges']}")
```

### 2. Analyze a Single Glomerulus

```python
from door_toolkit.connectomics.pathway_analysis import analyze_single_orn
from door_toolkit.connectomics.visualization import plot_orn_pathways

# Analyze DL5 glomerulus (responds to cis-vaccenyl acetate)
results = analyze_single_orn(network, 'ORN_DL5', by_glomerulus=True)

print(results.summary())
print(f"Found {results.num_pathways} pathways")

# Visualize
plot_orn_pathways(network, 'ORN_DL5', output_path='DL5_pathways.png')

# Export to CSV
df = results.to_dataframe()
df.to_csv('DL5_pathways.csv', index=False)
```

### 3. Compare Two Glomeruli

```python
from door_toolkit.connectomics.pathway_analysis import compare_orn_pair

# Compare DL5 vs VA1v
comparison = compare_orn_pair(network, 'ORN_DL5', 'ORN_VA1v', by_glomerulus=True)

print(comparison.summary())
print(f"Asymmetry ratio: {comparison.get_asymmetry_ratio():.3f}")
print(f"Shared LNs: {len(comparison.shared_intermediates['LNs'])}")
```

### 4. Full Network Analysis

```python
from door_toolkit.connectomics.statistics import NetworkStatistics
from door_toolkit.connectomics.visualization import plot_network, plot_heatmap

# Create statistics analyzer
stats = NetworkStatistics(network)

# Detect hub neurons
hubs = stats.detect_hub_neurons(method='betweenness', threshold_percentile=95)
print(f"Found {len(hubs)} hub neurons")

# Detect communities
communities = stats.detect_communities(algorithm='louvain', level='glomerulus')
print(f"Found {max(communities.values()) + 1} communities")

# Generate report
report = stats.generate_full_report()
print(report)

# Visualize
plot_network(network, output_path='full_network.png')
plot_heatmap(network, output_path='heatmap.png')
```

---

## Four Analysis Modes

### Mode 1: Single ORN Focus

**Purpose:** Analyze all pathways originating from one ORN or glomerulus

**Use Cases:**
- Understanding lateral inhibition patterns
- Identifying affected target glomeruli
- Quantifying cross-talk strength from a source

**Example:**
```python
results = analyze_single_orn(network, 'ORN_DL5', by_glomerulus=True)

# Get strongest pathways
top_pathways = results.get_strongest_pathways(n=10)

# Get target glomeruli distribution
targets = results.get_targets_by_glomerulus()
```

**Output:**
- List of all pathways with synapse counts
- Intermediate neuron identification (LNs, PNs)
- Target neuron categorization
- Summary statistics

### Mode 2: ORN Pair Comparison

**Purpose:** Compare cross-talk between two ORNs/glomeruli

**Use Cases:**
- Quantifying mutual inhibition
- Testing odor mixture interaction hypotheses
- Identifying asymmetric connections

**Example:**
```python
comparison = compare_orn_pair(network, 'ORN_DL5', 'ORN_VA1v', by_glomerulus=True)

# Check bidirectionality
if comparison.has_bidirectional_crosstalk:
    print("Mutual inhibition exists")

# Quantify asymmetry
asymmetry = comparison.get_asymmetry_ratio()
if asymmetry > 0.2:
    print(f"ORN_DL5 → ORN_VA1v is STRONGER")
elif asymmetry < -0.2:
    print(f"ORN_VA1v → ORN_DL5 is STRONGER")
else:
    print("Relatively symmetric")

# Find shared intermediate neurons
shared_lns = comparison.shared_intermediates['LNs']
```

**Output:**
- Pathways in both directions
- Shared intermediate neurons
- Cross-talk strength metrics
- Asymmetry quantification

### Mode 3: Full Network View

**Purpose:** Analyze complete network topology and organization

**Use Cases:**
- Understanding global connectivity patterns
- Detecting functional modules
- Identifying key hub neurons
- Publication-ready network visualizations

**Example:**
```python
stats = NetworkStatistics(network)

# Hub detection
hub_lns = stats.detect_hub_neurons(
    method='betweenness',
    neuron_category='Local_Neuron',
    threshold_percentile=90
)

# Community detection
communities = stats.detect_communities(algorithm='louvain', level='glomerulus')

# Asymmetry analysis
asym_matrix = stats.calculate_asymmetry_matrix()

# Path length analysis
path_stats = stats.analyze_path_lengths()

# Generate comprehensive report
report = stats.generate_full_report()
```

**Output:**
- Network-wide statistics
- Hub neuron rankings
- Community assignments
- Asymmetry matrix
- Path length distributions
- Clustering coefficients

### Mode 4: Pathway Search

**Purpose:** Find specific pathways between neurons/glomeruli

**Use Cases:**
- Testing specific connectivity hypotheses
- Finding strongest pathways between targets
- Identifying intermediate neurons in specific connections
- Shortest path analysis

**Example:**
```python
from door_toolkit.connectomics.pathway_analysis import find_pathways

# Find all pathways
results = find_pathways(
    network,
    source='ORN_DL5',
    target='ORN_VA1v',
    by_glomerulus=True,
    max_pathways=None  # Get all pathways
)

print(f"Found {results['num_pathways']} pathways")
print(f"Total synapses: {results['statistics']['total_synapses']}")
print(f"Shortest path length: {results['statistics']['shortest_path_length']}")

# Get intermediate neurons
lns = results['intermediate_neurons']['LNs']
pns = results['intermediate_neurons']['PNs']

# Filter by synapse strength
import pandas as pd
df = pd.DataFrame(results['pathways'])
strong_pathways = df[df['synapse_count_step2'] >= 20]
```

**Output:**
- List of pathways with full details
- Intermediate neuron identification
- Statistics (total/mean/median synapses)
- Shortest path length

---

## Data Files

### Required Input Files

The module expects CSV files with FlyWire connectivity data:

1. **`interglomerular_crosstalk_pathways.csv`** (~109K rows)
   - Complete 2-level pathways: ORN → Intermediate → Target
   - Columns: `orn_root_id`, `orn_label`, `orn_glomerulus`, `level1_root_id`, `level1_cell_type`, `level1_category`, `level2_root_id`, `level2_cell_type`, `level2_category`, `synapse_count_step1`, `synapse_count_step2`

2. **`crosstalk_ORN_LN_ORN.csv`** (optional)
   - Lateral inhibition pathways only

3. **`crosstalk_ORN_LN_PN.csv`** (optional)
   - Feedforward inhibition pathways

4. **`crosstalk_ORN_PN_feedback.csv`** (optional)
   - PN feedback pathways

5. **`crosstalk_matrix_glomerulus.csv`** (optional)
   - Aggregated glomerulus→glomerulus connectivity

### Data Format Details

**Neuron Categories:**
- `ORN`: Olfactory Receptor Neurons (1,839 individual neurons across ~47 glomeruli)
- `Local_Neuron`: GABAergic inhibitory neurons (~452 unique)
- `Projection_Neuron`: Cholinergic excitatory neurons

**Root IDs:** FlyWire neuron identifiers (e.g., `720575940617207185`)

**Synapse Counts:**
- `synapse_count_step1`: ORN → Intermediate connection strength
- `synapse_count_step2`: Intermediate → Target connection strength

---

## API Reference

### CrossTalkNetwork

Main class for network construction and analysis.

```python
class CrossTalkNetwork:
    @classmethod
    def from_csv(filepath, config=None) -> CrossTalkNetwork

    def set_min_synapse_threshold(threshold: int) -> None
    def get_pathways_from_orn(orn_identifier, by_glomerulus=False) -> List[Dict]
    def get_pathways_between_orns(source, target, by_glomerulus=False) -> List[Dict]
    def find_shortest_paths(source, target, max_paths=10) -> List[List[str]]
    def get_hub_neurons(neuron_category=None, top_n=10) -> List[Tuple[str, int]]
    def get_network_statistics() -> Dict
    def summary() -> str

    def export_to_graphml(filepath) -> None
    def export_to_gexf(filepath) -> None
```

### NetworkConfig

Configuration for network construction.

```python
class NetworkConfig:
    min_synapse_threshold: int = 1
    include_orn_ln_orn: bool = True
    include_orn_ln_pn: bool = True
    include_orn_pn_feedback: bool = True
    weight_scaling_factor: float = 0.1
    simulation_time: float = 1000.0

    def set_min_synapse_threshold(threshold: int) -> None
    def set_pathway_filters(orn_ln_orn, orn_ln_pn, orn_pn_feedback) -> None
    def get_neuron_params(neuron_category: str) -> Dict
    def get_synapse_params(presynaptic_category: str) -> Dict
    def to_json(filepath) -> None
    @classmethod
    def from_json(filepath) -> NetworkConfig
```

### NetworkStatistics

Statistical analysis of networks.

```python
class NetworkStatistics:
    def __init__(network: CrossTalkNetwork)

    def detect_hub_neurons(method='degree', threshold_percentile=90.0) -> List[Tuple]
    def detect_communities(algorithm='louvain', level='glomerulus') -> Dict
    def calculate_asymmetry_matrix() -> pd.DataFrame
    def analyze_path_lengths(source_glomerulus=None) -> Dict
    def calculate_clustering_coefficients(level='glomerulus') -> Dict
    def generate_full_report() -> str
```

### Visualization Functions

```python
def plot_network(network, output_path=None, **kwargs) -> None
def plot_orn_pathways(network, orn_identifier, output_path=None, **kwargs) -> None
def plot_heatmap(network, output_path=None, **kwargs) -> None
```

### Analysis Functions

```python
def analyze_single_orn(network, orn_identifier, by_glomerulus=True) -> SingleORNAnalysis
def compare_orn_pair(network, orn1, orn2, by_glomerulus=True) -> ORNPairComparison
def find_pathways(network, source, target, by_glomerulus=False) -> Dict
```

---

## Examples

Complete example scripts are provided in `examples/connectomics/`:

- `example_1_single_orn_analysis.py` - Mode 1: Single ORN focus
- `example_2_orn_pair_comparison.py` - Mode 2: ORN pair comparison
- `example_3_full_network_analysis.py` - Mode 3: Full network view
- `example_4_pathway_search.py` - Mode 4: Pathway search

Run examples:
```bash
cd examples/connectomics
python example_1_single_orn_analysis.py
```

---

## Biological Context

### Drosophila Olfactory System

The antennal lobe (AL) is the first processing center for olfactory information in *Drosophila*:

1. **ORNs (Olfactory Receptor Neurons):** Express specific odorant receptors, converge by type into glomeruli
2. **Local Neurons (LNs):** GABAergic inhibitory interneurons mediating lateral inhibition
3. **Projection Neurons (PNs):** Cholinergic neurons carrying processed signals to higher brain centers

### Interglomerular Cross-Talk

**Lateral inhibition** via LNs enables:
- Contrast enhancement
- Gain control
- Decorrelation of overlapping odor representations
- Background subtraction in odor mixtures

**Key mechanisms:**
- **ORN → LN → ORN:** Lateral inhibition between glomeruli
- **ORN → LN → PN:** Feedforward inhibition to PNs
- **ORN → PN → LN/ORN:** Feedback loops

### Research Applications

This toolkit enables testing hypotheses about:
- Odor mixture interactions (masking, synergy, suppression)
- Sparse coding mechanisms
- Functional organization of the AL
- Evolution of connectivity patterns
- Designing blocking experiments (optogenetics, RNAi)

---

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: File not found: interglomerular_crosstalk_pathways.csv`
**Solution:** Ensure data files are in the correct location or provide full path:
```python
network = CrossTalkNetwork.from_csv('/full/path/to/data.csv')
```

**Issue:** `MemoryError` when loading large files
**Solution:** Increase synapse threshold to reduce network size:
```python
config = NetworkConfig()
config.min_synapse_threshold = 20  # Only strong connections
network = CrossTalkNetwork.from_csv(filepath, config)
```

**Issue:** Visualization is cluttered
**Solution:** Filter by synapse strength:
```python
plot_network(network, min_synapse_display=50, show_individual_neurons=False)
```

**Issue:** Community detection fails
**Solution:** Install python-louvain:
```bash
pip install python-louvain
```

**Issue:** Heatmap not showing
**Solution:** Install seaborn:
```bash
pip install seaborn
```

---

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{door_python_toolkit_connectomics,
  title = {DoOR Python Toolkit - Connectomics Module},
  author = {Cole Hanan and Contributors},
  year = {2024},
  url = {https://github.com/colehanan1/door-python-toolkit},
  version = {0.2.0}
}
```

**Data Source:**
FlyWire Consortium. (2024). FlyWire: online community for whole-brain connectomics. *Nature*.

**Relevant Publications:**
- Wilson & Laurent (2005). Role of GABAergic inhibition in shaping odor-evoked spatiotemporal patterns in the Drosophila antennal lobe. *Journal of Neuroscience*.
- Olsen & Wilson (2008). Lateral presynaptic inhibition mediates gain control in olfactory glomeruli. *Nature*.
- Kazama & Wilson (2009). Origins of correlated activity in an olfactory circuit. *Nature Neuroscience*.

---

## Support

- **Issues:** https://github.com/colehanan1/door-python-toolkit/issues
- **Documentation:** https://door-python-toolkit.readthedocs.io
- **Examples:** `examples/connectomics/`

---

**Happy analyzing! 🧠✨**
