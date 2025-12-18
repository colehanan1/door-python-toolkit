[![PyPI version](https://badge.fury.io/py/door-python-toolkit.svg)](https://badge.fury.io/py/door-python-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

# DoOR Python Toolkit

**Comprehensive Python toolkit for Drosophila olfactory research: DoOR database integration, FlyWire connectomics, pathway analysis, and neural network preprocessing.**

Extract, analyze, and integrate *Drosophila melanogaster* odorant-receptor response data with connectome analysis. No R installation required.

---

## 🚀 Features

**NEW in v1.0.0:** Complete mushroom body circuit validation with ORN→PN→KC→MBON pathway tracing! 🎉

### Core DoOR Integration
- ✅ **Pure Python** - Extract DoOR R data files without installing R
- 🚀 **Fast** - Parquet-based caching for quick loading
- 📊 **693 odorants × 78 receptors** - Comprehensive olfactory data
- 🔍 **Search & Filter** - Query by odorant name, receptor, or properties

### FlyWire Connectomics
- 🧠 **Interglomerular Cross-Talk** - Analyze lateral inhibition pathways
- 🔬 **NetworkX Graphs** - 108,980+ pathways across 38 glomeruli
- 📈 **Statistical Analysis** - Hub detection, community detection, asymmetry
- 🎨 **Publication-Ready Figures** - High-resolution network visualizations

### Mushroom Body Circuit Validation
- 🎯 **ORN → PN → KC → MBON Tracing** - Complete learning circuit pathways
- 🧬 **Anatomical Validation** - Validate LASSO-identified receptors in MB circuits
- 🏆 **Priority Ranking** - Integrate behavioral importance with connectivity
- 📊 **Circuit Classification** - Appetitive (α/β) vs Aversive (γ) lobe specialization
- 🔬 **Experimental Design** - Generate priority matrices for optogenetic validation

### Advanced Features
- 🗺️ **FlyWire Integration** - Map receptors to neural connectivity (100K+ cells)
- 🛤️ **Pathway Analysis** - Trace Or47b, Or42b, Or92a pathways
- 🤖 **ML-Ready** - PyTorch/NumPy integration with sparse encoding
- 🧪 **Experiment Design** - PGCN blocking protocol generation
- 🎓 **LASSO Behavioral Prediction** - Identify sparse receptor circuits from optogenetic data

---

## 📦 Quick Start

### Installation

```bash
# Core package
pip install door-python-toolkit

# With all features
pip install door-python-toolkit[all]

# Individual feature sets
pip install door-python-toolkit[flywire]      # FlyWire integration
pip install door-python-toolkit[connectomics] # Connectomics module
pip install door-python-toolkit[torch]        # PyTorch support
pip install door-python-toolkit[extract]      # DoOR extraction
```

### Basic Usage

```python
from door_toolkit import DoOREncoder

# Load encoder
encoder = DoOREncoder("door_cache")

# Encode single odorant → 78-dim PN activation vector
pn_activation = encoder.encode("acetic acid")
print(pn_activation.shape)  # (78,)

# Search odorants
acetates = encoder.list_available_odorants(pattern="acetate")
print(f"Found {len(acetates)} acetates")  # 36
```

### Connectomics Analysis

```python
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import analyze_single_orn

# Load network
network = CrossTalkNetwork.from_csv('interglomerular_crosstalk_pathways.csv')
network.set_min_synapse_threshold(10)

# Analyze DL5 glomerulus
results = analyze_single_orn(network, 'ORN_DL5', by_glomerulus=True)
print(f"Found {results.num_pathways} cross-talk pathways")
```

---

## 📚 Table of Contents

- [Installation](#installation)
- [Core DoOR Features](#core-door-features)
- [Connectomics Module](#connectomics-module)
- [FlyWire Integration](#flywire-integration)
- [Mushroom Body Circuit Validation](#mushroom-body-circuit-validation)
- [Pathway Analysis](#pathway-analysis)
- [Neural Network Preprocessing](#neural-network-preprocessing)
- [Command-Line Interface](#command-line-interface)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

---

## Core DoOR Features

### What is DoOR?

The **Database of Odorant Responses (DoOR)** is a comprehensive collection of odorant-receptor response measurements for *Drosophila melanogaster*.

**Published:** Münch & Galizia (2016), *Scientific Data* 3:160122
**Citation:** https://doi.org/10.1038/sdata.2016.122

### Dataset Overview

| Metric | Value |
|--------|-------|
| Odorants | 693 compounds |
| Receptors | 78 ORN types (Or, Ir, Gr) |
| Measurements | 7,381 odorant-receptor pairs |
| Sparsity | 86% (typical for chemical screens) |
| Response Range | [0, 1] normalized |

### Extract DoOR Data

```python
from door_toolkit import DoORExtractor

# Extract R data files to Python formats
extractor = DoORExtractor(
    input_dir="path/to/DoOR.data/data",  # Unzipped DoOR R package
    output_dir="door_cache"
)
extractor.run()
```

### Use in Your Code

```python
from door_toolkit import DoOREncoder

# Load encoder
encoder = DoOREncoder("door_cache")

# Encode batch
odors = ["acetic acid", "1-pentanol", "ethyl acetate"]
pn_batch = encoder.batch_encode(odors)
print(pn_batch.shape)  # (3, 78)

# Get metadata
stats = encoder.get_receptor_coverage("acetic acid")
print(f"Active receptors: {stats['n_active']}")
```

---

## Connectomics Module

Comprehensive tools for analyzing interglomerular cross-talk in the *Drosophila* olfactory system using FlyWire connectome data.

### Key Features

✅ **Network Construction**
- NetworkX-based directed graph (108,980+ pathways)
- Hierarchical representation: individual neurons + glomerulus meta-nodes
- 2,828 neurons across 38 glomeruli
- Synapse-weighted edges with configurable thresholds

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

✅ **Biophysical Parameters**
- Research-based parameters (Wilson, Olsen, Kazama labs)
- Dale's law enforcement
- Synaptic time constants for ACh and GABA

### Quick Example

```python
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import analyze_single_orn, compare_orn_pair
from door_toolkit.connectomics.statistics import NetworkStatistics
from door_toolkit.connectomics.visualization import NetworkVisualizer

# Load network
network = CrossTalkNetwork.from_csv('interglomerular_crosstalk_pathways.csv')
network.set_min_synapse_threshold(10)

# Mode 1: Analyze single glomerulus
results = analyze_single_orn(network, 'ORN_DL5', by_glomerulus=True)
print(f"Found {results.num_pathways} pathways from DL5")

# Mode 2: Compare two glomeruli
comparison = compare_orn_pair(network, 'ORN_DL5', 'ORN_VA1v', by_glomerulus=True)
print(f"Asymmetry ratio: {comparison.get_asymmetry_ratio():.3f}")

# Mode 3: Full network analysis
stats = NetworkStatistics(network)
hubs = stats.detect_hub_neurons(method='betweenness', threshold_percentile=95)
communities = stats.detect_communities(algorithm='louvain', level='glomerulus')
print(f"Found {len(hubs)} hub neurons, {max(communities.values()) + 1} communities")

# Mode 4: Pathway search
from door_toolkit.connectomics.pathway_analysis import find_pathways
pathways = find_pathways(network, 'ORN_VM7v', 'ORN_D', by_glomerulus=True)
print(f"Found {pathways['num_pathways']} pathways")

# Visualization
visualizer = NetworkVisualizer(network)
visualizer.plot_full_network(output_path='network.png', min_synapse_display=50)
visualizer.plot_single_orn_pathways('ORN_DL5', output_path='DL5_pathways.png')
visualizer.plot_glomerulus_heatmap(output_path='heatmap.png')
```

### Biological Context

The antennal lobe processes olfactory information through:
1. **ORNs** - Express specific odorant receptors, converge into glomeruli
2. **Local Neurons (LNs)** - GABAergic inhibitory neurons mediating lateral inhibition
3. **Projection Neurons (PNs)** - Cholinergic neurons to higher brain centers

**Lateral inhibition** mechanisms:
- **ORN → LN → ORN**: Lateral inhibition between glomeruli (52% of pathways, median 3 synapses)
- **ORN → LN → PN**: Feedforward inhibition to PNs (16% of pathways)
- **ORN → PN → feedback**: Feedback loops (20% of pathways, up to 1,018 synapses)

### Key Discoveries

Our analysis revealed:
- **Hub LNs**: lLN2T_c, lLN2X04, lLN8, LN60b (prime optogenetic targets)
- **15 functional communities** with one major 22-glomerulus cluster
- **VM7v acts as convergence hub** receiving from multiple glomeruli
- **Asymmetric connectivity** patterns suggesting specialized functions

### ORN/Glomerulus Identifier Resolution

The connectomics module includes a **robust identifier resolution system** that automatically normalizes messy ORN/glomerulus names and maps receptor names to their glomerulus names.

**Key features:**
- **Format-agnostic**: Accepts `"DL3"`, `"dl3"`, `"ORN_DL3"`, `"ORN-DL3"`, `"Glomerulus DL3"` - all resolve to `"ORN_DL3"`
- **Receptor-to-glomerulus mapping**: Automatically maps `"Or7a"` → `"ORN_DL5"`, `"Ir31a"` → `"ORN_VL2p"`, `"Gr21a"` → `"ORN_V"`
- **Complete coverage**: Includes 44 receptors (33 Or, 10 Ir, 1 Gr) mapped to their FlyWire glomeruli
- **Fuzzy matching**: Suggests alternatives when exact matches fail (ranked by similarity)
- **Clear errors**: Provides actionable error messages with top 10 suggestions

In FlyWire, neurons are labeled by glomerulus name (e.g., `ORN_VL2p; Ir31a`), not receptor name. The resolver automatically handles this translation so you can use familiar receptor names like `"Ir31a"` or `"Or7a"` in your code. The system uses normalization (case-insensitive, separator-agnostic) combined with receptor mapping and fuzzy matching to prevent "non-matching ORN name" errors. All pathway analysis functions (`analyze_single_orn`, `compare_orn_pair`, `find_pathways`) accept both receptor names and glomerulus names. See [`examples/connectomics/example_orn_identifier_resolution.py`](examples/connectomics/example_orn_identifier_resolution.py) for a complete demonstration.

---

## FlyWire Integration

Map DoOR receptor data to FlyWire neural connectivity and community labels.

### Key Capabilities

- Parse 100K+ FlyWire community labels efficiently
- Map DoOR receptors to FlyWire root IDs
- Generate 3D spatial activation maps
- Export mappings in JSON/CSV formats

#### Namespace Translation & Diagnostics

- `DoORFlyWireIntegrator.get_connectivity_matrix_door_indexed()` translates FlyWire glomerulus labels (e.g., `ORN_DL5`) into DoOR receptor names (`Or7a`) so tuning and connectivity matrices share the same index before statistical analysis.
- `scripts/analysis_1_tuning_vs_connectivity.py` now logs detailed overlap diagnostics and generates a diagnostic report if insufficient overlapping receptors are found, making namespace issues easy to detect.

### Python API

```python
from door_toolkit.flywire import FlyWireMapper

# Initialize mapper
mapper = FlyWireMapper(
    community_labels_path="processed_labels.csv.gz",
    door_cache_path="door_cache",
    auto_parse=True
)

# Find cells expressing specific receptor
or42b_cells = mapper.find_receptor_cells("Or42b")
print(f"Found {len(or42b_cells)} Or42b neurons")

# Map all receptors
mappings = mapper.map_door_to_flywire()
print(f"Mapped {len(mappings)} receptors")

# Create spatial activation map
spatial_map = mapper.create_spatial_activation_map("ethyl butyrate")
print(f"Active at {spatial_map.total_cells} locations")

# Export mappings
mapper.export_mapping("flywire_mapping.json", format="json")
```

### CLI Usage

```bash
# Map receptors to FlyWire
door-flywire --labels processed_labels.csv.gz --cache door_cache --map-receptors

# Find specific receptor
door-flywire --labels processed_labels.csv.gz --find-receptor Or42b

# Create spatial map
door-flywire --labels processed_labels.csv.gz --cache door_cache \
  --spatial-map "ethyl butyrate" --output spatial_map.json
```

---

## Mushroom Body Circuit Validation

**NEW!** Validate LASSO-identified receptors using complete FlyWire mushroom body pathways.

### The Challenge

You've identified important receptors using LASSO regression on behavioral data. But **do these receptors actually connect to the learning circuit?**

This module answers: *"Are my receptors anatomically positioned in the mushroom body (MB), and which should I test first?"*

### Complete Workflow

```
LASSO Behavioral Prediction → FlyWire Pathway Tracing → Priority Matrix → Optogenetics
         ↓                              ↓                      ↓                ↓
   Or67c (weight=0.126)      23 ORNs → 6 PNs → 341 KCs    Final Score: 0.920   TEST FIRST!
                                        56.7% γ lobe        Circuit: Aversive
```

### Key Features

✅ **Complete Pathway Tracing**
- Trace: **ORN → PN → KC → MBON**
- Synapse-level connectivity (5.3M connections)
- Cell type classification (137K neurons)
- Mushroom body compartments (α/β, γ, α'β' lobes)

✅ **Circuit Validation Metrics**
- **ORN→PN Strength**: % of ORN output reaching PNs (commitment to learning pathway)
- **KC Coverage**: % of Kenyon Cells contacted (breadth of MB access)
- **Lobe Specialization**: α/β (appetitive) vs γ (aversive) fraction
- **Circuit Score**: Composite 0-1 score for "in learning circuit"

✅ **Integration with Behavioral Data**
- Load LASSO regression results
- Combine behavioral importance + anatomical validation
- Generate experimental priority matrix
- Export publication-ready figures

✅ **Sensillum Mapping**
- Automatic mapping: ab2B → Or85a, ab3A → Or22a, ab1A → Or42b
- Translates sensillum labels to specific Or receptors

### Python API

```python
from door_toolkit.flywire import FlyWireMapper
from door_toolkit.flywire.mushroom_body_tracer import MushroomBodyTracer

# Step 1: Map receptors to FlyWire ORN neurons
mapper = FlyWireMapper("processed_labels.csv.gz", auto_parse=True)
or67c_cells = mapper.find_receptor_cells("Or67c")
print(f"Found {len(or67c_cells)} Or67c ORNs")

# Step 2: Initialize mushroom body tracer
tracer = MushroomBodyTracer(
    synapse_path="connections_princeton.csv.gz",
    cell_types_path="consolidated_cell_types.csv.gz"
)

# Step 3: Trace complete pathway (ORN → PN → KC → MBON)
pathway = tracer.trace_receptor_pathway(
    receptor_name="Or67c",
    orn_ids=[cell["root_id"] for cell in or67c_cells]
)

print(f"Pathway Summary:")
print(f"  ORNs: {pathway.n_orns}")
print(f"  PNs: {len(pathway.unique_pns)}")
print(f"  KCs: {len(pathway.unique_kcs)}")
print(f"  Synapses (ORN→PN): {pathway.total_orn_to_pn_synapses}")
print(f"  Synapses (PN→KC): {pathway.total_pn_to_kc_synapses}")
print(f"  KC compartments: {pathway.kc_compartments}")

# Step 4: Calculate connectivity metrics
metrics = tracer.calculate_connectivity_metrics(pathway)
print(f"\nConnectivity Metrics:")
print(f"  ORN→PN strength: {metrics.orn_to_pn_strength:.2%}")
print(f"  KC coverage: {metrics.kc_coverage:.2%}")
print(f"  α/β lobe (appetitive): {metrics.alpha_beta_fraction:.2%}")
print(f"  γ lobe (aversive): {metrics.gamma_fraction:.2%}")
print(f"  Circuit score: {metrics.circuit_score:.3f}")
print(f"  Circuit type: {metrics.to_dict()['circuit_type']}")

# Step 5: Export results
tracer.export_pathway_csv([pathway], "pathway_summary.csv")
tracer.export_metrics_csv([metrics], "connectivity_metrics.csv")
```

### Complete Analysis Pipeline

Run the complete workflow from LASSO results to experimental priorities:

```python
# Full pipeline: examples/advanced/flywire_mb_pathway_analysis.py
python examples/advanced/flywire_mb_pathway_analysis.py
```

**Output:**
```
Top 3 High-Priority Receptors:
1. Or67c  - Final Score: 0.920  (AVERSIVE, γ lobe)   → TEST FIRST ⭐⭐⭐
2. Or22b  - Final Score: 0.686  (APPETITIVE, α/β)   → TEST SECOND ⭐⭐
3. Or85a  - Final Score: 0.658  (APPETITIVE, α/β)   → TEST SECOND ⭐⭐

Files generated:
  ✓ final_priority_matrix.csv       - Ranked receptors with all metrics
  ✓ flywire_pathway_summaries.csv   - ORN→PN→KC pathway stats
  ✓ flywire_connectivity_metrics.csv - Circuit validation scores
  ✓ priority_scatter.png             - LASSO vs Connectivity plot
  ✓ priority_bar.png                 - Priority ranking visualization
```

### Example Results

**Or67c (Top Candidate)**:
```
LASSO Weight: 0.126 (HIGHEST)
Pathway: 23 ORNs → 6 PNs → 341 KCs
Circuit: 56.7% γ lobe (AVERSIVE learning)
Final Score: 0.920
Recommendation: TEST FIRST - Silencing will impair learned aversive responses
```

**Or85a (ab2B sensillum)**:
```
LASSO Weight: 0.067 (3rd highest)
Pathway: 42 ORNs → 5 PNs → 391 KCs
Circuit: 55.6% α/β lobe (APPETITIVE learning)
ORN→PN Strength: 84.2% (HIGHEST commitment!)
Final Score: 0.658
Recommendation: TEST SECOND - Strong appetitive circuit
```

### Biological Interpretation

**Circuit Types:**
- **Appetitive (α/β lobe)**: Reward/feeding learning (Or22b, Or85a, Or42b)
- **Aversive (γ lobe)**: Avoidance/punishment learning (Or67c, Or49a)

**Connectivity Metrics:**
- **High ORN→PN strength** (>70%): Strong commitment to learning pathway
- **High KC coverage** (>20%): Broad access to memory encoding
- **Lobe specialization** (>50%): Clear circuit type assignment
- **Circuit score** (>0.80): High confidence in MB circuit membership

### Integration with LASSO

```python
from door_toolkit.pathways import LassoBehavioralPredictor

# Step 1: Run LASSO to identify important receptors
predictor = LassoBehavioralPredictor(
    doorcache_path="door_cache",
    behavior_csv_path="reaction_rates_summary.csv"
)

# Fit models for different optogenetic conditions
results_hex = predictor.fit_behavior("opto_hex")
results_eb = predictor.fit_behavior("opto_EB")
results_benz = predictor.fit_behavior("opto_benz_1")

print(f"Or22b LASSO weight (hexanol): {results_hex.lasso_weights.get('Or22b', 0):.4f}")
print(f"Or67c LASSO weight (EB): {results_eb.lasso_weights.get('Or67c', 0):.4f}")
print(f"Or85a LASSO weight (benz): {results_benz.lasso_weights.get('Or85a', 0):.4f}")

# Step 2: Validate with FlyWire (see above)
# ...

# Step 3: Generate final priority matrix
# Combines: 60% behavioral importance + 40% circuit connectivity
```

### CLI Usage

```bash
# Run complete mushroom body analysis
python examples/advanced/flywire_mb_pathway_analysis.py

# Output: flywire_mb_analysis/
#   ├── final_priority_matrix.csv       # Experimental priorities
#   ├── flywire_pathway_summaries.csv   # Pathway statistics
#   ├── flywire_connectivity_metrics.csv # Circuit validation
#   ├── priority_scatter.png            # Visualization
#   ├── priority_bar.png                # Rankings
#   └── UPDATED_SUMMARY.md              # Complete report
```

### Real-World Example

**Research Question**: "Which receptors are critical for learned olfactory behavior?"

**Workflow**:
1. ✅ **LASSO identifies** Or67c, Or22b, Or85a as important (sparse circuit)
2. ✅ **FlyWire validates** all 3 reach mushroom body via PN→KC pathways
3. ✅ **Circuit analysis** reveals:
   - Or67c: 56.7% γ lobe → aversive learning
   - Or22b: 69.5% α/β lobe → appetitive learning
   - Or85a: 55.6% α/β lobe → appetitive learning
4. ✅ **Priority matrix** ranks Or67c #1 (score: 0.920)
5. ✅ **Optogenetic validation** confirms Or67c silencing impairs learning

**Result**: Anatomically validated, prioritized receptor list for experiments! 🎯

---

## Pathway Analysis

Quantitative analysis of olfactory pathways and experiment protocol generation.

### Key Capabilities

- Trace known pathways (Or47b→feeding, Or42b, Or92a→avoidance)
- Custom pathway analysis
- Shapley importance computation
- PGCN experiment protocol generation
- Behavioral prediction

### Python API

```python
from door_toolkit.pathways import PathwayAnalyzer, BlockingExperimentGenerator, BehavioralPredictor

# Pathway analysis
analyzer = PathwayAnalyzer("door_cache")

# Trace Or47b feeding pathway
pathway = analyzer.trace_or47b_feeding_pathway()
print(f"Pathway strength: {pathway.strength:.3f}")
print(f"Top receptors: {pathway.get_top_receptors(5)}")

# Custom pathway
custom = analyzer.trace_custom_pathway(
    receptors=["Or92a"],
    odorants=["geosmin"],
    behavior="avoidance"
)

# Shapley importance
importance = analyzer.compute_shapley_importance("feeding")
top_receptors = sorted(importance.items(), key=lambda x: -x[1])[:10]

# Generate experiment protocol
generator = BlockingExperimentGenerator("door_cache")
protocol = generator.generate_experiment_1_protocol()  # Single-unit veto
protocol.export_json("experiment_protocol.json")

# Behavioral prediction (heuristic)
predictor = BehavioralPredictor("door_cache")
prediction = predictor.predict_behavior("hexanol")
print(f"Valence: {prediction.predicted_valence}")
print(f"Confidence: {prediction.confidence:.2%}")

# LASSO behavioral prediction (data-driven)
from door_toolkit.pathways import LassoBehavioralPredictor

lasso_predictor = LassoBehavioralPredictor(
    doorcache_path="door_cache",
    behavior_csv_path="reaction_rates_summary.csv"
)

# Fit model for optogenetic condition
results = lasso_predictor.fit_behavior("opto_hex")
print(f"R² = {results.cv_r2_score:.3f}")
print(f"Selected {results.n_receptors_selected} receptors")

# Get top predictive receptors
for receptor, weight in results.get_top_receptors(5):
    print(f"  {receptor}: {weight:.4f}")

# Generate plots
results.plot_predictions(save_to="opto_hex_predictions.png")
results.plot_receptors(save_to="opto_hex_receptors.png")

# Export results
results.export_csv("opto_hex_results.csv")
results.export_json("opto_hex_model.json")

# Compare multiple conditions
comparison = lasso_predictor.compare_conditions(
    conditions=["opto_hex", "opto_EB", "opto_benz_1"],
    plot=True,
    save_dir="comparison_results"
)
```

### LASSO Behavioral Prediction

The `LassoBehavioralPredictor` uses sparse regression (LASSO) to identify minimal receptor circuits that predict behavioral responses from optogenetic manipulation experiments:

**Features:**
- Automatic odorant name matching between behavioral data and DoOR
- Cross-validated LASSO regression with automatic λ selection
- Sparse receptor circuit identification (typically 3-10 receptors)
- Multiple prediction modes: test odorant, trained odorant, or interaction features
- Visualization: predicted vs actual PER, receptor importance rankings
- Export to CSV/JSON for downstream analysis

**Workflow:**
1. Load optogenetic behavioral data (PER responses)
2. Match odorant names to DoOR receptor profiles
3. Fit LASSO models with cross-validation
4. Extract sparse receptor weights
5. Visualize and export results

**Example dataset format** (`reaction_rates_summary.csv`):
```
dataset,3-Octonol,Benzaldehyde,Ethyl_Butyrate,Hexanol,Linalool
opto_hex,0.25,0.00,0.19,0.69,0.19
opto_EB,0.13,0.00,0.22,0.20,0.00
opto_benz_1,0.25,0.02,0.44,0.59,0.12
```

**Biological Interpretation:**
- Positive weights → receptors associated with higher PER
- Negative weights → receptors associated with lower PER (potential inhibition)
- Zero weights → receptors excluded by LASSO (not predictive)
- Sparse circuits (3-7 receptors) suggest minimal testable hypotheses

### CLI Usage

```bash
# Trace pathways
door-pathways --cache door_cache --trace or47b-feeding

# Custom pathway
door-pathways --cache door_cache --custom-pathway \
  --receptors Or92a --odorants geosmin --behavior avoidance

# Shapley importance
door-pathways --cache door_cache --shapley feeding --output importance.json

# Generate experiment
door-pathways --cache door_cache --generate-experiment 1 \
  --output exp1_protocol.json --format markdown

# Predict behavior
door-pathways --cache door_cache --predict-behavior "ethyl butyrate"
```

---

## Neural Network Preprocessing

Prepare DoOR data for neural network training with sparse encoding and augmentation.

### Key Capabilities

- Sparse KC-like encoding (5% sparsity)
- Hill equation concentration-response modeling
- Noise augmentation (Gaussian, Poisson, dropout)
- PyTorch/NumPy/HDF5 export
- PGCN-compatible dataset generation

### Python API

```python
from door_toolkit.neural import DoORNeuralPreprocessor

# Initialize preprocessor
preprocessor = DoORNeuralPreprocessor(
    "door_cache",
    n_kc_neurons=2000,
    random_seed=42
)

# Create sparse encoding
sparse_data = preprocessor.create_sparse_encoding(sparsity_level=0.05)
print(f"Shape: {sparse_data.shape}")
print(f"Sparsity: {(sparse_data > 0).mean():.2%}")

# Generate augmented dataset
aug_orn, aug_kc, labels = preprocessor.generate_noise_augmented_responses(
    n_augmentations=5,
    noise_level=0.1
)

# Export PGCN dataset
preprocessor.export_pgcn_dataset(
    output_dir="pgcn_dataset",
    format="pytorch",  # or "numpy", "h5"
    include_sparse=True
)

# Train/val split
train, val = preprocessor.create_training_validation_split(train_fraction=0.8)
```

### Concentration-Response Modeling

```python
from door_toolkit.neural.concentration_models import ConcentrationResponseModel

model = ConcentrationResponseModel()

# Fit Hill equation
concentrations = np.array([0.001, 0.01, 0.1, 1.0])
responses = np.array([0.1, 0.3, 0.7, 0.9])
params = model.fit_hill_equation(concentrations, responses)

print(f"EC50: {params.ec50:.3f}")
print(f"Hill coefficient: {params.hill_coefficient:.3f}")

# Generate concentration series
conc, resp = model.generate_concentration_series(params, n_points=50)

# Model odor mixtures
mixture_responses = model.model_mixture_interactions(
    [params1, params2],
    concentrations,
    interaction_type="additive"
)
```

### CLI Usage

```bash
# Sparse encoding
door-neural --cache door_cache --sparse-encode --sparsity 0.05 \
  --output sparse_data.npy

# Augment dataset
door-neural --cache door_cache --augment --n-augmentations 5 \
  --output-dir augmented_data/

# Export PGCN dataset
door-neural --cache door_cache --export-pgcn \
  --output-dir pgcn_dataset/ --format pytorch

# Dataset statistics
door-neural --cache door_cache --stats
```

---

## Command-Line Interface

### Core Commands

```bash
# Extract DoOR data
door-extract --input DoOR.data/data --output door_cache

# Validate cache contents
door-extract --validate door_cache

# List odorants (optional substring filter)
door-extract --list-odorants door_cache --pattern acetate

# Encode an odorant and show receptor responses
door-extract --cache door_cache --odor "ethyl butyrate" --coverage

# Compare multiple odorants
door-extract --cache door_cache --odors "ethyl butyrate" "acetic acid" \
  --top 15 --coverage --save reports/odor-comparison

# Inspect receptor response profiles
door-extract --cache door_cache --receptor Or42b --top 25
```

### Feature-Specific Commands

```bash
# FlyWire integration
door-flywire --labels processed_labels.csv.gz --cache door_cache --map-receptors

# Pathway analysis
door-pathways --cache door_cache --trace or47b-feeding

# Neural preprocessing
door-neural --cache door_cache --sparse-encode --sparsity 0.05 --output sparse_data.npy
```

Add `--debug` to any command for detailed tracebacks and logging.

**Receptor group shortcuts:**
- `or` – Odorant receptors (OrXX)
- `ir` – Ionotropic receptors (IrXX)
- `gr` – Gustatory receptors (GrXX)
- `neuron` – Antennal/palp neuron classes (ab*, ac*, pb*)

---

## API Reference

### DoORExtractor
Extract DoOR R data files to Python formats.

```python
from door_toolkit import DoORExtractor

extractor = DoORExtractor(input_dir, output_dir)
extractor.run()
extractor.extract_response_matrix()
extractor.extract_odor_metadata()
```

### DoOREncoder
Encode odorant names to neural activation patterns.

```python
from door_toolkit import DoOREncoder

encoder = DoOREncoder(cache_path, use_torch=False)
encoder.encode(odor_name)
encoder.batch_encode(odor_names)
encoder.list_available_odorants(pattern)
encoder.get_receptor_coverage(odor_name)
encoder.get_odor_metadata(odor_name)
```

### CrossTalkNetwork
Main class for connectomics network analysis.

```python
from door_toolkit.connectomics import CrossTalkNetwork

network = CrossTalkNetwork.from_csv(filepath, config=None)
network.set_min_synapse_threshold(threshold)
network.get_pathways_from_orn(orn_identifier, by_glomerulus=False)
network.get_pathways_between_orns(source, target, by_glomerulus=False)
network.find_shortest_paths(source, target, max_paths=10)
network.get_hub_neurons(neuron_category=None, top_n=10)
network.get_network_statistics()
network.export_to_graphml(filepath)
network.export_to_gexf(filepath)
```

### NetworkStatistics
Statistical analysis of connectomics networks.

```python
from door_toolkit.connectomics.statistics import NetworkStatistics

stats = NetworkStatistics(network)
stats.detect_hub_neurons(method='degree', threshold_percentile=90.0)
stats.detect_communities(algorithm='louvain', level='glomerulus')
stats.calculate_asymmetry_matrix()
stats.analyze_path_lengths(source_glomerulus=None)
stats.generate_full_report()
```

### Analysis Functions

```python
from door_toolkit.connectomics.pathway_analysis import (
    analyze_single_orn,
    compare_orn_pair,
    find_pathways
)

# Mode 1: Single ORN
results = analyze_single_orn(network, orn_identifier, by_glomerulus=True)

# Mode 2: ORN pair comparison
comparison = compare_orn_pair(network, orn1, orn2, by_glomerulus=True)

# Mode 4: Pathway search
pathways = find_pathways(network, source, target, by_glomerulus=False)
```

### Visualization

```python
from door_toolkit.connectomics.visualization import NetworkVisualizer

visualizer = NetworkVisualizer(network)
visualizer.plot_full_network(output_path='network.png', **kwargs)
visualizer.plot_single_orn_pathways(orn_identifier, output_path='pathways.png')
visualizer.plot_glomerulus_heatmap(output_path='heatmap.png')
```

### MushroomBodyTracer

**NEW!** Trace complete pathways through mushroom body learning circuits.

```python
from door_toolkit.flywire.mushroom_body_tracer import MushroomBodyTracer

# Initialize tracer
tracer = MushroomBodyTracer(
    synapse_path="connections_princeton.csv.gz",
    cell_types_path="consolidated_cell_types.csv.gz",
    min_synapse_threshold=1
)

# Trace pathway: ORN → PN → KC → MBON
pathway = tracer.trace_receptor_pathway(receptor_name, orn_ids)

# Calculate connectivity metrics
metrics = tracer.calculate_connectivity_metrics(pathway, total_kcs_in_brain=2000)

# Export results
tracer.export_pathway_csv([pathway], "pathway_summary.csv")
tracer.export_metrics_csv([metrics], "connectivity_metrics.csv")
```

**Key Classes:**
- `PathwayStep`: Single synapse connection
- `MushroomBodyPathway`: Complete ORN→PN→KC pathway
- `ConnectivityMetrics`: Circuit validation scores

**Attributes:**
- `pathway.n_orns`: Number of ORN neurons
- `pathway.n_pns`: Number of PN neurons contacted
- `pathway.n_kcs`: Number of KC neurons contacted
- `pathway.kc_compartments`: Dict of KC counts by lobe (α/β, γ, α'β')
- `metrics.orn_to_pn_strength`: ORN→PN pathway strength (0-1)
- `metrics.kc_coverage`: Fraction of KCs contacted (0-1)
- `metrics.alpha_beta_fraction`: Fraction in appetitive lobe (0-1)
- `metrics.circuit_score`: Overall connectivity score (0-1)

### Mapping Accounting

**IMPORTANT:** Prevents confusion between receptor counts and unique glomerulus counts in many-to-one mappings.

```python
from door_toolkit.integration.mapping_accounting import (
    compute_mapping_stats,
    format_mapping_summary,
    log_mapping_stats,
    write_mapping_stats_json
)

# Compute comprehensive mapping statistics
mapping = {'OR82A': 'VA6', 'OR94A': 'VA6', 'OR7A': 'DL5'}  # Example with collision
stats = compute_mapping_stats(
    mapping,
    note="Example mapping",
    adult_only=False  # Include larval receptors
)

# Get compact summary
summary = format_mapping_summary(stats)
# "3 receptors → 2 unique glomeruli (1 collision)"

# Check for many-to-one collapses
if stats['collision_count'] > 0:
    print(f"Collisions: {stats['collision_summary']}")
    # ['VA6: OR82A, OR94A']

# Write JSON artifact for reproducibility
write_mapping_stats_json("mapping_stats.json", stats)
```

**Key Stats Returned:**
- `n_receptors_mapped`: Number of receptor genes successfully mapped
- `n_unique_glomeruli_from_mapped_receptors`: Number of distinct glomeruli (may differ!)
- `collision_count`: Number of glomeruli with ≥2 receptors (many-to-one)
- `collisions`: Dict of glomerulus → [receptor list] for collisions
- `collision_summary`: Human-readable collision descriptions

📚 **See:** [docs/RECEPTOR_GLOMERULUS_MAPPING_ACCOUNTING.md](docs/RECEPTOR_GLOMERULUS_MAPPING_ACCOUNTING.md) for complete documentation on preventing receptor vs glomerulus count confusion.

---

## Examples

Complete working examples are available in the `examples/` directory:

### Basic DoOR Examples
- `examples/basic/encode_odorants.py` - Encode odorants to PN activations
- `examples/basic/search_odorants.py` - Search and filter odorants
- `examples/basic/receptor_analysis.py` - Analyze receptor responses

### Connectomics Examples
- `examples/connectomics/example_1_single_orn_analysis.py` - Mode 1: Single ORN focus
- `examples/connectomics/example_2_orn_pair_comparison.py` - Mode 2: ORN pair comparison
- `examples/connectomics/example_3_full_network_analysis.py` - Mode 3: Full network view
- `examples/connectomics/example_4_pathway_search.py` - Mode 4: Pathway search
- `examples/connectomics/example_orn_identifier_resolution.py` - Robust identifier resolution demo
- `examples/connectomics/analyze_data_characteristics.py` - Data quality analysis

### Advanced Examples
- `examples/advanced/flywire_integration_example.py` - FlyWire mapping
- `examples/advanced/flywire_mb_pathway_analysis.py` - **NEW!** Mushroom body circuit validation
- `examples/advanced/pathway_analysis_example.py` - Pathway tracing
- `examples/advanced/neural_preprocessing_example.py` - Neural network prep
- `examples/lasso_behavioral_prediction_demo.py` - LASSO regression for behavioral prediction

### Running Examples

```bash
# Extract DoOR data first
door-extract --input DoOR.data/data --output door_cache

# Run examples
python examples/basic/encode_odorants.py
python examples/connectomics/example_1_single_orn_analysis.py
python examples/advanced/flywire_integration_example.py

# NEW: Mushroom body circuit validation
python examples/advanced/flywire_mb_pathway_analysis.py
```

### Complete Workflow Example

**From LASSO to Optogenetics**:

```bash
# 1. Run LASSO behavioral prediction
python examples/lasso_behavioral_prediction_demo.py

# 2. Validate receptors with FlyWire mushroom body analysis
python examples/advanced/flywire_mb_pathway_analysis.py

# Output:
#   behavioral_prediction_results/
#     ├── opto_hex_results.csv        # LASSO identified receptors
#     └── opto_hex_predictions.png
#
#   flywire_mb_analysis/
#     ├── final_priority_matrix.csv   # Experimental priorities
#     ├── priority_scatter.png
#     └── UPDATED_SUMMARY.md          # Complete analysis report

# 3. Use priority matrix to design optogenetic experiments!
```

---

## Requirements

### Core Dependencies
- Python ≥ 3.8
- pandas ≥ 1.5.0
- numpy ≥ 1.21.0
- pyarrow ≥ 12.0.0
- networkx ≥ 2.8
- matplotlib ≥ 3.5.0
- scipy ≥ 1.9.0

### Optional Dependencies
- **pyreadr ≥ 0.4.7** - Required for DoORExtractor
- **torch ≥ 2.0.0** - For PyTorch integration
- **seaborn ≥ 0.11.0** - For heatmaps
- **python-louvain ≥ 0.16** - For Louvain community detection
- **plotly ≥ 5.11.0** - For interactive visualizations
- **h5py ≥ 3.7.0** - For HDF5 export

---

## Installation from Source

```bash
# Clone repository
git clone https://github.com/yourusername/door-python-toolkit.git
cd door-python-toolkit

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install development dependencies
make install-dev

# Extract DoOR data
make extract INPUT=path/to/DoOR.data/data OUTPUT=door_cache

# Run tests
make test

# Lint and format
make lint
make format
```

---

## Data Sources

### DoOR Database
This toolkit extracts data from the original DoOR R packages:
- **DoOR.data** - https://github.com/ropensci/DoOR.data
- **DoOR.functions** - https://github.com/ropensci/DoOR.functions

Download DoOR data:
```bash
wget https://github.com/ropensci/DoOR.data/archive/refs/tags/v2.0.0.zip
unzip v2.0.0.zip
door-extract --input DoOR.data-2.0.0/data --output door_cache
```

### FlyWire Connectome
FlyWire connectome data is available from:
- **FlyWire** - https://flywire.ai/
- **Community labels** - Available through CAVE API

---

## Performance

- **DoOR extraction**: Full dataset in <10 seconds
- **FlyWire parsing**: 100K+ labels in <30 seconds
- **Network construction**: 108,980 pathways loaded in <5 seconds
- **Receptor mapping**: >80% success rate
- **Sparse encoding**: Maintains 5±1% sparsity
- **Memory usage**: <2GB for largest datasets

---

## Testing

Run the comprehensive test suite:

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=door_toolkit --cov-report=html

# Specific test modules
pytest tests/test_connectomics.py -v
pytest tests/test_encoder.py -v
```

---

## Receptor Mapping References

1. **Couto, A., et al. (2005)** "Molecular, Anatomical, and Functional Organization of the Drosophila Olfactory System." *Current Biology* 15(17): 1535-1547. DOI: 10.1016/j.cub.2005.07.034
2. **Hallem, E. A. & Carlson, J. R. (2006)** "Coding of Odors by a Receptor Repertoire." *Cell* 125(1): 143-160. DOI: 10.1016/j.cell.2006.01.050
3. **Silbering, A. F., et al. (2011)** "Complementary Function and Integrated Wiring of the Evolutionarily Distinct Drosophila Olfactory Subsystems." *Journal of Neuroscience* 31(38): 13357-13375. DOI: 10.1523/JNEUROSCI.2360-11.2011
4. **Fishilevich, E. & Vosshall, L. B. (2005)** "Genetic and Functional Subdivision of the Drosophila Antennal Lobe." *Current Biology* 15(17): 1548-1553. DOI: 10.1016/j.cub.2005.07.066
5. **Benton, R., et al. (2009)** "Variant Ionotropic Glutamate Receptors as Chemosensory Receptors in Drosophila." *Cell* 136(1): 149-162. DOI: 10.1016/j.cell.2008.12.001

## Citation

If you use this toolkit in your research, please cite:

### This Toolkit
```bibtex
@software{door_python_toolkit,
  author = {Hanan, Cole and Contributors},
  title = {DoOR Python Toolkit: Comprehensive Tools for Drosophila Olfactory Research},
  year = {2025},
  version = {1.0.0},
  url = {https://github.com/colehanan1/door-python-toolkit},
  note = {Production-ready toolkit with mushroom body circuit validation and LASSO behavioral prediction}
}
```

### Original DoOR Database
```bibtex
@article{muench2016door,
  title={DoOR 2.0--Comprehensive Mapping of Drosophila melanogaster Odorant Responses},
  author={M{\"u}nch, Daniel and Galizia, C Giovanni},
  journal={Scientific Data},
  volume={3},
  number={1},
  pages={1--14},
  year={2016},
  publisher={Nature Publishing Group}
}
```

### FlyWire Consortium
```bibtex
@article{flywire2024,
  title={FlyWire: online community for whole-brain connectomics},
  author={FlyWire Consortium and Others},
  journal={Nature},
  year={2024}
}
```

### Relevant Publications
- Wilson & Laurent (2005). Role of GABAergic inhibition in shaping odor-evoked spatiotemporal patterns in the Drosophila antennal lobe. *Journal of Neuroscience*.
- Olsen & Wilson (2008). Lateral presynaptic inhibition mediates gain control in olfactory glomeruli. *Nature*.
- Kazama & Wilson (2009). Origins of correlated activity in an olfactory circuit. *Nature Neuroscience*.

---

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Development setup:**
```bash
git clone https://github.com/yourusername/door-python-toolkit.git
cd door-python-toolkit
python -m venv .venv
source .venv/bin/activate
make install-dev
make test
```

**Code Style:**
- Follow PEP 8
- Use Black for formatting (`make format`)
- Add type hints
- Write docstrings for public APIs
- Add tests for new features

---

## Troubleshooting

### DoOR Issues

**"Odorant not found"**
→ Use `encoder.list_available_odorants()` to see exact names (case-insensitive)

**"Cache not found"**
→ Run `DoORExtractor` first to extract R data files

**"High sparsity"**
→ Normal for DoOR (86%). Use `fillna(0.0)` or filter to well-covered receptors

**PyTorch not available**
→ Install with `pip install door-python-toolkit[torch]`

### Connectomics Issues

**`FileNotFoundError: interglomerular_crosstalk_pathways.csv`**
→ Ensure data files are in correct location or provide full path

**`MemoryError` when loading large files**
→ Increase synapse threshold to reduce network size:
```python
network.set_min_synapse_threshold(20)  # Only strong connections
```

**Visualization is cluttered**
→ Filter by synapse strength:
```python
visualizer.plot_full_network(min_synapse_display=50, show_individual_neurons=False)
```

**Community detection fails**
→ Install python-louvain: `pip install python-louvain`

**Heatmap not showing**
→ Install seaborn: `pip install seaborn`

**Qt/matplotlib crash**
→ Module uses non-interactive 'Agg' backend by default. If issues persist, check your matplotlib configuration.

---

## Acknowledgments

- **DoOR database creators**: Daniel Münch & C. Giovanni Galizia
- **Original R package**: rOpenSci DoOR project
- **FlyWire Consortium**: For comprehensive connectome data
- **Contributors**: Cole Hanan and the *Drosophila* neuroscience community
- **Raman Lab**: WashU neuroscience research

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Links

- **PyPI:** https://pypi.org/project/door-python-toolkit/
- **GitHub:** https://github.com/yourusername/door-python-toolkit
- **Documentation:** https://door-python-toolkit.readthedocs.io
- **Issues:** https://github.com/yourusername/door-python-toolkit/issues
- **Original DoOR:** https://github.com/ropensci/DoOR.data
- **FlyWire:** https://flywire.ai/
- **Raman Lab:** https://ramanlab.wustl.edu/

---

**Made with ❤️ for the *Drosophila* neuroscience community**
