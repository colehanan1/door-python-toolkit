# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- _Placeholder_ – add new entries here.

### Changed
- _Placeholder_ – record behaviour changes here.

### Fixed
- _Placeholder_ – document bug fixes here.

## [0.3.0] - 2025-11-06

### Added - Connectomics Module (Major Feature)

**New Module: `door_toolkit.connectomics`**
- Complete toolkit for analyzing interglomerular cross-talk in Drosophila olfactory system using FlyWire connectome data
- NetworkX-based directed graph construction with 108,980+ pathways across 38 glomeruli
- Hierarchical representation: individual neurons (2,828 nodes) + glomerulus meta-nodes
- Biophysically realistic parameters based on Wilson, Olsen, Kazama lab research

**Four Analysis Modes:**
1. **Single ORN Focus** - Analyze all pathways from one ORN/glomerulus (`analyze_single_orn`)
2. **ORN Pair Comparison** - Bidirectional cross-talk with asymmetry quantification (`compare_orn_pair`)
3. **Full Network View** - Global topology, hub detection, community structure
4. **Pathway Search** - Find specific connections between neurons (`find_pathways`)

**Statistical Analyses:**
- Hub neuron detection (degree, betweenness, closeness, eigenvector centrality)
- Community detection (Louvain, greedy modularity, label propagation)
- Asymmetry quantification for directional connectivity
- Path length distributions and clustering coefficients
- Network motif analysis

**Visualization System:**
- Publication-ready network plots (300 DPI, PNG/PDF/SVG)
- Hierarchical neuron/glomerulus visualization
- Glomerulus connectivity heatmaps (with seaborn)
- Single ORN pathway diagrams
- Force-directed, hierarchical, and circular layouts
- Non-interactive backend (matplotlib Agg) for headless servers

**Data Handling:**
- Efficient CSV loading for large datasets (100K+ pathways)
- Configurable synapse thresholds (1-200+ synapses)
- Pathway type filtering (ORN→LN→ORN, ORN→LN→PN, ORN→PN→feedback)
- Export to Cytoscape (GraphML) and Gephi (GEXF) formats
- JSON configuration serialization

**Core Modules (7 files, ~3,500 lines):**
- `config.py` - Network configuration with biophysical parameters
- `data_loader.py` - CSV loading and preprocessing
- `network_builder.py` - NetworkX graph construction (CrossTalkNetwork class)
- `pathway_analysis.py` - Four analysis modes with result classes
- `visualization.py` - Publication-ready plotting (NetworkVisualizer class)
- `statistics.py` - Hub detection, communities, asymmetry (NetworkStatistics class)
- `__init__.py` - Clean API with exported classes

**Example Scripts (5 files):**
- `example_1_single_orn_analysis.py` - DL5 glomerulus pathway analysis
- `example_2_orn_pair_comparison.py` - Compare DL5 vs VA1v cross-talk
- `example_3_full_network_analysis.py` - Hub detection, communities, asymmetry
- `example_4_pathway_search.py` - Find pathways between VM7v and D
- `analyze_data_characteristics.py` - Data quality and threshold analysis

**Documentation (3 comprehensive files):**
- `CONNECTOMICS_README.md` - Full user guide (500+ lines)
- `ANALYSIS_FINDINGS.md` - Data analysis results and biological insights
- `CONNECTOMICS_SUMMARY.md` - Implementation details

**Unit Tests:**
- `tests/test_connectomics.py` - 32 comprehensive tests covering all functionality
- Tests for config, data loading, network building, all 4 analysis modes, statistics, and edge cases

**Key Discoveries from Data Analysis:**
- Lateral inhibition (ORN→LN→ORN) is widespread (52% of pathways) but weak (median 3 synapses)
- PN feedback pathways are rare (20%) but strong (up to 1,018 synapses)
- DL5 glomerulus uses primarily PN feedback, minimal lateral inhibition
- VM7v acts as convergence hub receiving from multiple glomeruli
- 15 functional communities detected, with one major 22-glomerulus cluster
- Hub LNs identified: lLN2T_c, lLN2X04, lLN8, LN60b (prime optogenetic targets)

### Changed
- Package version bumped to 0.3.0
- Added matplotlib (>=3.5.0) to core dependencies for visualization
- Updated package description to include "FlyWire connectomics"
- Updated main `__init__.py` with connectomics module documentation and examples

### Fixed
- Matplotlib Qt plugin crashes on headless servers (added `matplotlib.use('Agg')`)
- KeyError when pathways not found (added default values in empty results)
- Examples now use biologically appropriate thresholds based on pathway strength analysis

### Dependencies
- **New core dependency:** matplotlib>=3.5.0
- **New optional dependency group:** `[connectomics]`
  - seaborn>=0.11.0 (for heatmaps)
  - python-louvain>=0.16 (for community detection)
- Install with: `pip install door-python-toolkit[connectomics]` or `pip install door-python-toolkit[all]`

## [0.2.0] - 2025-11-06

### Added
- Multi-odor CLI workflow via `door-extract --odors` for side-by-side receptor comparisons, including automatic spread ranking.
- Receptor group shortcuts (`--receptor or|ir|gr|neuron`) with tail summaries that highlight the lowest responding odorants alongside the top hits.
- CSV export support (`--save`) for receptor and odor comparison tables, writing dash-separated headers for easy downstream processing.
- Coverage output now reports both the strongest and weakest receptors to speed up exploratory analysis.

### Changed
- README instructions updated with multi-odor and receptor-tail examples, plus clarified debugging guidance.

### Fixed
- Normalised cache indices to use `InChIKey`, preventing lookup errors when encoding odors from extracted datasets.
- Coerced response matrices to numeric dtype to keep coverage statistics and ranking functions stable.

## [0.1.0] - 2025-11-06

### Added
- Initial public release of the DoOR Python Toolkit.
- `DoORExtractor` for converting DoOR R package assets into Python-friendly formats.
- `DoOREncoder` for encoding odorant names into projection neuron activation patterns.
- Utilities for listing odorants, loading response matrices, exporting subsets, and validating caches.
- Command-line interface (`door-extract`) for extraction, validation, and cache inspection.
- Optional PyTorch integration and accompanying unit tests.
- Continuous integration workflows, documentation scaffolding, and example notebooks.

### Changed
- Not applicable (initial release).

### Fixed
- Not applicable (initial release).

## Future versions

Upcoming releases will continue to expand the toolkit (e.g., receptor selection strategies, similarity search improvements, and data import helpers). Contributions are welcome—see `CONTRIBUTING.md`.

---

Release links:

- [Unreleased](https://github.com/colehanan1/door-python-toolkit/compare/v0.3.0...HEAD)
- [0.3.0](https://github.com/colehanan1/door-python-toolkit/compare/v0.2.0...v0.3.0)
- [0.2.0](https://github.com/colehanan1/door-python-toolkit/compare/v0.1.0...v0.2.0)
- [0.1.0](https://github.com/colehanan1/door-python-toolkit/releases/tag/v0.1.0)
