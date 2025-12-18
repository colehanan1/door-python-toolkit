# Changelog

All notable changes to this project will be documented in this file. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and the project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Authoritative DoOR→FlyWire mapping system with provenance + strict validations (`src/door_toolkit/integration/door_to_flywire_mapping.py`).
- Publication-critical mapping artifacts tracked under `data/mappings/`:
  - `data/mappings/door_to_flywire_mapping.csv`
  - `data/mappings/door_to_flywire_manual_overrides.csv`
  - `data/mappings/sensillum_to_receptor_reference.csv`
- Mapping pipeline documentation: `docs/DOOR_TO_FLYWIRE_MAPPING.md`.

### Changed
- `data/mappings/receptor_inventory.csv` is generated from the authoritative mapping artifact and defines “mapped” as “mapped to a valid `ORN_` FlyWire label” (not passthrough strings).
- Adult-only filtering now follows DoOR.mappings (`adult=False`, `larva=True`) larval-only flags (DoOR 2.0; DOI: 10.1038/srep21841) throughout integration and inventory.
- `.gitignore` now tracks `data/mappings/**` while continuing to ignore bulk `data/*`.

### Fixed
- Corrected known mapping mismatches with explicit provenance:
  - `Or10a → ORN_DL1` (DoOR.mappings; Münch & Galizia 2016, DOI: 10.1038/srep21841)
  - `Ir64a.DC4 → ORN_DC4` and `Ir64a.DP1m → ORN_DP1m` (DoOR dotted-suffix convention)
- Prevented ambiguous multi-glomerulus DoOR units (e.g., `DM5+DM3`, `DL2d/v`) from being silently treated as single mappings in adult analyses.

## [1.0.0] - 2025-12-17

### Added - Mushroom Body Circuit Validation (Major Release - Production Ready)

**New Module: Mushroom Body Pathway Tracer**
- Complete ORN→PN→KC→MBON pathway tracing through FlyWire connectome
- Anatomical validation of LASSO-identified receptors in learning circuits
- Circuit classification: appetitive (α/β lobe) vs aversive (γ lobe)
- Priority matrix generation for experimental validation

**Core Components:**
- `mushroom_body_tracer.py` - Complete pathway tracer with 4 key classes:
  - `PathwayStep`: Single synapse connection
  - `MushroomBodyPathway`: Complete ORN→PN→KC pathway
  - `ConnectivityMetrics`: Circuit validation scores
  - `MushroomBodyTracer`: Main tracing and analysis engine

**Connectivity Metrics:**
- **ORN→PN Strength**: % of ORN output reaching PNs (0-1)
- **KC Coverage**: % of Kenyon Cells contacted (0-1)
- **Lobe Fractions**: α/β (appetitive), γ (aversive), α'β' compartments
- **Circuit Score**: Composite 0-1 validation score
- **Circuit Type**: Automatic appetitive/aversive classification

**Integration Features:**
- Sensillum-to-receptor mapping (ab2B→Or85a, ab3A→Or22a, ab1A→Or42b)
- LASSO behavioral data integration
- Priority ranking: 60% behavioral + 40% connectivity
- Experimental priority matrices for optogenetic validation

**Analysis Pipeline:**
- `flywire_mb_pathway_analysis.py` - Complete workflow from LASSO to priorities
- Batch processing of multiple receptors
- CSV/JSON export of pathway summaries and metrics
- Publication-ready visualizations (scatter plots, bar charts)

**ORN Database System (NEW!):**
- `build_complete_orn_database.py` - Batch processor for all 78 DoOR receptors
- `orn_database_tools.py` - Instant lookup API for pre-computed database
- Checkpoint saving every 10 receptors (crash-resistant)
- Zero-latency queries after one-time build (3-5 hours)
- CSV + JSON database formats
- Comprehensive filtering, ranking, and comparison tools

**Database Features:**
- `ORNDatabase` class with full query interface
- Convenience functions: `get_orn_mapping()`, `compare_orns()`, `rank_orns_by_metric()`
- Filter by circuit type, score range
- Rank by any metric (circuit_score, kc_coverage, etc.)
- Statistics and formatted summaries

**Example Scripts:**
- `test_database_build.py` - Verification test (5 receptors)
- `test_database_lookup.py` - Database query testing
- Complete documentation in `BUILD_GUIDE.md` and `STATUS.md`

**Bug Fixes:**
- Fixed MB neuropil filters to include hemisphere suffixes (_L, _R)
- Added proper sensillum-to-receptor translation
- Fixed pathway attribute access (use `len(pathway.orn_ids)` not `pathway.n_orns`)

**Documentation:**
- New "Mushroom Body Circuit Validation" section in README (~200 lines)
- Complete workflow examples: LASSO → FlyWire → Priorities → Optogenetics
- API documentation for all classes and methods
- Real-world example with Or67c, Or22b, Or85a validation
- Biological interpretation guide

**Results Achieved:**
- Successfully traced pathways for 10 test receptors
- Validated 3 high-priority candidates (Or22b, Or85a, Or42a)
- Circuit type distribution: 40% appetitive, 60% aversive
- Average circuit score: 0.813 across tested receptors

### Changed
- Updated README.md with comprehensive mushroom body documentation
- **Major version bump to 1.0.0** - Production/Stable release
- Enhanced package description to include new features
- Added mushroom-body, behavioral-prediction, lasso, optogenetics keywords
- Updated development status to "Production/Stable"

### Performance
- Database build: ~7 seconds for 5 receptors
- Projected full build: 3-5 hours for 78 receptors
- Query latency: <1ms after database build
- Database load time: <100ms

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
