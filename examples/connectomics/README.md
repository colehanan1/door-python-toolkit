# Connectomics Examples

This directory contains example scripts demonstrating the four analysis modes of the door-toolkit connectomics module.

## Prerequisites

```bash
# Install the toolkit with connectomics support
pip install -e ".[connectomics]"

# Or install all dependencies
pip install -e ".[all]"
```

## Data Files

Before running the examples, ensure you have the connectivity data files in the project root:

- `interglomerular_crosstalk_pathways.csv` (~109K rows)
- `crosstalk_ORN_LN_ORN.csv` (optional)
- `crosstalk_ORN_LN_PN.csv` (optional)
- `crosstalk_ORN_PN_feedback.csv` (optional)
- `crosstalk_matrix_glomerulus.csv` (optional)

## Examples

### Example 1: Single ORN/Glomerulus Analysis
```bash
python example_1_single_orn_analysis.py
```

**Demonstrates:**
- Loading a network from CSV
- Setting synapse thresholds
- Analyzing pathways from a single glomerulus
- Exporting results to CSV
- Creating visualizations

**Output:**
- `{GLOMERULUS}_pathways.csv` - All pathways from the glomerulus
- `{GLOMERULUS}_pathways.png` - Visualization of pathways

---

### Example 2: ORN Pair Comparison
```bash
python example_2_orn_pair_comparison.py
```

**Demonstrates:**
- Comparing cross-talk between two glomeruli
- Quantifying asymmetry
- Finding shared intermediate neurons
- Bidirectional analysis

**Output:**
- `{GLOM1}_to_{GLOM2}_pathways.csv` - Forward pathways
- `{GLOM2}_to_{GLOM1}_pathways.csv` - Reverse pathways
- `pair_comparisons.csv` - Summary table

---

### Example 3: Full Network Analysis
```bash
python example_3_full_network_analysis.py
```

**Demonstrates:**
- Hub neuron detection (degree, betweenness centrality)
- Community detection
- Path length analysis
- Asymmetry matrix calculation
- Comprehensive network statistics
- Multiple visualization types

**Output:**
- `full_network_glomerulus.png` - Full network visualization
- `glomerulus_heatmap.png` - Connectivity heatmap
- `network_analysis_report.txt` - Statistical report
- `asymmetry_matrix.csv` - Complete asymmetry data
- `network.graphml` - For Cytoscape
- `network.gexf` - For Gephi

---

### Example 4: Pathway Search
```bash
python example_4_pathway_search.py
```

**Demonstrates:**
- Finding all pathways between two glomeruli
- Filtering by intermediate neuron type
- Matrix search across multiple pairs
- Shortest path analysis
- Hub LN identification

**Output:**
- `{SOURCE}_to_{TARGET}_pathways.csv` - Pathway details
- `connectivity_matrix.csv` - Connection strength matrix

---

## Customization

All examples can be easily modified by changing:

```python
# Change data file location
DATA_FILE = "/path/to/your/data.csv"

# Change output directory
OUTPUT_DIR = Path("/path/to/output")

# Change glomeruli to analyze
glomerulus = "ORN_VA1v"  # Instead of ORN_DL5

# Change synapse threshold
network.set_min_synapse_threshold(20)  # Stricter filtering

# Change pathway filters
config = NetworkConfig()
config.set_pathway_filters(
    orn_ln_orn=True,   # Lateral inhibition
    orn_ln_pn=False,   # Disable feedforward
    orn_pn_feedback=False  # Disable feedback
)
network = CrossTalkNetwork.from_csv(DATA_FILE, config)
```

## Running All Examples

To run all examples in sequence:

```bash
for example in example_*.py; do
    echo "Running $example..."
    python "$example"
    echo "---"
done
```

## Common Issues

**Issue:** `FileNotFoundError`
**Solution:** Make sure data files are in the correct location or update the `DATA_FILE` path

**Issue:** `ModuleNotFoundError: No module named 'seaborn'`
**Solution:** Install optional dependencies: `pip install seaborn python-louvain`

**Issue:** Visualizations are cluttered
**Solution:** Increase `min_synapse_display` parameter or set higher threshold

## Further Help

- See [CONNECTOMICS_README.md](../../CONNECTOMICS_README.md) for full documentation
- API reference and detailed usage examples
- Biological context and research applications
