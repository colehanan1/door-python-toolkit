# Custom Analysis CLI Tool

Flexible command-line interface for analyzing specific glomeruli and odorants in the DoOR Python Toolkit.

## What It Does

This tool lets you run custom analyses on:
- Specific glomeruli/ORNs
- Specific odorants
- Network pathways
- Hub neurons
- Communities
- Receptor responses

All from the command line with flexible options!

## Quick Start

```bash
# Make script executable (first time only)
chmod +x scripts/custom_analysis.py

# Analyze a glomerulus
python scripts/custom_analysis.py --mode single-orn --glomerulus DL5 --threshold 10

# Compare two glomeruli
python scripts/custom_analysis.py --mode compare --glomeruli DL5 VA1v

# Analyze odorants
python scripts/custom_analysis.py --mode odorant --odorants "acetic acid" "ethanol"

# Full network analysis
python scripts/custom_analysis.py --mode network --detect-hubs --detect-communities
```

## Documentation

- **[CLI_QUICK_REFERENCE.md](../CLI_QUICK_REFERENCE.md)** - Quick reference card with all commands
- **[CUSTOM_USAGE_GUIDE.md](../CUSTOM_USAGE_GUIDE.md)** - Comprehensive guide for using the API programmatically

## Six Analysis Modes

1. **single-orn** - Analyze all pathways from one glomerulus
2. **compare** - Compare two glomeruli bidirectionally
3. **pathway** - Find specific pathways between source and target
4. **network** - Full network analysis with statistics
5. **odorant** - Encode odorants and analyze receptor responses
6. **odorant-pathway** - Find best odorants for a receptor

## Examples

### Mode 1: Single ORN Analysis
```bash
# Analyze DL5 glomerulus, export CSV and visualize
python scripts/custom_analysis.py \
  --mode single-orn \
  --glomerulus DL5 \
  --threshold 10 \
  --export-csv \
  --visualize \
  --output-dir results/DL5
```

### Mode 2: Compare ORNs
```bash
# Compare DL5 vs VA1v cross-talk
python scripts/custom_analysis.py \
  --mode compare \
  --glomeruli DL5 VA1v \
  --threshold 10 \
  --visualize
```

### Mode 3: Pathway Search
```bash
# Find pathways from VM7v to D
python scripts/custom_analysis.py \
  --mode pathway \
  --source VM7v \
  --target D \
  --max-paths 5
```

### Mode 4: Network Analysis
```bash
# Full network characterization
python scripts/custom_analysis.py \
  --mode network \
  --threshold 10 \
  --detect-hubs \
  --detect-communities \
  --export-graph \
  --visualize
```

### Mode 5: Odorant Analysis
```bash
# Analyze specific odorants
python scripts/custom_analysis.py \
  --mode odorant \
  --odorants "acetic acid" "ethanol" "methyl acetate"

# Search by pattern
python scripts/custom_analysis.py \
  --mode odorant \
  --odorant-pattern "alcohol" \
  --receptors Or42b Or47b \
  --top-n 20
```

### Mode 6: Odorant-Pathway
```bash
# Find best odorants for Or47b
python scripts/custom_analysis.py \
  --mode odorant-pathway \
  --receptor Or47b \
  --top-n 10 \
  --export-csv
```

## Key Options

### Essential
- `--mode` - Analysis mode (required)
- `--threshold N` - Minimum synapse count (default: 5)
- `--top-n N` - Number of top results (default: 10)

### Data Sources
- `--connectomics-data PATH` - Path to connectomics CSV
- `--door-cache PATH` - Path to DoOR cache

### Analysis
- `--by-glomerulus` - Analyze at glomerulus level
- `--detect-hubs` - Find hub neurons
- `--detect-communities` - Detect communities
- `--max-paths N` - Max shortest paths to find

### Output
- `--export-csv` - Save results to CSV
- `--export-graph` - Save network graph
- `--visualize` - Generate plots
- `--output-dir PATH` - Output directory
- `--verbose` - Verbose output

## Common Workflows

### Characterize a New Glomerulus
```bash
# Step 1: Analyze pathways
python scripts/custom_analysis.py \
  --mode single-orn \
  --glomerulus DL5 \
  --threshold 10 \
  --export-csv \
  --visualize \
  --verbose

# Step 2: Compare with another
python scripts/custom_analysis.py \
  --mode compare \
  --glomeruli DL5 VA1v \
  --visualize

# Step 3: Find specific pathways
python scripts/custom_analysis.py \
  --mode pathway \
  --source DL5 \
  --target VA1v \
  --max-paths 10
```

### Screen Odorants for Receptor
```bash
# Step 1: Find best odorants
python scripts/custom_analysis.py \
  --mode odorant-pathway \
  --receptor Or47b \
  --top-n 20 \
  --export-csv

# Step 2: Analyze specific odorants
python scripts/custom_analysis.py \
  --mode odorant \
  --odorants "acetic acid" "ethanol" \
  --receptors Or42b Or47b Or7a \
  --export-csv
```

### Full Network Characterization
```bash
python scripts/custom_analysis.py \
  --mode network \
  --threshold 10 \
  --detect-hubs \
  --detect-communities \
  --hub-method betweenness \
  --top-n-hubs 20 \
  --export-graph \
  --export-csv \
  --visualize \
  --output-dir results/network_full \
  --verbose
```

## Get Help

```bash
# Show all options
python scripts/custom_analysis.py --help

# Quick reference
cat CLI_QUICK_REFERENCE.md

# Comprehensive guide
cat CUSTOM_USAGE_GUIDE.md
```

## Using the Python API

For even more flexibility, use the Python API directly:

```python
from door_toolkit import DoOREncoder
from door_toolkit.connectomics import CrossTalkNetwork

# Your custom analysis
encoder = DoOREncoder("door_cache")
network = CrossTalkNetwork.from_csv("interglomerular_crosstalk_pathways.csv")

# ... your code here ...
```

See [CUSTOM_USAGE_GUIDE.md](../CUSTOM_USAGE_GUIDE.md) for detailed API examples.

## Tips

1. Use `--verbose` to see what's happening
2. Start with `--no-output` for quick testing
3. Adjust `--threshold` based on your needs:
   - `5-10`: Include weak connections
   - `50+`: Only strong connections
4. Export with `--export-csv` for further analysis
5. Use `--visualize` for publication-ready figures

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "DoOR cache not found" | Run `door-extract` first |
| "File not found" | Use `--connectomics-data PATH` |
| "No pathways found" | Lower `--threshold` |
| Import errors | `pip install -e .[connectomics]` |

---

Happy analyzing!
