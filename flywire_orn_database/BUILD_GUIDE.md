# Complete ORN-FlyWire Database - Build Guide

## Overview

This system creates a **complete, reusable database** mapping all 78 DoOR receptors to FlyWire mushroom body connectivity. Once built, you get **instant zero-latency queries** instead of running 3-4 hour analyses per receptor.

## What's Included

### 1. Database Builder (`examples/advanced/build_complete_orn_database.py`)

**Purpose**: Batch-process all 78 DoOR receptors through mushroom body pathway tracer.

**Features**:
- ✅ Automatic FlyWire mapping for all Or, Ir, Gr receptors
- ✅ Complete pathway tracing (ORN → PN → KC → MBON)
- ✅ Connectivity metrics calculation
- ✅ Checkpoint saving every 10 receptors (crash-resistant)
- ✅ Multiple output formats (CSV, JSON)
- ✅ Automatic visualization and statistics

**Runtime**: 3-5 hours for 78 receptors

**Output Files**:
```
flywire_orn_database/
  ├── flywire_orn_complete_v1.0.csv     # Excel-friendly database
  ├── flywire_orn_complete_v1.0.json    # Python-friendly database
  ├── database_statistics.txt           # Coverage report
  ├── database_visualization.png        # 4-panel overview
  └── checkpoint_*.csv                  # Progress checkpoints
```

### 2. Database Lookup Tools (`src/door_toolkit/flywire/orn_database_tools.py`)

**Purpose**: Instant queries once database is built.

**Key Classes**:
- `ORNDatabase`: Main lookup interface with filtering, ranking, comparison
- Convenience functions: `get_orn_mapping()`, `compare_orns()`, `rank_orns_by_metric()`

**Example Usage**:
```python
from door_toolkit.flywire.orn_database_tools import get_orn_mapping, compare_orns

# Instant lookup!
data = get_orn_mapping("Or49a")
print(f"Circuit score: {data['circuit_score']:.3f}")
print(f"Circuit type: {data['circuit_type']}")

# Compare multiple receptors
comparison = compare_orns(["Or22b", "Or85a", "Or42a"])
print(comparison)
```

## Build Status

### ✅ Testing Complete

**Test Results** (5 receptors):
```
✓ Or22b: 1 ORN → 4 PNs → 612 KCs (score: 0.806, appetitive)
✓ Or85a: 42 ORNs → 5 PNs → 391 KCs (score: 0.848, appetitive)
✓ Or42a: 33 ORNs → 12 PNs → 604 KCs (score: 0.752, appetitive)
✓ Or49a: 52 ORNs → 3 PNs → 211 KCs (score: 0.846, aversive)
✓ Or46a: 15 ORNs → 4 PNs → 380 KCs (score: 0.815, aversive)
```

**Test Files**:
- `examples/advanced/test_database_build.py` - Build system test (✓ PASSED)
- `examples/advanced/test_database_lookup.py` - Lookup tools test (✓ PASSED)
- `flywire_orn_database/test/test_results.csv` - Test database

### 🚀 Ready for Full Build

The system is fully tested and ready to build the complete 78-receptor database.

## How to Build Complete Database

### Prerequisites

Ensure you have the required FlyWire data files:
```bash
data/flywire/
  ├── processed_labels.csv.gz          # Cell labels (100K+ cells)
  ├── connections_princeton.csv.gz     # Synapses (5.3M connections)
  └── consolidated_cell_types.csv.gz   # Cell types (137K neurons)
```

And DoOR cache:
```bash
door_cache/
  ├── response_matrix_norm.parquet
  └── ...
```

### Run the Build

```bash
# Full 78-receptor build (3-5 hours)
python examples/advanced/build_complete_orn_database.py
```

**What happens**:
1. Loads all 78 DoOR receptors
2. Maps each to FlyWire ORN neurons
3. Traces complete pathways (ORN → PN → KC → MBON)
4. Calculates connectivity metrics
5. Saves checkpoint every 10 receptors
6. Generates final database files
7. Creates statistics report
8. Generates 4-panel visualization

**Progress tracking**:
- Real-time console output with tqdm progress bars
- Log file: `orn_database_build.log`
- Checkpoints: `flywire_orn_database/checkpoint_*.csv`

### Output Database Schema

**CSV Columns**:
```
receptor                    - Receptor name (Or49a, Ir75a, etc.)
status                      - success | not_found | error
n_orns                      - Number of ORN neurons
n_pns                       - Number of PN neurons contacted
n_kcs                       - Number of KC neurons contacted
n_mbons                     - Number of MBON neurons contacted
orn_to_pn_synapses          - Total ORN→PN synapses
pn_to_kc_synapses           - Total PN→KC synapses
kc_alpha_beta               - KCs in α/β lobe (appetitive)
kc_gamma                    - KCs in γ lobe (aversive)
kc_alpha_prime_beta_prime   - KCs in α'β' lobe
orn_to_pn_strength          - % of ORN output to PNs (0-1)
kc_coverage                 - % of total KCs contacted (0-1)
alpha_beta_fraction         - Fraction in α/β lobe (0-1)
gamma_fraction              - Fraction in γ lobe (0-1)
mbon_diversity              - Number of unique MBON types
circuit_score               - Overall connectivity score (0-1)
circuit_type                - appetitive | aversive
```

## Using the Database

### Method 1: Convenience Functions

```python
from door_toolkit.flywire.orn_database_tools import (
    get_orn_mapping,
    get_circuit_score,
    compare_orns,
    rank_orns_by_metric,
    print_orn_summary
)

# Get complete data
data = get_orn_mapping("Or49a")

# Get specific metric
score = get_circuit_score("Or49a")

# Compare receptors
comparison = compare_orns(["Or22b", "Or85a", "Or42a"])

# Rank by metric
top10 = rank_orns_by_metric("kc_coverage", top_n=10)

# Print formatted summary
print_orn_summary("Or49a")
```

### Method 2: ORNDatabase Class

```python
from door_toolkit.flywire.orn_database_tools import ORNDatabase

# Initialize (auto-finds database)
db = ORNDatabase()

# Or specify path
db = ORNDatabase("flywire_orn_database/flywire_orn_complete_v1.0.csv")

# Get data
data = db.get("Or49a")

# Filter by circuit type
appetitive = db.filter_by_circuit_type("appetitive")
aversive = db.filter_by_circuit_type("aversive")

# Filter by score range
high_connectivity = db.filter_by_circuit_score(min_score=0.80)

# Rank by metric
top_kc = db.rank_by_metric("kc_coverage", top_n=15)

# Compare multiple
comparison = db.compare_receptors(["Or22b", "Or85a", "Or42a"])

# Get statistics
stats = db.get_statistics()
```

## Example Workflows

### Workflow 1: Find High-Connectivity Receptors

```python
from door_toolkit.flywire.orn_database_tools import ORNDatabase

db = ORNDatabase()

# Find receptors with high MB connectivity
high_conn = db.filter_by_circuit_score(min_score=0.80)

# Rank by KC coverage
top_kc = high_conn.nlargest(10, "kc_coverage")

print(top_kc[["receptor", "circuit_score", "kc_coverage", "circuit_type"]])
```

### Workflow 2: Compare Appetitive vs Aversive Circuits

```python
from door_toolkit.flywire.orn_database_tools import ORNDatabase

db = ORNDatabase()

appetitive = db.filter_by_circuit_type("appetitive")
aversive = db.filter_by_circuit_type("aversive")

print(f"Appetitive receptors: {len(appetitive)}")
print(f"  Mean circuit score: {appetitive['circuit_score'].mean():.3f}")
print(f"  Mean KC coverage: {appetitive['kc_coverage'].mean():.2%}")

print(f"\nAversive receptors: {len(aversive)}")
print(f"  Mean circuit score: {aversive['circuit_score'].mean():.3f}")
print(f"  Mean KC coverage: {aversive['kc_coverage'].mean():.2%}")
```

### Workflow 3: Integrate with LASSO Results

```python
from door_toolkit.flywire.orn_database_tools import ORNDatabase
import pandas as pd

# Load LASSO results
lasso_df = pd.read_csv("behavioral_prediction_results/opto_hex_model.json")
lasso_weights = lasso_df["lasso_weights"]

# Load FlyWire database
db = ORNDatabase()

# Get FlyWire data for LASSO receptors
flywire_data = []
for receptor, weight in lasso_weights.items():
    data = db.get(receptor)
    if data and data["status"] == "success":
        flywire_data.append({
            "receptor": receptor,
            "lasso_weight": weight,
            "circuit_score": data["circuit_score"],
            "circuit_type": data["circuit_type"],
            "kc_coverage": data["kc_coverage"],
        })

# Create priority matrix
priority_df = pd.DataFrame(flywire_data)
priority_df["final_score"] = (
    0.6 * priority_df["lasso_weight"] +
    0.4 * priority_df["circuit_score"]
)
priority_df = priority_df.sort_values("final_score", ascending=False)

print("Experimental Priority Matrix:")
print(priority_df)
```

## Publication Checklist

### Before Publishing

- [ ] Run full database build (78 receptors)
- [ ] Verify statistics report
- [ ] Check database visualization
- [ ] Test lookup functions
- [ ] Update version number
- [ ] Write CHANGELOG.md entry

### Database Publication

**Option 1: GitHub Release**
```bash
# Create release with database as asset
git tag v0.4.0-database
git push origin v0.4.0-database

# Upload to GitHub release:
# - flywire_orn_complete_v1.0.csv
# - flywire_orn_complete_v1.0.json
# - database_statistics.txt
# - database_visualization.png
```

**Option 2: Zenodo DOI**
- Upload database to Zenodo for permanent DOI
- Include in paper citations

### PyPI Publication

Update `setup.py`:
```python
version="0.4.0",
description="DoOR Python Toolkit with FlyWire ORN connectivity database",
```

Build and publish:
```bash
python setup.py sdist bdist_wheel
twine check dist/*
twine upload dist/*
```

## Troubleshooting

### Database Build Issues

**Problem**: Build crashes mid-way
- **Solution**: Resume from latest checkpoint
- Checkpoints saved every 10 receptors in `flywire_orn_database/checkpoint_*.csv`

**Problem**: Some receptors not found
- **Solution**: Normal - not all DoOR receptors are in FlyWire
- Check `database_statistics.txt` for mapping success rate

**Problem**: Out of memory
- **Solution**: Reduce batch size or process in smaller chunks
- Modify `checkpoint_interval` in build script

### Lookup Issues

**Problem**: Database not found
- **Solution**: Specify path explicitly: `ORNDatabase("path/to/database.csv")`

**Problem**: Receptor not in database
- **Solution**: Check `db.list_all_receptors()` for available receptors

## Support

- **Issues**: https://github.com/yourusername/door-python-toolkit/issues
- **Documentation**: See README.md
- **Examples**: `examples/advanced/` directory

---

**Last Updated**: 2025-12-17
**Database Version**: v1.0
**Status**: ✅ Tested, Ready to Build
