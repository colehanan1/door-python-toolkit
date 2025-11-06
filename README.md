[![PyPI version](https://badge.fury.io/py/door-python-toolkit.svg)](https://badge.fury.io/py/door-python-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

# DoOR Python Toolkit

**Python toolkit for working with the DoOR (Database of Odorant Responses) database.**

Extract, analyze, and integrate *Drosophila* odorant-receptor response data in pure Python. No R installation required.

---

## Features

- ✅ **Pure Python** - Extract DoOR R data files without installing R
- 🚀 **Fast** - Parquet-based caching for quick loading
- 🧠 **ML-Ready** - PyTorch/NumPy integration for neural network training
- 📊 **693 odorants × 78 receptors** - Comprehensive *Drosophila* olfactory data
- 🔍 **Search & Filter** - Query by odorant name, receptor, or chemical properties
- 📦 **PyPI Package** - `pip install door-python-toolkit`

---

## Quick Start

### Installation

```bash
pip install door-python-toolkit
```

For extraction workflows (requires `pyreadr`):
```bash
pip install door-python-toolkit[extract]
```

For PyTorch support:
```bash
pip install door-python-toolkit[torch]
```

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

# Encode single odorant → 78-dim PN activation vector
pn_activation = encoder.encode("acetic acid")
print(pn_activation.shape)  # (78,)

# Encode batch
odors = ["acetic acid", "1-pentanol", "ethyl acetate"]
pn_batch = encoder.batch_encode(odors)
print(pn_batch.shape)  # (3, 78)

# Search odorants
acetates = encoder.list_available_odorants(pattern="acetate")
print(f"Found {len(acetates)} acetates")  # 36

# Get metadata
stats = encoder.get_receptor_coverage("acetic acid")
print(f"Active receptors: {stats['n_active']}")
```

---

## What is DoOR?

The **Database of Odorant Responses (DoOR)** is a comprehensive collection of odorant-receptor response measurements for *Drosophila melanogaster* (fruit fly).

**Published:** Münch & Galizia (2016), *Scientific Data* 3:160122  
**Citation:** https://doi.org/10.1038/sdata.2016.122  
**Original R package:** https://github.com/ropensci/DoOR.data

### Dataset Overview

| Metric | Value |
|--------|-------|
| Odorants | 693 compounds |
| Receptors | 78 ORN types (Or, Ir, Gr) |
| Measurements | 7,381 odorant-receptor pairs |
| Sparsity | 86% (typical for chemical screens) |
| Response Range | [0, 1] normalized |

**Top receptors by coverage:**
- Or19a: 71.7% (497/693 odorants)
- Or10a: 33.9% (235/693 odorants)
- Or22a: 32.5% (225/693 odorants)

---

## Use Cases

### 1. **Neuroscience Research**
Map odorants to glomerular activation patterns for modeling olfactory processing.

```python
from door_toolkit import DoOREncoder
import matplotlib.pyplot as plt

encoder = DoOREncoder("door_cache")

# Compare response profiles
odors = ["acetic acid", "ethyl acetate", "1-pentanol"]
responses = [encoder.encode(o) for o in odors]

plt.figure(figsize=(10, 4))
for i, (odor, resp) in enumerate(zip(odors, responses)):
    plt.subplot(1, 3, i+1)
    plt.bar(range(len(resp)), resp)
    plt.title(odor)
    plt.xlabel("Receptor")
    plt.ylabel("Response")
plt.tight_layout()
plt.show()
```

### 2. **Machine Learning**
Train neural networks with empirical odorant-receptor data.

```python
from door_toolkit import DoOREncoder
import torch
import torch.nn as nn

# Load encoder
encoder = DoOREncoder("door_cache", use_torch=True)

# Create dataset
class OdorDataset(torch.utils.data.Dataset):
    def __init__(self, odor_names, labels):
        self.encoder = DoOREncoder("door_cache")
        self.odor_names = odor_names
        self.labels = labels
    
    def __len__(self):
        return len(self.odor_names)
    
    def __getitem__(self, idx):
        pn_activation = self.encoder.encode(self.odor_names[idx])
        return pn_activation, self.labels[idx]

# Train model
model = nn.Sequential(
    nn.Linear(78, 256),  # 78 = n_receptors
    nn.ReLU(),
    nn.Linear(256, 2)    # Binary classification
)

# ... standard PyTorch training loop
```

### 3. **Chemical Similarity**
Find odorants with similar receptor response patterns.

```python
from door_toolkit.utils import find_similar_odorants

similar = find_similar_odorants(
    target_odor="acetic acid",
    cache_path="door_cache",
    top_k=5,
    method="correlation"
)

for name, similarity in similar:
    print(f"{name}: {similarity:.3f}")
```

### 4. **Data Analysis**
Export subsets for custom analyses.

```python
from door_toolkit.utils import export_subset, list_odorants

# Export all acetates
acetates = list_odorants("door_cache", pattern="acetate")
export_subset(
    cache_path="door_cache",
    output_path="acetates.csv",
    odorants=acetates
)
```

---

## Command-Line Interface

```bash
# Extract DoOR data
door-extract --input DoOR.data/data --output door_cache

# Validate cache contents
door-extract --validate door_cache

# List odorants (optional substring filter)
door-extract --list-odorants door_cache --pattern acetate

# Encode an odorant and show receptor responses
door-extract --cache door_cache --odor "ethyl butyrate" --coverage

# Compare multiple odorants (max/min tables, CSV export)
door-extract --cache door_cache --odors "ethyl butyrate" "acetic acid" "butyric acid" --top 15 --coverage --save reports/odor-comparison

# Inspect receptor / neuron response profiles (limit to top 25 hits + bottom 25 summary)
door-extract --cache door_cache --receptor Or42b --top 25

# Use receptor families (odorant, ionotropic, gustatory, neuron groups)
door-extract --cache door_cache --receptor or --top 10
```

Add `--debug` to any command to print detailed tracebacks and internal logging if something fails.

Receptor group shortcuts:
- `or` – Odorant receptors (OrXX)
- `ir` – Ionotropic receptors (IrXX)
- `gr` – Gustatory receptors (GrXX)
- `neuron` – Antennal/palp neuron classes (ab*, ac*, pb*)

Coverage output now reports both the strongest and weakest responding receptors for quick scanning.

Multi-odor mode renders two tables—one ranked by `Δ(max-min)` (spread) and another by minimum response—so you can quickly identify both the most differential and the most suppressed receptors. Add `--save path/prefix` to emit CSVs with dash-separated column headers (e.g., `prefix-max-spread-receptors.csv`). Receptor lookups now include a trailing “Lowest responses” section mirroring the top list when `--top` is specified.

---

## API Reference

### `DoORExtractor`
Extract DoOR R data files to Python formats.

**Methods:**
- `run()` - Execute full extraction pipeline
- `extract_response_matrix()` - Extract odorant-receptor matrix
- `extract_odor_metadata()` - Extract chemical properties

### `DoOREncoder`
Encode odorant names to neural activation patterns.

**Methods:**
- `encode(odor_name)` - Encode single odorant
- `batch_encode(odor_names)` - Encode multiple odorants
- `list_available_odorants(pattern)` - Search odorants
- `get_receptor_coverage(odor_name)` - Get coverage stats
- `get_odor_metadata(odor_name)` - Get chemical metadata

### Utilities
Helper functions in `door_toolkit.utils`:

- `load_response_matrix(cache_path)` - Load response data
- `load_odor_metadata(cache_path)` - Load metadata
- `list_odorants(cache_path, pattern)` - List odorants
- `get_receptor_info(cache_path)` - Receptor statistics
- `find_similar_odorants(target, cache_path, top_k)` - Similarity search
- `export_subset(cache_path, output_path, odorants, receptors)` - Export data
- `validate_cache(cache_path)` - Validate cache integrity

---

## Installation from Source

```bash
# Clone repository
git clone https://github.com/yourusername/door-python-toolkit.git
cd door-python-toolkit

# Create and activate a virtualenv (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install dependencies and expose the CLI locally
make install-dev    # equivalent to: pip install -e .[dev]

# Extract DoOR data (defaults use data/raw/DoOR.data-2.0.0/data -> door_cache)
make extract INPUT=path/to/DoOR.data/data OUTPUT=door_cache
# or run the helper script directly:
./scripts/door-extract --input path/to/DoOR.data/data --output door_cache

# Validate a cache
./scripts/validate-cache door_cache

# Run tests
make test

# Lint and format
make lint
make format
```

---

## Requirements

**Core:**
- Python ≥ 3.8
- pandas ≥ 1.5.0
- numpy ≥ 1.21.0
- pyarrow ≥ 12.0.0

**Optional:**
- pyreadr ≥ 0.4.7 (required for `DoORExtractor`)
- torch ≥ 2.0.0 (for PyTorch integration)

---

## Data Sources

This toolkit extracts data from the original DoOR R packages:

- **DoOR.data** - https://github.com/ropensci/DoOR.data
- **DoOR.functions** - https://github.com/ropensci/DoOR.functions

To use this toolkit, download and unzip the DoOR R packages:

```bash
# Download from GitHub
wget https://github.com/ropensci/DoOR.data/archive/refs/tags/v2.0.0.zip
unzip v2.0.0.zip

# Extract to Python
door-extract --input DoOR.data-2.0.0/data --output door_cache
# or, from a local checkout without installing:
./scripts/door-extract --input DoOR.data-2.0.0/data --output door_cache
```

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

---

## Citation

If you use this toolkit in your research, please cite both:

**This toolkit:**
```bibtex
@software{door_python_toolkit,
  author = {Your Name},
  title = {DoOR Python Toolkit},
  year = {2025},
  url = {https://github.com/yourusername/door-python-toolkit}
}
```

**Original DoOR database:**
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

---

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

## Troubleshooting

**"Odorant not found"**
→ Use `encoder.list_available_odorants()` to see exact names (case-insensitive)

**"Cache not found"**
→ Run `DoORExtractor` first to extract R data files

**"High sparsity"**
→ Normal for DoOR (86%). Use `fillna(0.0)` or filter to well-covered receptors

**PyTorch not available**
→ Install with `pip install door-python-toolkit[torch]`

---

## Acknowledgments

- DoOR database creators: Daniel Münch & C. Giovanni Galizia
- Original R package: rOpenSci DoOR project
- Contributors: Cole Hanan

---

## Links

- **PyPI:** https://pypi.org/project/door-python-toolkit/
- **GitHub:** https://github.com/yourusername/door-python-toolkit
- **Documentation:** https://door-python-toolkit.readthedocs.io
- **Issues:** https://github.com/yourusername/door-python-toolkit/issues
- **Original DoOR:** https://github.com/ropensci/DoOR.data

---

**Made with ❤️ for the *Drosophila* neuroscience community**
