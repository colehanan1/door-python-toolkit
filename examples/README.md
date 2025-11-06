# Examples

This directory contains example scripts demonstrating DoOR Toolkit usage.

## Quick Start

### Basic Usage
```bash
python examples/basic_usage.py
```

Demonstrates:
- Extracting DoOR data
- Loading encoder
- Encoding single and batch odorants
- Searching odorants
- Finding similar compounds

### PyTorch Integration
```bash
python examples/pytorch_example.py
```

Demonstrates:
- Creating PyTorch datasets from DoOR
- Training a neural network classifier
- Batch processing with DataLoader
- Making predictions

## Running Examples

1. **First, extract DoOR data:**
   ```bash
   door-extract --input path/to/DoOR.data/data --output door_cache
   ```

2. **Then run examples:**
   ```bash
   cd door-python-toolkit
   python examples/basic_usage.py
   ```

## Custom Examples

Feel free to modify these examples for your own research. Common use cases:

### Use Case 1: Olfactory Circuit Modeling
```python
from door_toolkit import DoOREncoder
import numpy as np

encoder = DoOREncoder("door_cache")

# Get PN responses for experimental stimuli
odors = ["acetic acid", "ethyl acetate", "1-pentanol"]
pn_activations = encoder.batch_encode(odors)

# Feed into your circuit model
kc_responses = your_kc_model(pn_activations)
```

### Use Case 2: Chemical Space Analysis
```python
from door_toolkit.utils import find_similar_odorants

# Find compounds with similar receptor profiles
similar = find_similar_odorants(
    target_odor="acetic acid",
    cache_path="door_cache",
    top_k=10,
    method="correlation"
)

for name, similarity in similar:
    print(f"{name}: {similarity:.3f}")
```

### Use Case 3: Data Export for R/MATLAB
```python
from door_toolkit.utils import load_response_matrix

# Load and export subset
df = load_response_matrix("door_cache")
acetates = df[df.index.str.contains("acetate", case=False)]
acetates.to_csv("acetates_for_analysis.csv")
```

## Need Help?

- Check the [main README](../README.md)
- Read the [API docs](https://door-python-toolkit.readthedocs.io)
- Open an [issue](https://github.com/yourusername/door-python-toolkit/issues)
