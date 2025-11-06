# DoOR Python Toolkit - New Features

This document describes the three major feature additions to the door-python-toolkit.

## Table of Contents

- [Installation](#installation)
- [Feature 1: FlyWire Integration](#feature-1-flywire-integration)
- [Feature 2: Pathway Analysis Tools](#feature-2-pathway-analysis-tools)
- [Feature 3: Neural Network Preprocessing](#feature-3-neural-network-preprocessing)
- [CLI Commands](#cli-commands)
- [Examples](#examples)

## Installation

### Core Package with New Features

```bash
pip install door-python-toolkit[all]
```

### Individual Feature Groups

```bash
# FlyWire integration
pip install door-python-toolkit[flywire]

# With PyTorch support
pip install door-python-toolkit[torch]

# All optional dependencies
pip install door-python-toolkit[all]
```

## Feature 1: FlyWire Integration

Map DoOR receptor data to FlyWire neural connectivity and community labels.

### Key Capabilities

- Parse 100K+ FlyWire community labels efficiently
- Map DoOR receptors to FlyWire root IDs
- Generate 3D spatial activation maps
- Export mappings in JSON/CSV formats

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

## Feature 2: Pathway Analysis Tools

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

# Behavioral prediction
predictor = BehavioralPredictor("door_cache")
prediction = predictor.predict_behavior("hexanol")
print(f"Valence: {prediction.predicted_valence}")
print(f"Confidence: {prediction.confidence:.2%}")
```

### CLI Usage

```bash
# Trace pathways
door-pathways --cache door_cache --trace or47b-feeding
door-pathways --cache door_cache --trace or42b

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

## Feature 3: Neural Network Preprocessing

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

# Dataset statistics
stats = preprocessor.get_dataset_statistics()
print(f"Coverage: {stats['mean_receptor_coverage']:.1%}")
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

# Train/val split
door-neural --cache door_cache --split --train-fraction 0.8 \
  --output train_val_split.json
```

## CLI Commands

### New Commands

- `door-flywire` - FlyWire integration tools
- `door-pathways` - Pathway analysis and experiments
- `door-neural` - Neural network preprocessing

### Getting Help

```bash
door-flywire --help
door-pathways --help
door-neural --help
```

## Examples

Complete working examples are available in the `examples/advanced/` directory:

- **[flywire_integration_example.py](examples/advanced/flywire_integration_example.py)** - FlyWire mapping and spatial analysis
- **[pathway_analysis_example.py](examples/advanced/pathway_analysis_example.py)** - Pathway tracing and experiment generation
- **[neural_preprocessing_example.py](examples/advanced/neural_preprocessing_example.py)** - Neural network data preparation

### Running Examples

```bash
# Make sure you have DoOR cache
door-extract --input DoOR.data/data --output door_cache

# Run examples
python examples/advanced/flywire_integration_example.py
python examples/advanced/pathway_analysis_example.py
python examples/advanced/neural_preprocessing_example.py
```

## Performance

- **FlyWire parsing**: 100K+ labels in <30 seconds
- **Receptor mapping**: >80% success rate
- **Sparse encoding**: Maintains 5±1% sparsity
- **Memory usage**: <2GB for largest datasets

## Testing

Run the comprehensive test suite:

```bash
# Install dev dependencies
pip install -e .[dev]

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=door_toolkit --cov-report=html
```

## Dependencies

### Required
- pandas>=1.5.0
- numpy>=1.21.0
- pyarrow>=12.0.0
- scipy>=1.9.0
- scikit-learn>=1.1.0
- networkx>=2.8
- tqdm>=4.64.0

### Optional
- torch>=2.0.0 (neural network export)
- plotly>=5.11.0 (visualization)
- matplotlib>=3.5.0 (plotting)
- h5py>=3.7.0 (HDF5 export)

## Citation

If you use these features in your research, please cite:

```bibtex
@software{door_python_toolkit,
  author = {Hanan, Cole},
  title = {DoOR Python Toolkit: FlyWire Integration and Neural Network Tools},
  year = {2024},
  url = {https://github.com/colehanan1/door-python-toolkit}
}
```

## Support

- **Issues**: [GitHub Issues](https://github.com/colehanan1/door-python-toolkit/issues)
- **Documentation**: [ReadTheDocs](https://door-python-toolkit.readthedocs.io)
- **Lab**: [Raman Lab at WashU](https://ramanlab.wustl.edu/)

## License

MIT License - see LICENSE file for details.
