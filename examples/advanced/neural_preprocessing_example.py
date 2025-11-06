"""
Neural Network Preprocessing Example
======================================

This example demonstrates preparing DoOR data for neural network training,
including sparse encoding, noise augmentation, and PGCN dataset export.

Requirements:
    - DoOR cache (run door-extract first)
    - PyTorch (optional, for PyTorch export)
"""

import numpy as np
from door_toolkit.neural import DoORNeuralPreprocessor

# Example 1: Create sparse KC-like encoding
print("=" * 70)
print("Example 1: Sparse KC-like Encoding")
print("=" * 70)

preprocessor = DoORNeuralPreprocessor(
    "door_cache",
    n_kc_neurons=2000,
    random_seed=42,
)

# Use subset of odorants for demonstration
odorants = preprocessor.encoder.odorant_names[:50]

sparse_data = preprocessor.create_sparse_encoding(
    sparsity_level=0.05,  # 5% active neurons (KC-like)
    odorants=odorants,
)

print(f"\nSparse Encoding:")
print(f"  Shape: {sparse_data.shape}")
print(f"  Actual sparsity: {(sparse_data > 0).mean():.2%}")
print(f"  Mean activation (active neurons): {sparse_data[sparse_data > 0].mean():.3f}")
print(f"  Max activation: {sparse_data.max():.3f}")

# Save sparse encoding
np.save("output/sparse_encoding.npy", sparse_data)
print(f"\n  Saved to output/sparse_encoding.npy")


# Example 2: Generate noise-augmented dataset
print("\n" + "=" * 70)
print("Example 2: Noise-Augmented Dataset")
print("=" * 70)

print("\nGenerating augmented dataset (this may take a moment)...")
aug_orn, aug_kc, labels = preprocessor.generate_noise_augmented_responses(
    n_augmentations=3,  # 3x augmentation
    noise_level=0.1,
    odorants=odorants[:20],  # Use subset for speed
)

print(f"\nAugmented Dataset:")
print(f"  Original odorants: 20")
print(f"  Augmented samples: {len(labels)}")
print(f"  ORN shape: {aug_orn.shape}")
print(f"  KC shape: {aug_kc.shape}")
print(f"  KC sparsity: {(aug_kc > 0).mean():.2%}")

# Save augmented data
np.save("output/augmented_orn.npy", aug_orn)
np.save("output/augmented_kc.npy", aug_kc)
with open("output/labels.txt", "w") as f:
    f.write("\n".join(labels))

print(f"\n  Saved to output/augmented_*.npy")


# Example 3: Export PGCN dataset
print("\n" + "=" * 70)
print("Example 3: Export PGCN Dataset")
print("=" * 70)

print("\nExporting PGCN dataset...")

# Try PyTorch export
try:
    preprocessor.export_pgcn_dataset(
        output_dir="output/pgcn_dataset",
        format="pytorch",
        include_sparse=True,
        include_metadata=True,
    )
    print(f"\n  Exported PyTorch dataset to output/pgcn_dataset/")
except ImportError:
    print("\n  PyTorch not available, exporting NumPy format instead...")
    preprocessor.export_pgcn_dataset(
        output_dir="output/pgcn_dataset",
        format="numpy",
        include_sparse=True,
        include_metadata=True,
    )
    print(f"\n  Exported NumPy dataset to output/pgcn_dataset/")


# Example 4: Dataset statistics
print("\n" + "=" * 70)
print("Example 4: Dataset Statistics")
print("=" * 70)

stats = preprocessor.get_dataset_statistics()

print(f"\nDataset Statistics:")
print(f"  Number of odorants: {stats['n_odorants']}")
print(f"  Number of receptors: {stats['n_receptors']}")
print(f"  Mean response: {stats['mean_response']:.3f}")
print(f"  Std response: {stats['std_response']:.3f}")
print(f"  Mean receptor coverage: {stats['mean_receptor_coverage']:.1%}")
print(f"  Sparsity (>0.3): {stats['sparsity_at_threshold_0.3']:.2%}")
print(f"  Response range: [{stats['min_response']:.3f}, {stats['max_response']:.3f}]")


# Example 5: Train/validation split
print("\n" + "=" * 70)
print("Example 5: Train/Validation Split")
print("=" * 70)

train_odorants, val_odorants = preprocessor.create_training_validation_split(
    train_fraction=0.8,
    random_seed=42,
)

print(f"\nTrain/Validation Split:")
print(f"  Training samples: {len(train_odorants)}")
print(f"  Validation samples: {len(val_odorants)}")
print(f"  Split ratio: {len(train_odorants) / (len(train_odorants) + len(val_odorants)):.2%} train")

print(f"\n  Sample training odorants: {', '.join(train_odorants[:5])}")
print(f"  Sample validation odorants: {', '.join(val_odorants[:5])}")

# Save split
import json

split_data = {
    "train": train_odorants,
    "validation": val_odorants,
    "train_fraction": 0.8,
}
with open("output/train_val_split.json", "w") as f:
    json.dump(split_data, f, indent=2)

print(f"\n  Saved split to output/train_val_split.json")


# Example 6: Concentration-response modeling
print("\n" + "=" * 70)
print("Example 6: Concentration-Response Modeling")
print("=" * 70)

from door_toolkit.neural.concentration_models import ConcentrationResponseModel

model = ConcentrationResponseModel()

# Generate synthetic concentration-response curve
concentrations = np.array([0.001, 0.01, 0.1, 1.0])
responses = np.array([0.1, 0.3, 0.7, 0.9])

print("\nFitting Hill equation...")
params = model.fit_hill_equation(concentrations, responses)

print(f"\nHill Parameters:")
print(f"  R_max: {params.r_max:.3f}")
print(f"  EC50: {params.ec50:.3f}")
print(f"  Hill coefficient: {params.hill_coefficient:.3f}")
print(f"  R_baseline: {params.r_baseline:.3f}")

# Generate concentration series
conc_series, resp_series = model.generate_concentration_series(
    params, log_start=-4, log_end=0, n_points=50
)

print(f"\nGenerated {len(conc_series)} concentration points")
print(f"  Concentration range: [{conc_series.min():.2e}, {conc_series.max():.2e}]")
print(f"  Response range: [{resp_series.min():.3f}, {resp_series.max():.3f}]")


print("\n" + "=" * 70)
print("All Examples Complete!")
print("=" * 70)
print("\nGenerated files in output/ directory:")
print("  - sparse_encoding.npy")
print("  - augmented_orn.npy, augmented_kc.npy")
print("  - pgcn_dataset/ (PyTorch or NumPy format)")
print("  - train_val_split.json")
