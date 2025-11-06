"""Test fixtures for DoOR Python Toolkit tests."""

import json
import pytest
import pandas as pd
import numpy as np
from pathlib import Path


@pytest.fixture
def mock_door_cache(tmp_path):
    """
    Create minimal synthetic DoOR cache for testing.

    Creates a realistic cache structure with:
    - 20 synthetic odorants (InChIKeys)
    - 10 synthetic receptors (Or1a-Or10a)
    - Normalized response values [0, 1] with some NaN (missing data)
    - Realistic metadata (Name, CAS, Formula, etc.)
    """
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()

    # Create synthetic odorant data
    n_odorants = 20
    n_receptors = 10

    # Generate InChIKeys (realistic format)
    inchikeys = [
        f"TESTKEY{i:014d}-UHFFFAOYSA-N" for i in range(n_odorants)
    ]

    # Generate receptor names (Or1a, Or2a, etc.)
    receptors = [f"Or{i+1}a" for i in range(n_receptors)]

    # Create synthetic response matrix with some NaN values (realistic)
    np.random.seed(42)
    response_data = np.random.rand(n_odorants, n_receptors).astype(np.float32)

    # Add some missing data (NaN) to simulate real DoOR
    missing_mask = np.random.rand(n_odorants, n_receptors) < 0.3
    response_data[missing_mask] = np.nan

    # Create response DataFrame
    response_df = pd.DataFrame(
        response_data,
        index=inchikeys,
        columns=receptors
    )
    response_df.to_parquet(cache_dir / "response_matrix_norm.parquet")

    # Create metadata DataFrame
    odor_names = [
        "acetic acid", "ethanol", "1-pentanol", "butyric acid",
        "methyl acetate", "ethyl acetate", "propionic acid",
        "1-butanol", "benzaldehyde", "2-heptanone",
        "geraniol", "linalool", "limonene", "citral",
        "vanillin", "eugenol", "cinnamaldehyde", "menthol",
        "isoamyl acetate", "hexanal"
    ]

    meta_df = pd.DataFrame({
        'Name': odor_names,
        'CAS': [f'{100+i:03d}-{i:02d}-{i}' for i in range(n_odorants)],
        'Formula': ['C2H4O2', 'C2H6O', 'C5H12O', 'C4H8O2',
                    'C3H6O2', 'C4H8O2', 'C3H6O2', 'C4H10O',
                    'C7H6O', 'C7H14O', 'C10H18O', 'C10H18O',
                    'C10H16', 'C10H16O', 'C8H8O3', 'C10H12O2',
                    'C9H8O', 'C10H20O', 'C7H14O2', 'C6H12O'],
        'MolecularWeight': np.random.uniform(60, 200, n_odorants),
        'SMILES': [f'C{i}' for i in range(n_odorants)],  # Simplified
    }, index=inchikeys)
    meta_df.to_parquet(cache_dir / "odor_metadata.parquet")

    # Create index files for NumPy format support
    pd.Series(receptors, name='receptor').to_csv(
        cache_dir / "receptor_index.csv", index=False
    )
    pd.Series(inchikeys, name='odorant').to_csv(
        cache_dir / "odorant_index.csv", index=False
    )

    # Create CSV and NumPy versions for format testing
    response_df.to_csv(cache_dir / "response_matrix_norm.csv")
    np.save(cache_dir / "response_matrix_norm.npy", response_data)

    # Create metadata.json
    metadata = {
        "source": "synthetic_test_data",
        "version": "test_v1.0",
        "n_odorants": n_odorants,
        "n_receptors": n_receptors,
        "date_created": "2025-01-01"
    }
    with open(cache_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    return cache_dir


@pytest.fixture
def mock_encoder(mock_door_cache):
    """Create DoOREncoder instance with mock cache."""
    from door_toolkit import DoOREncoder
    return DoOREncoder(str(mock_door_cache), use_torch=False)


@pytest.fixture
def mock_encoder_torch(mock_door_cache):
    """Create DoOREncoder instance with PyTorch enabled (if available)."""
    from door_toolkit import DoOREncoder
    try:
        return DoOREncoder(str(mock_door_cache), use_torch=True)
    except ImportError:
        pytest.skip("PyTorch not installed")
