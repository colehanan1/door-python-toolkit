"""
Data Loader
===========

Functions for loading and preprocessing FlyWire connectivity data
from CSV files.

This module handles:
- Loading large CSV files efficiently
- Filtering by synapse thresholds
- Combining multiple pathway types
- Caching processed data
- Validating data integrity
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass

from door_toolkit.connectomics.config import NetworkConfig

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ConnectivityData:
    """
    Container for connectivity data.

    Attributes:
        pathways: DataFrame with complete pathway information
        neurons: Dictionary mapping neuron IDs to their properties
        glomeruli: List of unique glomeruli in the dataset
        pathway_counts: Dictionary counting pathways by type
    """
    pathways: pd.DataFrame
    neurons: Dict[str, Dict]
    glomeruli: List[str]
    pathway_counts: Dict[str, int]

    @property
    def num_pathways(self) -> int:
        """Total number of pathways."""
        return len(self.pathways)

    @property
    def num_neurons(self) -> int:
        """Total number of unique neurons."""
        return len(self.neurons)

    @property
    def num_glomeruli(self) -> int:
        """Total number of unique glomeruli."""
        return len(self.glomeruli)

    def get_neuron_info(self, neuron_id: str) -> Optional[Dict]:
        """Get information about a specific neuron."""
        return self.neurons.get(neuron_id)

    def filter_by_synapse_count(
        self,
        min_count: int = 1,
        max_count: Optional[int] = None
    ) -> 'ConnectivityData':
        """
        Filter pathways by synapse count.

        Args:
            min_count: Minimum synapse count (step 2) to include
            max_count: Maximum synapse count (step 2) to include (optional)

        Returns:
            New ConnectivityData object with filtered pathways
        """
        filtered = self.pathways.copy()
        filtered = filtered[filtered['synapse_count_step2'] >= min_count]

        if max_count is not None:
            filtered = filtered[filtered['synapse_count_step2'] <= max_count]

        # Rebuild neuron and glomeruli lists
        return _build_connectivity_data(filtered)

    def get_pathways_for_orn(
        self,
        orn_identifier: Union[str, int],
        by_glomerulus: bool = False
    ) -> pd.DataFrame:
        """
        Get all pathways originating from a specific ORN.

        Args:
            orn_identifier: Either ORN root_id (int) or glomerulus name (str)
            by_glomerulus: If True, treat identifier as glomerulus name

        Returns:
            DataFrame of pathways from this ORN
        """
        if by_glomerulus or isinstance(orn_identifier, str):
            return self.pathways[
                self.pathways['orn_glomerulus'] == orn_identifier
            ]
        else:
            return self.pathways[
                self.pathways['orn_root_id'] == orn_identifier
            ]

    def summary(self) -> str:
        """Generate summary statistics string."""
        lines = [
            "Connectivity Data Summary",
            "=" * 50,
            f"Total pathways: {self.num_pathways:,}",
            f"Unique neurons: {self.num_neurons:,}",
            f"Unique glomeruli: {self.num_glomeruli}",
            "",
            "Pathway type breakdown:",
        ]

        for pathway_type, count in self.pathway_counts.items():
            lines.append(f"  {pathway_type}: {count:,}")

        # Synapse count statistics
        syn_counts = self.pathways['synapse_count_step2']
        lines.extend([
            "",
            "Synapse count statistics (step 2):",
            f"  Mean: {syn_counts.mean():.2f}",
            f"  Median: {syn_counts.median():.0f}",
            f"  Min: {syn_counts.min()}",
            f"  Max: {syn_counts.max()}",
            f"  Std: {syn_counts.std():.2f}",
        ])

        return "\n".join(lines)


def load_connectivity_data(
    filepath: Union[str, Path],
    config: Optional[NetworkConfig] = None
) -> ConnectivityData:
    """
    Load connectivity data from a CSV file.

    Args:
        filepath: Path to CSV file
        config: NetworkConfig object for filtering (optional)

    Returns:
        ConnectivityData object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CSV has incorrect format
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading connectivity data from {filepath}")

    try:
        # Read CSV with appropriate dtypes
        df = pd.read_csv(filepath, dtype={
            'orn_root_id': str,
            'orn_label': str,
            'orn_glomerulus': str,
            'level1_root_id': str,
            'level1_cell_type': str,
            'level1_category': str,
            'level2_root_id': str,
            'level2_cell_type': str,
            'level2_category': str,
            'synapse_count_step1': int,
            'synapse_count_step2': int,
        })
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    # Validate required columns
    required_cols = [
        'orn_root_id', 'orn_label', 'orn_glomerulus',
        'level1_root_id', 'level1_cell_type', 'level1_category',
        'level2_root_id', 'level2_cell_type', 'level2_category',
        'synapse_count_step1', 'synapse_count_step2'
    ]

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Apply config filters if provided
    if config is not None:
        df = _apply_config_filters(df, config)

    logger.info(f"Loaded {len(df):,} pathways")

    return _build_connectivity_data(df)


def load_glomerulus_matrix(
    filepath: Union[str, Path]
) -> pd.DataFrame:
    """
    Load glomerulus-to-glomerulus connectivity matrix.

    Args:
        filepath: Path to crosstalk_matrix_glomerulus.csv

    Returns:
        DataFrame with columns: source_glom, target_glom, synapse_count_step2
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logger.info(f"Loading glomerulus matrix from {filepath}")

    df = pd.read_csv(filepath, dtype={
        'source_glom': str,
        'target_glom': str,
        'synapse_count_step2': int,
    })

    # Remove any empty rows
    df = df.dropna()

    logger.info(f"Loaded matrix with {len(df):,} connections")

    return df


def load_multiple_pathway_types(
    filepaths: Dict[str, Union[str, Path]],
    config: Optional[NetworkConfig] = None
) -> ConnectivityData:
    """
    Load and combine multiple pathway type files.

    Args:
        filepaths: Dictionary mapping pathway types to file paths
                   e.g., {'ORN_LN_ORN': 'crosstalk_ORN_LN_ORN.csv', ...}
        config: NetworkConfig object for filtering (optional)

    Returns:
        Combined ConnectivityData object
    """
    dfs = []

    for pathway_type, filepath in filepaths.items():
        logger.info(f"Loading {pathway_type} from {filepath}")
        try:
            df = pd.read_csv(filepath, dtype={
                'orn_root_id': str,
                'orn_label': str,
                'orn_glomerulus': str,
                'level1_root_id': str,
                'level1_cell_type': str,
                'level1_category': str,
                'level2_root_id': str,
                'level2_cell_type': str,
                'level2_category': str,
                'synapse_count_step1': int,
                'synapse_count_step2': int,
            })

            # Add pathway type column
            df['pathway_type'] = pathway_type
            dfs.append(df)

        except Exception as e:
            logger.warning(f"Failed to load {pathway_type}: {e}")

    if not dfs:
        raise ValueError("No pathway files successfully loaded")

    # Combine all dataframes
    combined = pd.concat(dfs, ignore_index=True)

    # Apply config filters if provided
    if config is not None:
        combined = _apply_config_filters(combined, config)

    logger.info(f"Combined total: {len(combined):,} pathways")

    return _build_connectivity_data(combined)


def _apply_config_filters(df: pd.DataFrame, config: NetworkConfig) -> pd.DataFrame:
    """Apply filters from NetworkConfig to dataframe."""
    filtered = df.copy()

    # Apply synapse threshold
    filtered = filtered[
        filtered['synapse_count_step2'] >= config.min_synapse_threshold
    ]

    if config.max_synapse_threshold is not None:
        filtered = filtered[
            filtered['synapse_count_step2'] <= config.max_synapse_threshold
        ]

    # Apply pathway type filters
    if not config.include_orn_ln_orn:
        filtered = filtered[
            ~((filtered['level1_category'] == 'Local_Neuron') &
              (filtered['level2_category'] == 'ORN'))
        ]

    if not config.include_orn_ln_pn:
        filtered = filtered[
            ~((filtered['level1_category'] == 'Local_Neuron') &
              (filtered['level2_category'] == 'Projection_Neuron'))
        ]

    if not config.include_orn_pn_feedback:
        filtered = filtered[
            ~(filtered['level1_category'] == 'Projection_Neuron')
        ]

    return filtered


def _build_connectivity_data(df: pd.DataFrame) -> ConnectivityData:
    """Build ConnectivityData object from dataframe."""

    # Extract unique neurons with their properties
    neurons = {}

    # Add ORNs
    for _, row in df[['orn_root_id', 'orn_label', 'orn_glomerulus']].drop_duplicates().iterrows():
        neurons[row['orn_root_id']] = {
            'type': 'ORN',
            'label': row['orn_label'],
            'glomerulus': row['orn_glomerulus'],
            'category': 'ORN',
        }

    # Add level 1 neurons (LNs or PNs)
    for _, row in df[['level1_root_id', 'level1_cell_type', 'level1_category']].drop_duplicates().iterrows():
        if row['level1_root_id'] not in neurons:
            neurons[row['level1_root_id']] = {
                'type': row['level1_cell_type'],
                'category': row['level1_category'],
            }

    # Add level 2 neurons (targets)
    for _, row in df[['level2_root_id', 'level2_cell_type', 'level2_category']].drop_duplicates().iterrows():
        if row['level2_root_id'] not in neurons:
            neurons[row['level2_root_id']] = {
                'type': row['level2_cell_type'],
                'category': row['level2_category'],
            }

    # Extract unique glomeruli
    glomeruli = sorted(df['orn_glomerulus'].unique())

    # Count pathway types
    pathway_counts = {}
    if 'pathway_type' in df.columns:
        pathway_counts = df['pathway_type'].value_counts().to_dict()
    else:
        # Infer pathway types from categories
        for _, row in df.iterrows():
            l1_cat = row['level1_category']
            l2_cat = row['level2_category']

            if l1_cat == 'Local_Neuron' and l2_cat == 'ORN':
                ptype = 'ORN_LN_ORN'
            elif l1_cat == 'Local_Neuron' and l2_cat == 'Projection_Neuron':
                ptype = 'ORN_LN_PN'
            elif l1_cat == 'Projection_Neuron':
                ptype = 'ORN_PN_feedback'
            else:
                ptype = 'Other'

            pathway_counts[ptype] = pathway_counts.get(ptype, 0) + 1

    return ConnectivityData(
        pathways=df,
        neurons=neurons,
        glomeruli=glomeruli,
        pathway_counts=pathway_counts,
    )


def validate_data_files(data_dir: Union[str, Path]) -> Dict[str, bool]:
    """
    Validate that all required data files exist and are readable.

    Args:
        data_dir: Directory containing data files

    Returns:
        Dictionary mapping file types to validation status
    """
    data_dir = Path(data_dir)

    required_files = {
        'full': 'interglomerular_crosstalk_pathways.csv',
        'orn_ln_orn': 'crosstalk_ORN_LN_ORN.csv',
        'orn_ln_pn': 'crosstalk_ORN_LN_PN.csv',
        'orn_pn_feedback': 'crosstalk_ORN_PN_feedback.csv',
        'glomerulus_matrix': 'crosstalk_matrix_glomerulus.csv',
    }

    validation_results = {}

    for file_type, filename in required_files.items():
        filepath = data_dir / filename
        exists = filepath.exists()
        validation_results[file_type] = exists

        if exists:
            logger.info(f"✓ Found {filename}")
        else:
            logger.warning(f"✗ Missing {filename}")

    return validation_results
