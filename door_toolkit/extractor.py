#!/usr/bin/env python3
"""
DoOR Data Extraction Utility for PGCN Project
==============================================

Extracts odorant response data from DoOR (Database of Odorant Responses) v2.0.0
R packages into Python-native formats (pandas DataFrames, numpy arrays, parquet).

Usage:
    python door_extractor.py --input /path/to/DoOR.data-xxx/data --output data/door_cache

Requirements:
    pip install pyreadr pandas numpy

Author: PGCN Project
Date: 2025-11-05
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple
import hashlib
import json

import numpy as np
import pandas as pd

try:
    import pyreadr
except ImportError:
    print("ERROR: pyreadr not installed. Install with: pip install pyreadr", file=sys.stderr)
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class DoORExtractor:
    """
    Extract and cache DoOR odorant response data in Python-native formats.
    
    Attributes:
        input_dir: Path to DoOR.data/data directory containing .RData files
        output_dir: Path to output cache directory
    """
    
    def __init__(self, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Validate input
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {self.input_dir}")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"[INIT] Input: {self.input_dir}")
        logger.info(f"[INIT] Output: {self.output_dir}")
    
    def load_rdata(self, rdata_path: Path) -> Dict:
        """
        Load an RData file and return all objects as a dictionary.
        
        Args:
            rdata_path: Path to .RData file
            
        Returns:
            Dictionary mapping object names to their values (typically pandas DataFrames)
        """
        logger.debug(f"Loading RData: {rdata_path.name}")
        
        try:
            result = pyreadr.read_r(str(rdata_path))
            logger.debug(f"  -> Loaded {len(result)} objects: {list(result.keys())}")
            return result
        except Exception as e:
            logger.error(f"Failed to load {rdata_path.name}: {e}")
            raise
    
    def extract_response_matrix(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extract the main odorant × receptor response matrix.
        
        Returns:
            Tuple of (normalized_matrix, non_normalized_matrix) as DataFrames
            Rows = odorants, Columns = receptors/ORNs
        """
        logger.info("[EXTRACT] Response matrices...")
        
        # Load normalized response matrix
        norm_path = self.input_dir / "response.matrix.RData"
        non_norm_path = self.input_dir / "response.matrix_non.normalized.RData"
        
        if not norm_path.exists():
            raise FileNotFoundError(f"response.matrix.RData not found in {self.input_dir}")
        
        norm_data = self.load_rdata(norm_path)
        response_norm = norm_data['response.matrix']
        
        non_norm_data = self.load_rdata(non_norm_path)
        response_non_norm = non_norm_data['response.matrix_non.normalized']
        
        logger.info(f"  Normalized matrix: {response_norm.shape} (odorants × receptors)")
        logger.info(f"  Non-normalized matrix: {response_non_norm.shape}")
        logger.info(f"  Sparsity: {(response_norm.isna().sum().sum() / response_norm.size) * 100:.2f}% missing")
        
        return response_norm, response_non_norm
    
    def extract_odor_metadata(self) -> pd.DataFrame:
        """
        Extract odor metadata (CAS numbers, names, chemical properties).
        
        Returns:
            DataFrame with odor information
        """
        logger.info("[EXTRACT] Odor metadata...")
        
        odor_path = self.input_dir / "odor.RData"
        if not odor_path.exists():
            logger.warning("odor.RData not found, skipping metadata")
            return pd.DataFrame()
        
        odor_data = self.load_rdata(odor_path)
        odor_df = odor_data['odor']
        
        logger.info(f"  Odor metadata: {odor_df.shape[0]} odorants, {odor_df.shape[1]} attributes")
        logger.info(f"  Columns: {list(odor_df.columns)}")
        
        return odor_df
    
    def extract_al_map(self) -> pd.DataFrame:
        """
        Extract antennal lobe (AL) glomerular mapping.
        
        Returns:
            DataFrame mapping receptors to glomeruli
        """
        logger.info("[EXTRACT] AL glomerular map...")
        
        al_path = self.input_dir / "AL.map.RData"
        if not al_path.exists():
            logger.warning("AL.map.RData not found, skipping")
            return pd.DataFrame()
        
        al_data = self.load_rdata(al_path)
        
        # Handle different possible key names or empty data
        if not al_data:
            logger.warning("  AL.map.RData is empty")
            return pd.DataFrame()
        
        # Try common key names
        for key in ['AL.map', 'ALmap', 'al.map']:
            if key in al_data:
                al_df = al_data[key]
                logger.info(f"  AL map: {al_df.shape[0]} mappings")
                return al_df
        
        logger.warning(f"  AL map keys not found. Available: {list(al_data.keys())}")
        return pd.DataFrame()
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """Compute SHA256 hash of DataFrame for cache validation."""
        return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:16]
    
    def save_cache(self, response_norm: pd.DataFrame, response_non_norm: pd.DataFrame,
                   odor_meta: pd.DataFrame, al_map: pd.DataFrame):
        """
        Save extracted data to cache in multiple formats.
        
        Saves:
            - Parquet files (efficient, type-safe)
            - CSV files (human-readable)
            - NumPy arrays (for direct use in training)
            - Metadata JSON
        """
        logger.info("[CACHE] Writing cache files...")
        
        # Save parquet (recommended for Python workflows)
        response_norm.to_parquet(self.output_dir / "response_matrix_norm.parquet")
        response_non_norm.to_parquet(self.output_dir / "response_matrix_non_norm.parquet")
        
        if not odor_meta.empty:
            odor_meta.to_parquet(self.output_dir / "odor_metadata.parquet")
        
        if not al_map.empty:
            al_map.to_parquet(self.output_dir / "al_map.parquet")
        
        # Save CSV (for inspection/compatibility)
        response_norm.to_csv(self.output_dir / "response_matrix_norm.csv")
        
        # Save numpy arrays (for ML training)
        # Convert to float32 for efficiency, fill NaN with 0
        response_numeric = response_norm.fillna(0.0).astype(np.float32).values
        np.save(self.output_dir / "response_matrix_norm.npy", response_numeric)
        
        # Save receptor/odor indices
        receptor_names = pd.Series(response_norm.columns, name='receptor')
        receptor_names.to_csv(self.output_dir / "receptor_index.csv", index=False)
        
        odor_names = pd.Series(response_norm.index, name='odorant')
        odor_names.to_csv(self.output_dir / "odorant_index.csv", index=False)
        
        # Save metadata
        metadata = {
            "source": "DoOR v2.0.0",
            "extracted_at": pd.Timestamp.now().isoformat(),
            "n_odorants": int(response_norm.shape[0]),
            "n_receptors": int(response_norm.shape[1]),
            "sparsity_pct": float((response_norm.isna().sum().sum() / response_norm.size) * 100),
            "response_range": [float(response_norm.min().min()), float(response_norm.max().max())],
            "data_hash": self.compute_data_hash(response_norm),
            "receptor_list": list(response_norm.columns),
            "odorant_count_per_receptor": response_norm.notna().sum().to_dict()
        }
        
        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"  ✓ Cached {len(list(self.output_dir.glob('*')))} files")
        
    def generate_report(self, response_norm: pd.DataFrame, odor_meta: pd.DataFrame):
        """Generate a markdown summary report."""
        logger.info("[REPORT] Generating summary...")
        
        report = f"""# DoOR Data Extraction Report

**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Source:** DoOR v2.0.0 (Database of Odorant Responses)

## Dataset Overview

- **Odorants:** {response_norm.shape[0]} unique odorant compounds
- **Receptors:** {response_norm.shape[1]} ORN/receptor types
- **Total measurements:** {response_norm.notna().sum().sum():,}
- **Sparsity:** {(response_norm.isna().sum().sum() / response_norm.size) * 100:.2f}% missing values
- **Value range:** [{response_norm.min().min():.2f}, {response_norm.max().max():.2f}]

## Top Receptors by Coverage

| Receptor | Odorants Tested | Coverage % |
|----------|----------------|------------|
"""
        
        coverage = response_norm.notna().sum().sort_values(ascending=False).head(10)
        for receptor, count in coverage.items():
            pct = (count / response_norm.shape[0]) * 100
            report += f"| {receptor} | {count} | {pct:.1f}% |\n"
        
        report += f"""
## Glomerular Distribution

{response_norm.columns.tolist()}

## Usage in PGCN Project

### Load Response Matrix (Recommended)

```python
import pandas as pd
import numpy as np

# Load normalized response matrix (odorants × receptors)
response_df = pd.read_parquet('data/door_cache/response_matrix_norm.parquet')

# Or load as numpy array
response_array = np.load('data/door_cache/response_matrix_norm.npy')
receptor_names = pd.read_csv('data/door_cache/receptor_index.csv')['receptor'].tolist()
```

### Create Odor-PN Encoder

```python
# Map odorants to PN (glomerular) activation patterns
def door_odor_to_pn(odor_name: str, response_df: pd.DataFrame) -> np.ndarray:
    '''Convert odor name to PN activation vector.'''
    if odor_name not in response_df.index:
        raise KeyError(f"Odor {{odor_name}} not in DoOR database")
    
    pn_response = response_df.loc[odor_name].fillna(0.0).values
    return pn_response

# Example: Get ethyl acetate response
ethyl_acetate_pn = door_odor_to_pn('ethyl acetate', response_df)
```

### Integration with PGCN

```python
# In pgcn/encoders.py
class DoOREncoder:
    def __init__(self, door_cache_path: str):
        self.response_matrix = pd.read_parquet(f'{{door_cache_path}}/response_matrix_norm.parquet')
        self.n_channels = self.response_matrix.shape[1]
    
    def encode(self, odor_name: str) -> torch.Tensor:
        pn_activation = self.response_matrix.loc[odor_name].fillna(0.0).values
        return torch.from_numpy(pn_activation).float()
```

## Files Generated

- `response_matrix_norm.parquet` - Normalized response matrix (recommended)
- `response_matrix_norm.csv` - CSV format for inspection
- `response_matrix_norm.npy` - NumPy array for training
- `odor_metadata.parquet` - Odorant chemical properties
- `al_map.parquet` - Receptor-to-glomerulus mapping
- `receptor_index.csv` - Ordered receptor names
- `odorant_index.csv` - Ordered odorant names
- `metadata.json` - Extraction metadata & data hash

## Next Steps

1. Verify receptor names match your connectome annotation (Or10a, Or22a, etc.)
2. Decide on PN channel count (use all {response_norm.shape[1]} or subset)
3. Handle missing values (fill with 0, mean, or train a predictor)
4. Integrate into `pgcn/encoders.py` as `DoOREncoder` class

---
*DoOR Database: Münch & Galizia (2016), Scientific Data 3:160122*
"""
        
        report_path = self.output_dir / "extraction_report.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"  ✓ Report saved: {report_path}")
        print(f"\n{report}\n")
    
    def run(self):
        """Execute full extraction pipeline."""
        logger.info("="*60)
        logger.info("DoOR Data Extraction Pipeline - PGCN Project")
        logger.info("="*60)
        
        # Extract all components
        response_norm, response_non_norm = self.extract_response_matrix()
        odor_meta = self.extract_odor_metadata()
        al_map = self.extract_al_map()
        
        # Save cache
        self.save_cache(response_norm, response_non_norm, odor_meta, al_map)
        
        # Generate report
        self.generate_report(response_norm, odor_meta)
        
        logger.info("="*60)
        logger.info("[SUCCESS] DoOR extraction complete!")
        logger.info(f"Cache location: {self.output_dir.absolute()}")
        logger.info("="*60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract DoOR odorant response data for PGCN project",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract from unzipped DoOR.data package
    python door_extractor.py --input Dahaniel-DoOR.data-6436660/data --output data/door_cache
    
    # With custom input/output
    python door_extractor.py -i /path/to/DoOR.data/data -o /path/to/output
    
    # Enable debug logging
    python door_extractor.py -i DoOR.data/data -o cache --debug
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        required=True,
        help='Path to DoOR.data/data directory containing .RData files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        required=True,
        help='Output directory for cached data'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run extraction
    try:
        extractor = DoORExtractor(args.input, args.output)
        extractor.run()
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
