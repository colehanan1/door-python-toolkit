"""
FlyWire Community Labels Parser
=================================

Efficient parsing and searching of FlyWire community labels for olfactory cells.

This module handles large datasets (100K+ rows) efficiently using streaming
and optimized search algorithms.
"""

import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class CellLabel:
    """Structure for a single FlyWire cell label."""

    root_id: str
    label: str
    supervoxel_id: Optional[str] = None
    position_x: Optional[float] = None
    position_y: Optional[float] = None
    position_z: Optional[float] = None

    @property
    def coordinates(self) -> Optional[Tuple[float, float, float]]:
        """Get 3D coordinates if available."""
        if all(v is not None for v in [self.position_x, self.position_y, self.position_z]):
            return (self.position_x, self.position_y, self.position_z)
        return None


class CommunityLabelsParser:
    """
    Parser for FlyWire community labels with efficient search capabilities.

    This class handles large community label datasets efficiently using
    streaming I/O and optimized indexing for fast lookups.

    Attributes:
        labels_df: DataFrame containing all parsed labels
        n_labels: Total number of labels parsed
        olfactory_patterns: Predefined patterns for olfactory cell types

    Example:
        >>> parser = CommunityLabelsParser("processed_labels.csv.gz")
        >>> parser.parse()
        >>> or_cells = parser.search_patterns(["Or42b", "Or47b"])
        >>> print(f"Found {len(or_cells)} Or cells")
    """

    # Predefined olfactory receptor patterns
    OLFACTORY_PATTERNS = {
        "or": r"\bOr\d+[a-z]*\b",  # Or42b, Or47b, etc.
        "ir": r"\bIr\d+[a-z]*\b",  # Ir75a, Ir84a, etc.
        "gr": r"\bGr\d+[a-z]*\b",  # Gr21a, Gr63a, etc.
        "sensillum_ab": r"\bab\d+[a-zA-Z]*\b",  # ab1A, ab2A, etc.
        "sensillum_ac": r"\bac\d+[a-zA-Z]*\b",  # ac1, ac2, etc.
        "sensillum_pb": r"\bpb\d+[a-zA-Z]*\b",  # pb1A, pb2A, etc.
    }

    def __init__(self, labels_path: str):
        """
        Initialize community labels parser.

        Args:
            labels_path: Path to community labels file (CSV or CSV.GZ)

        Raises:
            FileNotFoundError: If labels file doesn't exist
        """
        self.labels_path = Path(labels_path)
        if not self.labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {self.labels_path}")

        self.labels_df: Optional[pd.DataFrame] = None
        self.n_labels: int = 0
        self._index_cache: Dict[str, pd.DataFrame] = {}

    def parse(self, chunk_size: int = 10000, show_progress: bool = True) -> pd.DataFrame:
        """
        Parse community labels file with progress tracking.

        Args:
            chunk_size: Number of rows to read at once
            show_progress: Show progress bar during parsing

        Returns:
            DataFrame with parsed labels

        Example:
            >>> parser = CommunityLabelsParser("labels.csv.gz")
            >>> df = parser.parse()
            >>> print(df.columns)
            Index(['root_id', 'label', 'position_x', ...])
        """
        logger.info(f"Parsing community labels from {self.labels_path}")

        # Determine if file is compressed
        is_compressed = self.labels_path.suffix.lower() == ".gz"

        if is_compressed:
            open_fn = gzip.open
            mode = "rt"
        else:
            open_fn = open
            mode = "r"

        # First pass: count lines for progress bar
        if show_progress:
            logger.debug("Counting lines for progress tracking")
            with open_fn(self.labels_path, mode, encoding="utf-8") as f:
                total_lines = sum(1 for _ in f) - 1  # Subtract header
        else:
            total_lines = None

        # Parse CSV with pandas
        chunks = []
        with tqdm(total=total_lines, desc="Parsing labels", disable=not show_progress) as pbar:
            for chunk in pd.read_csv(self.labels_path, chunksize=chunk_size, low_memory=False):
                chunks.append(chunk)
                if show_progress:
                    pbar.update(len(chunk))

        self.labels_df = pd.concat(chunks, ignore_index=True)
        self.n_labels = len(self.labels_df)

        # Standardize column names
        self._standardize_columns()

        logger.info(f"Parsed {self.n_labels:,} labels successfully")
        return self.labels_df

    def _standardize_columns(self) -> None:
        """Standardize column names across different label formats."""
        if self.labels_df is None:
            return

        # Common column name mappings
        column_mappings = {
            "pt_root_id": "root_id",
            "root_id": "root_id",
            "tag": "label",
            "cell_type": "label",
            "processed_labels": "label",  # FlyWire standard format
            "pt_position_x": "position_x",
            "pt_position_y": "position_y",
            "pt_position_z": "position_z",
            "pt_supervoxel_id": "supervoxel_id",
        }

        # Apply mappings
        for old_name, new_name in column_mappings.items():
            if old_name in self.labels_df.columns and new_name not in self.labels_df.columns:
                self.labels_df = self.labels_df.rename(columns={old_name: new_name})

        # Ensure required columns exist
        if "root_id" not in self.labels_df.columns:
            logger.warning("No 'root_id' column found in labels")
        if "label" not in self.labels_df.columns:
            logger.warning("No 'label' column found in labels")

    def search_patterns(
        self, patterns: List[str], case_sensitive: bool = False
    ) -> Dict[str, List[CellLabel]]:
        """
        Search for cells matching specific patterns.

        Args:
            patterns: List of regex patterns or exact strings to match
            case_sensitive: Whether to perform case-sensitive search

        Returns:
            Dictionary mapping pattern to list of matching CellLabel objects

        Example:
            >>> parser = CommunityLabelsParser("labels.csv.gz")
            >>> parser.parse()
            >>> results = parser.search_patterns(["Or42b", "Or47b"])
            >>> for pattern, cells in results.items():
            ...     print(f"{pattern}: {len(cells)} cells")
        """
        if self.labels_df is None:
            raise RuntimeError("Must call parse() before searching")

        results: Dict[str, List[CellLabel]] = {}

        for pattern in patterns:
            matching_cells = []

            # Try regex match
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(pattern, flags)

                mask = (
                    self.labels_df["label"]
                    .astype(str)
                    .str.contains(regex.pattern, case=case_sensitive, regex=True, na=False)
                )
                matching_rows = self.labels_df[mask]

                for _, row in matching_rows.iterrows():
                    cell = self._row_to_cell_label(row)
                    matching_cells.append(cell)

                results[pattern] = matching_cells
                logger.debug(f"Pattern '{pattern}': found {len(matching_cells)} cells")

            except Exception as e:
                logger.error(f"Error searching pattern '{pattern}': {e}")
                results[pattern] = []

        return results

    def extract_olfactory_cells(self, receptor_types: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Extract all olfactory receptor cells from labels.

        Args:
            receptor_types: Specific receptor types to extract (or, ir, gr)
                          If None, extracts all types

        Returns:
            DataFrame with olfactory receptor cells

        Example:
            >>> parser = CommunityLabelsParser("labels.csv.gz")
            >>> parser.parse()
            >>> or_cells = parser.extract_olfactory_cells(["or"])
            >>> print(f"Found {len(or_cells)} Or neurons")
        """
        if self.labels_df is None:
            raise RuntimeError("Must call parse() before extracting")

        if receptor_types is None:
            receptor_types = ["or", "ir", "gr"]

        all_matches = []
        for receptor_type in receptor_types:
            if receptor_type.lower() not in self.OLFACTORY_PATTERNS:
                logger.warning(f"Unknown receptor type: {receptor_type}")
                continue

            pattern = self.OLFACTORY_PATTERNS[receptor_type.lower()]
            mask = (
                self.labels_df["label"]
                .astype(str)
                .str.contains(pattern, case=False, regex=True, na=False)
            )
            matches = self.labels_df[mask].copy()
            matches["receptor_type"] = receptor_type.upper()
            all_matches.append(matches)

        if not all_matches:
            return pd.DataFrame()

        result = pd.concat(all_matches, ignore_index=True)
        logger.info(f"Extracted {len(result)} olfactory cells ({', '.join(receptor_types)})")
        return result

    def find_by_root_id(self, root_ids: List[str]) -> List[CellLabel]:
        """
        Find cells by their FlyWire root IDs.

        Args:
            root_ids: List of root IDs to search for

        Returns:
            List of matching CellLabel objects

        Example:
            >>> parser = CommunityLabelsParser("labels.csv.gz")
            >>> parser.parse()
            >>> cells = parser.find_by_root_id(["720575940610453042"])
            >>> print(cells[0].label)
        """
        if self.labels_df is None:
            raise RuntimeError("Must call parse() before searching")

        # Convert to string for comparison
        root_ids_str = [str(rid) for rid in root_ids]

        mask = self.labels_df["root_id"].astype(str).isin(root_ids_str)
        matching_rows = self.labels_df[mask]

        results = []
        for _, row in matching_rows.iterrows():
            cell = self._row_to_cell_label(row)
            results.append(cell)

        return results

    def get_unique_receptors(self) -> Dict[str, int]:
        """
        Get count of unique receptor types in the dataset.

        Returns:
            Dictionary mapping receptor name to count

        Example:
            >>> parser = CommunityLabelsParser("labels.csv.gz")
            >>> parser.parse()
            >>> receptors = parser.get_unique_receptors()
            >>> print(f"Or42b: {receptors.get('Or42b', 0)} cells")
        """
        if self.labels_df is None:
            raise RuntimeError("Must call parse() before analysis")

        receptor_counts: Dict[str, int] = {}

        # Search for all olfactory patterns
        for pattern_type, pattern in self.OLFACTORY_PATTERNS.items():
            if pattern_type.startswith("sensillum"):
                continue  # Skip sensillum patterns for receptor counts

            mask = (
                self.labels_df["label"]
                .astype(str)
                .str.contains(pattern, case=False, regex=True, na=False)
            )
            matches = self.labels_df[mask]["label"].astype(str)

            # Extract receptor names
            regex = re.compile(pattern, re.IGNORECASE)
            for label in matches:
                match = regex.search(label)
                if match:
                    receptor_name = match.group()
                    receptor_counts[receptor_name] = receptor_counts.get(receptor_name, 0) + 1

        return dict(sorted(receptor_counts.items()))

    def _row_to_cell_label(self, row: pd.Series) -> CellLabel:
        """Convert DataFrame row to CellLabel object."""
        return CellLabel(
            root_id=str(row.get("root_id", "")),
            label=str(row.get("label", "")),
            supervoxel_id=(
                str(row.get("supervoxel_id", "")) if pd.notna(row.get("supervoxel_id")) else None
            ),
            position_x=float(row.get("position_x")) if pd.notna(row.get("position_x")) else None,
            position_y=float(row.get("position_y")) if pd.notna(row.get("position_y")) else None,
            position_z=float(row.get("position_z")) if pd.notna(row.get("position_z")) else None,
        )

    def export_filtered(self, output_path: str, receptor_types: Optional[List[str]] = None) -> None:
        """
        Export filtered olfactory cells to CSV.

        Args:
            output_path: Path for output CSV file
            receptor_types: Receptor types to include (or, ir, gr)

        Example:
            >>> parser = CommunityLabelsParser("labels.csv.gz")
            >>> parser.parse()
            >>> parser.export_filtered("olfactory_cells.csv", ["or"])
        """
        filtered_df = self.extract_olfactory_cells(receptor_types)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        filtered_df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(filtered_df)} cells to {output_path}")
