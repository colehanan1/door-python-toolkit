"""
FlyWire-DoOR Mapper
===================

Core integration class for mapping DoOR receptor data to FlyWire community labels.

This module enables quantitative mapping between DoOR odorant response profiles
and FlyWire neural connectivity data.
"""

import json
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from door_toolkit.flywire.community_labels import CellLabel, CommunityLabelsParser
from door_toolkit.utils import load_response_matrix, load_odor_metadata

logger = logging.getLogger(__name__)


@dataclass
class ReceptorMapping:
    """Mapping between DoOR receptor and FlyWire cells."""

    receptor_name: str
    flywire_root_ids: List[str]
    cell_count: int
    mean_position: Optional[Tuple[float, float, float]] = None
    confidence_score: float = 1.0

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "receptor_name": self.receptor_name,
            "flywire_root_ids": self.flywire_root_ids,
            "cell_count": self.cell_count,
            "mean_position": self.mean_position,
            "confidence_score": self.confidence_score,
        }


@dataclass
class SpatialMap:
    """3D spatial activation map for an odorant."""

    odorant_name: str
    receptor_activations: Dict[str, float]  # receptor -> response strength
    spatial_points: List[Tuple[float, float, float, float]]  # (x, y, z, intensity)
    total_cells: int

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "odorant_name": self.odorant_name,
            "receptor_activations": self.receptor_activations,
            "spatial_points": [
                {"x": x, "y": y, "z": z, "intensity": i} for x, y, z, i in self.spatial_points
            ],
            "total_cells": self.total_cells,
        }


class FlyWireMapper:
    """
    Map DoOR receptor response data to FlyWire neural connectivity.

    This class integrates DoOR odorant-receptor response profiles with FlyWire
    community labels to enable spatial and quantitative analysis of olfactory
    pathway activation.

    Attributes:
        labels_parser: Community labels parser instance
        door_cache_path: Path to DoOR cache directory
        receptor_mappings: Mapping from receptor names to FlyWire cells

    Example:
        >>> mapper = FlyWireMapper("processed_labels.csv.gz")
        >>> mapper.parse_labels()
        >>> or42b_cells = mapper.find_receptor_cells("Or42b")
        >>> print(f"Found {len(or42b_cells)} Or42b neurons")
    """

    def __init__(
        self,
        community_labels_path: str,
        door_cache_path: Optional[str] = None,
        auto_parse: bool = False,
    ):
        """
        Initialize FlyWire-DoOR mapper.

        Args:
            community_labels_path: Path to FlyWire community labels (CSV/CSV.GZ)
            door_cache_path: Optional path to DoOR cache directory
            auto_parse: Automatically parse labels on initialization

        Raises:
            FileNotFoundError: If community labels file not found
        """
        self.labels_parser = CommunityLabelsParser(community_labels_path)
        self.door_cache_path = Path(door_cache_path) if door_cache_path else None
        self.receptor_mappings: Dict[str, ReceptorMapping] = {}

        if auto_parse:
            self.parse_labels()

    def parse_labels(self, show_progress: bool = True) -> None:
        """
        Parse community labels file.

        Args:
            show_progress: Show progress bar during parsing
        """
        self.labels_parser.parse(show_progress=show_progress)
        logger.info(f"Parsed {self.labels_parser.n_labels:,} community labels")

    def find_receptor_cells(self, receptor_name: str) -> List[Dict]:
        """
        Find all FlyWire cells expressing a specific receptor.

        Args:
            receptor_name: Receptor to search for (e.g., "Or42b", "Ir75a")

        Returns:
            List of cell dictionaries with root_id, label, and position

        Example:
            >>> mapper = FlyWireMapper("processed_labels.csv.gz")
            >>> mapper.parse_labels()
            >>> cells = mapper.find_receptor_cells("Or42b")
            >>> for cell in cells:
            ...     print(f"Root ID: {cell['root_id']}, Label: {cell['label']}")
        """
        if self.labels_parser.labels_df is None:
            raise RuntimeError("Must call parse_labels() first")

        results = self.labels_parser.search_patterns([receptor_name])
        cells = results.get(receptor_name, [])

        # Convert to dictionary format
        cell_dicts = []
        for cell in cells:
            cell_dict = {
                "root_id": cell.root_id,
                "label": cell.label,
                "supervoxel_id": cell.supervoxel_id,
            }
            if cell.coordinates:
                cell_dict["position"] = {
                    "x": cell.position_x,
                    "y": cell.position_y,
                    "z": cell.position_z,
                }
            cell_dicts.append(cell_dict)

        logger.info(f"Found {len(cell_dicts)} cells for receptor {receptor_name}")
        return cell_dicts

    def map_door_to_flywire(
        self, door_cache_path: Optional[str] = None
    ) -> Dict[str, ReceptorMapping]:
        """
        Create comprehensive mapping between DoOR receptors and FlyWire cells.

        Args:
            door_cache_path: Path to DoOR cache (uses self.door_cache_path if None)

        Returns:
            Dictionary mapping receptor names to ReceptorMapping objects

        Example:
            >>> mapper = FlyWireMapper("labels.csv.gz", "door_cache")
            >>> mapper.parse_labels()
            >>> mappings = mapper.map_door_to_flywire()
            >>> print(f"Mapped {len(mappings)} receptors")
        """
        if self.labels_parser.labels_df is None:
            raise RuntimeError("Must call parse_labels() first")

        cache_path = door_cache_path or self.door_cache_path
        if not cache_path:
            raise ValueError("door_cache_path must be provided")

        # Load DoOR response matrix to get receptor list
        logger.info(f"Loading DoOR cache from {cache_path}")
        response_matrix = load_response_matrix(str(cache_path))
        door_receptors = response_matrix.columns.tolist()

        logger.info(f"Mapping {len(door_receptors)} DoOR receptors to FlyWire")

        mappings = {}
        found_count = 0
        total_cells = 0

        for receptor in tqdm(door_receptors, desc="Mapping receptors"):
            cells = self.find_receptor_cells(receptor)

            if cells:
                # Calculate mean position if coordinates available
                positions = [
                    (c["position"]["x"], c["position"]["y"], c["position"]["z"])
                    for c in cells
                    if "position" in c
                ]

                mean_position = None
                if positions:
                    mean_position = tuple(np.mean(positions, axis=0).tolist())

                mapping = ReceptorMapping(
                    receptor_name=receptor,
                    flywire_root_ids=[c["root_id"] for c in cells],
                    cell_count=len(cells),
                    mean_position=mean_position,
                    confidence_score=1.0,  # Could be refined based on label quality
                )

                mappings[receptor] = mapping
                found_count += 1
                total_cells += len(cells)

        self.receptor_mappings = mappings

        success_rate = (found_count / len(door_receptors)) * 100
        logger.info(
            f"Mapping complete: {found_count}/{len(door_receptors)} receptors "
            f"({success_rate:.1f}%), {total_cells} total cells"
        )

        return mappings

    def create_spatial_activation_map(
        self, odorant: str, door_cache_path: Optional[str] = None
    ) -> SpatialMap:
        """
        Create 3D spatial activation map for an odorant.

        Args:
            odorant: Odorant name (e.g., "ethyl butyrate")
            door_cache_path: Path to DoOR cache

        Returns:
            SpatialMap object with spatial activation data

        Example:
            >>> mapper = FlyWireMapper("labels.csv.gz", "door_cache")
            >>> mapper.parse_labels()
            >>> mapper.map_door_to_flywire()
            >>> spatial_map = mapper.create_spatial_activation_map("ethyl butyrate")
            >>> print(f"Activation at {spatial_map.total_cells} locations")
        """
        cache_path = door_cache_path or self.door_cache_path
        if not cache_path:
            raise ValueError("door_cache_path must be provided")

        if not self.receptor_mappings:
            logger.info("No receptor mappings found, creating mappings first")
            self.map_door_to_flywire(str(cache_path))

        # Load DoOR response for this odorant
        from door_toolkit.encoder import DoOREncoder

        encoder = DoOREncoder(str(cache_path), use_torch=False)

        try:
            response_vector = encoder.encode(odorant)
        except Exception as e:
            raise ValueError(f"Failed to encode odorant '{odorant}': {e}")

        # Build spatial map
        receptor_activations = {}
        spatial_points = []
        total_cells = 0

        for i, receptor in enumerate(encoder.receptor_names):
            response = float(response_vector[i])

            if np.isnan(response) or response == 0:
                continue

            receptor_activations[receptor] = response

            # Get FlyWire cells for this receptor
            if receptor in self.receptor_mappings:
                mapping = self.receptor_mappings[receptor]

                # Find cells with positions
                cells = self.find_receptor_cells(receptor)
                for cell in cells:
                    if "position" in cell:
                        pos = cell["position"]
                        spatial_points.append((pos["x"], pos["y"], pos["z"], response))
                        total_cells += 1

        spatial_map = SpatialMap(
            odorant_name=odorant,
            receptor_activations=receptor_activations,
            spatial_points=spatial_points,
            total_cells=total_cells,
        )

        logger.info(
            f"Created spatial map for '{odorant}': "
            f"{len(receptor_activations)} active receptors, "
            f"{total_cells} cells with positions"
        )

        return spatial_map

    def export_mapping(self, output_path: str, format: str = "json") -> None:
        """
        Export receptor mappings to file.

        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')

        Example:
            >>> mapper = FlyWireMapper("labels.csv.gz", "door_cache")
            >>> mapper.parse_labels()
            >>> mapper.map_door_to_flywire()
            >>> mapper.export_mapping("flywire_mapping.json")
        """
        if not self.receptor_mappings:
            raise RuntimeError("No mappings to export. Call map_door_to_flywire() first")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format.lower() == "json":
            export_data = {
                "mappings": {
                    name: mapping.to_dict() for name, mapping in self.receptor_mappings.items()
                },
                "summary": {
                    "total_receptors": len(self.receptor_mappings),
                    "total_cells": sum(m.cell_count for m in self.receptor_mappings.values()),
                },
            }

            with open(output_path, "w") as f:
                json.dump(export_data, f, indent=2)

        elif format.lower() == "csv":
            rows = []
            for name, mapping in self.receptor_mappings.items():
                for root_id in mapping.flywire_root_ids:
                    row = {
                        "receptor_name": name,
                        "root_id": root_id,
                        "cell_count": mapping.cell_count,
                    }
                    if mapping.mean_position:
                        row["mean_x"] = mapping.mean_position[0]
                        row["mean_y"] = mapping.mean_position[1]
                        row["mean_z"] = mapping.mean_position[2]
                    rows.append(row)

            df = pd.DataFrame(rows)
            df.to_csv(output_path, index=False)

        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'csv'")

        logger.info(f"Exported mappings to {output_path}")

    def get_mapping_statistics(self) -> Dict[str, any]:
        """
        Get statistics about receptor mappings.

        Returns:
            Dictionary with mapping statistics

        Example:
            >>> mapper = FlyWireMapper("labels.csv.gz", "door_cache")
            >>> mapper.parse_labels()
            >>> mapper.map_door_to_flywire()
            >>> stats = mapper.get_mapping_statistics()
            >>> print(f"Coverage: {stats['mapping_rate']:.1%}")
        """
        if not self.receptor_mappings:
            return {
                "total_receptors": 0,
                "mapped_receptors": 0,
                "mapping_rate": 0.0,
                "total_cells": 0,
                "mean_cells_per_receptor": 0.0,
            }

        total_cells = sum(m.cell_count for m in self.receptor_mappings.values())
        mapped_count = len(self.receptor_mappings)

        stats = {
            "total_receptors": mapped_count,
            "mapped_receptors": mapped_count,
            "mapping_rate": 1.0,  # Would need total DoOR receptor count for accurate rate
            "total_cells": total_cells,
            "mean_cells_per_receptor": total_cells / mapped_count if mapped_count > 0 else 0,
            "receptors_with_positions": sum(
                1 for m in self.receptor_mappings.values() if m.mean_position is not None
            ),
        }

        return stats
