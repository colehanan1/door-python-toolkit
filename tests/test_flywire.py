"""Tests for FlyWire integration modules."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from door_toolkit.flywire.community_labels import CellLabel, CommunityLabelsParser
from door_toolkit.flywire.mapper import FlyWireMapper, ReceptorMapping, SpatialMap
from door_toolkit.flywire.spatial_analysis import (
    SpatialActivationMap,
    compare_spatial_maps,
    calculate_spatial_overlap,
)


@pytest.fixture
def mock_community_labels():
    """Create mock community labels CSV file."""
    data = {
        "root_id": ["123", "456", "789", "101112"],
        "label": ["Or42b neuron", "Or47b ORN", "Or42b cell", "antennal lobe PN"],
        "position_x": [1000.0, 2000.0, 1500.0, 3000.0],
        "position_y": [1000.0, 2000.0, 1500.0, 3000.0],
        "position_z": [1000.0, 2000.0, 1500.0, 3000.0],
    }
    df = pd.DataFrame(data)

    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        df.to_csv(f, index=False)
        temp_path = f.name

    yield temp_path

    # Cleanup
    Path(temp_path).unlink()


class TestCommunityLabelsParser:
    """Tests for CommunityLabelsParser class."""

    def test_initialization(self, mock_community_labels):
        """Test parser initialization."""
        parser = CommunityLabelsParser(mock_community_labels)
        assert parser.labels_path.exists()
        assert parser.labels_df is None

    def test_parse(self, mock_community_labels):
        """Test parsing community labels."""
        parser = CommunityLabelsParser(mock_community_labels)
        df = parser.parse(show_progress=False)

        assert len(df) == 4
        assert "root_id" in df.columns
        assert "label" in df.columns

    def test_search_patterns(self, mock_community_labels):
        """Test pattern searching."""
        parser = CommunityLabelsParser(mock_community_labels)
        parser.parse(show_progress=False)

        results = parser.search_patterns(["Or42b"])
        assert "Or42b" in results
        assert len(results["Or42b"]) == 2  # Two Or42b cells

    def test_extract_olfactory_cells(self, mock_community_labels):
        """Test olfactory cell extraction."""
        parser = CommunityLabelsParser(mock_community_labels)
        parser.parse(show_progress=False)

        or_cells = parser.extract_olfactory_cells(["or"])
        assert len(or_cells) > 0
        assert any("Or42b" in str(label) for label in or_cells["label"])

    def test_get_unique_receptors(self, mock_community_labels):
        """Test unique receptor counting."""
        parser = CommunityLabelsParser(mock_community_labels)
        parser.parse(show_progress=False)

        receptors = parser.get_unique_receptors()
        assert isinstance(receptors, dict)
        assert len(receptors) > 0


class TestFlyWireMapper:
    """Tests for FlyWireMapper class."""

    def test_initialization(self, mock_community_labels):
        """Test mapper initialization."""
        mapper = FlyWireMapper(mock_community_labels, auto_parse=False)
        assert mapper.labels_parser is not None

    def test_find_receptor_cells(self, mock_community_labels):
        """Test finding receptor cells."""
        mapper = FlyWireMapper(mock_community_labels, auto_parse=True)
        cells = mapper.find_receptor_cells("Or42b")

        assert len(cells) == 2
        assert all("root_id" in cell for cell in cells)
        assert all("label" in cell for cell in cells)

    def test_receptor_mapping(self):
        """Test ReceptorMapping dataclass."""
        mapping = ReceptorMapping(
            receptor_name="Or42b",
            flywire_root_ids=["123", "456"],
            cell_count=2,
            mean_position=(1000.0, 1000.0, 1000.0),
            confidence_score=0.9,
        )

        assert mapping.receptor_name == "Or42b"
        assert mapping.cell_count == 2
        assert len(mapping.flywire_root_ids) == 2

        # Test serialization
        mapping_dict = mapping.to_dict()
        assert isinstance(mapping_dict, dict)
        assert "receptor_name" in mapping_dict


class TestSpatialAnalysis:
    """Tests for spatial analysis functions."""

    def test_spatial_activation_map_creation(self):
        """Test creating spatial activation map."""
        points = [
            (100.0, 100.0, 100.0, 0.5),
            (200.0, 200.0, 200.0, 0.8),
            (150.0, 150.0, 150.0, 0.6),
        ]
        receptor_data = {"Or42b": 0.7, "Or47b": 0.5}

        spatial_map = SpatialActivationMap(
            odorant_name="ethyl butyrate",
            points=points,
            receptor_data=receptor_data,
        )

        assert spatial_map.n_points == 3
        assert spatial_map.mean_activation > 0
        assert spatial_map.max_activation == 0.8

    def test_spatial_map_centroid(self):
        """Test centroid calculation."""
        points = [
            (0.0, 0.0, 0.0, 1.0),
            (100.0, 100.0, 100.0, 1.0),
        ]

        spatial_map = SpatialActivationMap(
            odorant_name="test",
            points=points,
            receptor_data={},
        )

        centroid = spatial_map.get_centroid()
        assert len(centroid) == 3
        # Should be midpoint
        assert centroid[0] == pytest.approx(50.0, abs=1.0)

    def test_filter_by_threshold(self):
        """Test threshold filtering."""
        points = [
            (100.0, 100.0, 100.0, 0.5),
            (200.0, 200.0, 200.0, 0.8),
            (150.0, 150.0, 150.0, 0.3),
        ]

        spatial_map = SpatialActivationMap(
            odorant_name="test",
            points=points,
            receptor_data={"Or42b": 0.8, "Or47b": 0.3},
        )

        filtered = spatial_map.filter_by_threshold(0.4)
        assert filtered.n_points == 2  # Only two points above 0.4

    def test_compare_spatial_maps(self):
        """Test spatial map comparison."""
        map1 = SpatialActivationMap(
            odorant_name="odor1",
            points=[(100.0, 100.0, 100.0, 0.5)],
            receptor_data={"Or42b": 0.5},
        )

        map2 = SpatialActivationMap(
            odorant_name="odor2",
            points=[(200.0, 200.0, 200.0, 0.8)],
            receptor_data={"Or47b": 0.8},
        )

        comparison = compare_spatial_maps([map1, map2])
        assert len(comparison) == 2
        assert "odorant" in comparison.columns

    def test_calculate_spatial_overlap(self):
        """Test spatial overlap calculation."""
        map1 = SpatialActivationMap(
            odorant_name="odor1",
            points=[(100.0, 100.0, 100.0, 0.5)],
            receptor_data={},
        )

        map2 = SpatialActivationMap(
            odorant_name="odor2",
            points=[(105.0, 105.0, 105.0, 0.8)],  # Very close
            receptor_data={},
        )

        overlap = calculate_spatial_overlap(map1, map2, radius=10.0)
        assert overlap >= 0.0
        assert overlap <= 1.0


def test_cell_label():
    """Test CellLabel dataclass."""
    cell = CellLabel(
        root_id="123456",
        label="Or42b neuron",
        position_x=1000.0,
        position_y=2000.0,
        position_z=3000.0,
    )

    assert cell.root_id == "123456"
    assert cell.label == "Or42b neuron"

    coords = cell.coordinates
    assert coords is not None
    assert len(coords) == 3
    assert coords[0] == 1000.0
