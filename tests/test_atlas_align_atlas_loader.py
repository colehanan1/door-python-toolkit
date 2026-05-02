"""Tests for :mod:`door_toolkit.atlas_align.io.atlas_loader`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from door_toolkit.atlas_align.atlas_builder import build_labelmap
from door_toolkit.atlas_align.io.atlas_loader import (
    AtlasBundle,
    GRAYSCALE_FILENAME,
    LABELMAP_FILENAME,
    LABELS_FILENAME,
    load_atlas_bundle,
)


@pytest.fixture
def mock_atlas(tmp_path: Path) -> Path:
    build_labelmap(output_dir=tmp_path, mock_flywire=True)
    return tmp_path


@pytest.mark.atlas_align
class TestAtlasBundleLoading:

    def test_loads_labelmap_shape(self, mock_atlas: Path) -> None:
        bundle = load_atlas_bundle(mock_atlas)
        assert isinstance(bundle, AtlasBundle)
        assert bundle.labelmap.dtype == np.uint16
        assert bundle.labelmap.ndim == 3
        assert bundle.labelmap.shape == (30, 60, 60)

    def test_labels_parsed_with_int_keys(self, mock_atlas: Path) -> None:
        bundle = load_atlas_bundle(mock_atlas)
        assert len(bundle.labels) == 8
        assert all(isinstance(k, int) for k in bundle.labels)
        assert "DM1" in bundle.labels.values()

    def test_grayscale_is_synthesised_when_missing(self, mock_atlas: Path) -> None:
        bundle = load_atlas_bundle(mock_atlas)
        assert bundle.grayscale_synthesised is True
        assert bundle.grayscale.shape == bundle.labelmap.shape
        assert bundle.grayscale.dtype == np.float32
        # synthetic gray = (labelmap > 0) as float
        expected = (bundle.labelmap > 0).astype(np.float32)
        np.testing.assert_array_equal(bundle.grayscale, expected)

    def test_grayscale_file_is_preferred_when_present(
        self, mock_atlas: Path
    ) -> None:
        # Drop a fake grayscale TIF matching the labelmap shape.
        labelmap = tifffile.imread(mock_atlas / LABELMAP_FILENAME)
        fake_gray = np.random.default_rng(0).random(labelmap.shape).astype(
            np.float32
        )
        tifffile.imwrite(mock_atlas / GRAYSCALE_FILENAME, fake_gray)

        bundle = load_atlas_bundle(mock_atlas)
        assert bundle.grayscale_synthesised is False
        np.testing.assert_array_equal(bundle.grayscale, fake_gray)

    def test_grayscale_file_with_wrong_shape_falls_back(
        self, mock_atlas: Path
    ) -> None:
        wrong_gray = np.zeros((5, 5, 5), dtype=np.float32)
        tifffile.imwrite(mock_atlas / GRAYSCALE_FILENAME, wrong_gray)

        bundle = load_atlas_bundle(mock_atlas)
        assert bundle.grayscale_synthesised is True
        assert bundle.grayscale.shape == bundle.labelmap.shape

    def test_manifest_parsed(self, mock_atlas: Path) -> None:
        bundle = load_atlas_bundle(mock_atlas)
        assert bundle.manifest["mock"] is True
        assert bundle.manifest["template"] == "MOCK"

    def test_spacing_from_manifest_when_tif_has_none(
        self, mock_atlas: Path
    ) -> None:
        bundle = load_atlas_bundle(mock_atlas)
        assert bundle.spacing_um == (1.0, 1.0, 1.0)

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_atlas_bundle(tmp_path / "does_not_exist")

    def test_missing_labelmap_raises(self, mock_atlas: Path) -> None:
        (mock_atlas / LABELMAP_FILENAME).unlink()
        with pytest.raises(FileNotFoundError):
            load_atlas_bundle(mock_atlas)

    def test_missing_labels_json_raises(self, mock_atlas: Path) -> None:
        (mock_atlas / LABELS_FILENAME).unlink()
        with pytest.raises(FileNotFoundError):
            load_atlas_bundle(mock_atlas)

    def test_label_name_helper(self, mock_atlas: Path) -> None:
        bundle = load_atlas_bundle(mock_atlas)
        first_key = sorted(bundle.labels.keys())[0]
        assert bundle.label_name(first_key) == bundle.labels[first_key]
        assert bundle.label_name(9999) == "UNK_9999"

    def test_malformed_labels_json_raises(self, mock_atlas: Path) -> None:
        (mock_atlas / LABELS_FILENAME).write_text(json.dumps([1, 2, 3]))
        with pytest.raises(ValueError):
            load_atlas_bundle(mock_atlas)
