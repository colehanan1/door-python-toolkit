"""Tests for the atlas_align mock-flywire builder path.

Exercises :func:`door_toolkit.atlas_align.atlas_builder.build_labelmap`
in ``mock_flywire=True`` mode. The real FlyWire fetch path is never
run in CI (that would require network + CAVE auth + minutes of
skeleton I/O).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

from door_toolkit.atlas_align.atlas_builder import build_labelmap
from door_toolkit.atlas_align.atlas_builder.build_atlas import (
    MOCK_SHAPE,
    _MOCK_GLOMERULI,
    main as builder_cli_main,
)


@pytest.mark.atlas_align
class TestBuilderMockPath:
    """Smoke + correctness tests for the deterministic mock atlas."""

    def test_build_labelmap_returns_paths(self, tmp_path: Path) -> None:
        result = build_labelmap(
            output_dir=tmp_path, mock_flywire=True, save_meshes=False
        )
        assert result["labelmap"].is_file()
        assert result["labels"].is_file()
        assert result["manifest"].is_file()
        assert result["qc_dir"].is_dir()
        assert "meshes_dir" not in result

    def test_build_labelmap_with_meshes(self, tmp_path: Path) -> None:
        result = build_labelmap(
            output_dir=tmp_path, mock_flywire=True, save_meshes=True
        )
        assert result["meshes_dir"].is_dir()
        ply_files = sorted(result["meshes_dir"].glob("*.ply"))
        assert len(ply_files) == len(_MOCK_GLOMERULI)

    def test_labelmap_is_uint16_and_shape_matches(self, tmp_path: Path) -> None:
        build_labelmap(output_dir=tmp_path, mock_flywire=True)
        lm = tifffile.imread(tmp_path / "flywire_al_labelmap.tif")
        assert lm.dtype == np.uint16
        assert lm.shape == MOCK_SHAPE

    def test_labels_json_has_eight_entries(self, tmp_path: Path) -> None:
        build_labelmap(output_dir=tmp_path, mock_flywire=True)
        labels = json.loads(
            (tmp_path / "flywire_al_labels.json").read_text()
        )
        assert len(labels) == len(_MOCK_GLOMERULI)
        assert set(labels.values()) == {g[0] for g in _MOCK_GLOMERULI}
        # keys are string integers starting at 1
        numeric_keys = sorted(int(k) for k in labels.keys())
        assert numeric_keys == list(range(1, len(_MOCK_GLOMERULI) + 1))

    def test_labelmap_contents_match_labels_json(self, tmp_path: Path) -> None:
        build_labelmap(output_dir=tmp_path, mock_flywire=True)
        lm = tifffile.imread(tmp_path / "flywire_al_labelmap.tif")
        labels = json.loads(
            (tmp_path / "flywire_al_labels.json").read_text()
        )
        present = {int(v) for v in np.unique(lm) if v != 0}
        declared = {int(k) for k in labels.keys()}
        assert present == declared

    def test_manifest_is_valid(self, tmp_path: Path) -> None:
        build_labelmap(output_dir=tmp_path, mock_flywire=True)
        manifest = json.loads(
            (tmp_path / "build_manifest.json").read_text()
        )
        assert manifest["mock"] is True
        assert manifest["template"] == "MOCK"
        assert manifest["n_glomeruli"] == len(_MOCK_GLOMERULI)
        assert len(manifest["labelmap_sha256"]) == 64
        assert manifest["template_shape_zyx"] == list(MOCK_SHAPE)
        assert "generated_utc" in manifest

    def test_qc_mips_written(self, tmp_path: Path) -> None:
        build_labelmap(output_dir=tmp_path, mock_flywire=True)
        qc = tmp_path / "qc"
        assert (qc / "mip_z.png").is_file()
        assert (qc / "mip_y.png").is_file()
        assert (qc / "mip_x.png").is_file()

    def test_mock_build_is_deterministic(self, tmp_path: Path) -> None:
        """Two independent mock builds must produce byte-identical labelmaps."""
        out_a = tmp_path / "a"
        out_b = tmp_path / "b"
        build_labelmap(output_dir=out_a, mock_flywire=True)
        build_labelmap(output_dir=out_b, mock_flywire=True)
        lm_a = (out_a / "flywire_al_labelmap.tif").read_bytes()
        lm_b = (out_b / "flywire_al_labelmap.tif").read_bytes()
        # The TIFF header carries metadata, so compare the decoded array
        arr_a = tifffile.imread(out_a / "flywire_al_labelmap.tif")
        arr_b = tifffile.imread(out_b / "flywire_al_labelmap.tif")
        np.testing.assert_array_equal(arr_a, arr_b)

    def test_mock_build_runs_in_under_30_seconds(self, tmp_path: Path) -> None:
        t0 = time.time()
        build_labelmap(output_dir=tmp_path, mock_flywire=True)
        assert time.time() - t0 < 30.0

    def test_cli_entry_point(self, tmp_path: Path) -> None:
        """Invoke :func:`main` as if from a shell; expect zero exit."""
        rc = builder_cli_main(
            ["--mock-flywire", "--output-dir", str(tmp_path)]
        )
        assert rc == 0
        assert (tmp_path / "flywire_al_labelmap.tif").is_file()

    def test_each_glomerulus_has_voxels(self, tmp_path: Path) -> None:
        """Every mock glomerulus must actually be rasterized (no silent drops)."""
        build_labelmap(output_dir=tmp_path, mock_flywire=True)
        lm = tifffile.imread(tmp_path / "flywire_al_labelmap.tif")
        labels = json.loads(
            (tmp_path / "flywire_al_labels.json").read_text()
        )
        for label_idx_str in labels:
            label_idx = int(label_idx_str)
            count = int((lm == label_idx).sum())
            assert count > 0, f"label {label_idx} ({labels[label_idx_str]}) empty"
