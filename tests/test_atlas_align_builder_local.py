"""Tests for the ``--local-flywire`` path of the atlas builder.

These tests write a tiny synthetic Schlegel-style dump (two CSVs and a
zipped SWC archive) and verify the builder reads it correctly. Bypasses
CAVE and the heavy ``xform_brain`` transform by mocking
:func:`door_toolkit.atlas_align.atlas_builder.build_atlas.xform_mesh_to_jrc2018f`
with an identity transform, and by monkey-patching
:data:`JRC2018F_SHAPE` / :data:`JRC2018F_SPACING_UM` to a small grid so
the rasterisation finishes in milliseconds.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from door_toolkit.atlas_align.atlas_builder import build_atlas


def _write_synthetic_skeleton(
    zf: zipfile.ZipFile, root_id: int, center_nm: tuple[float, float, float]
) -> None:
    """Emit a tiny 64-node SWC inside a 3 µm cube around ``center_nm``."""
    cx, cy, cz = center_nm
    rng = np.random.default_rng(seed=root_id % 1000)
    # 64 random points within ±1500 nm of centre = ±1.5 µm
    offsets = rng.uniform(-1500.0, 1500.0, size=(64, 3))
    pts = offsets + np.array([cx, cy, cz])

    lines = ["# synthetic swc"]
    for i, (x, y, z) in enumerate(pts, start=1):
        parent = -1 if i == 1 else 1
        lines.append(f"{i} 3 {x:.3f} {y:.3f} {z:.3f} 100 {parent}")
    zf.writestr(f"{root_id}.swc", "\n".join(lines) + "\n")


def _identity_xform(mesh):
    """Stand-in for ``xform_mesh_to_jrc2018f`` — drops the unit change."""
    import trimesh

    # Scale FAFB14 nanometres to microns so values fit inside the small
    # test raster grid.
    verts_um = np.asarray(mesh.vertices, dtype=np.float64) / 1000.0
    return trimesh.Trimesh(
        vertices=verts_um, faces=mesh.faces, process=False
    )


@pytest.fixture
def synthetic_local_dump(tmp_path: Path) -> Path:
    """Write classification.csv, consolidated_cell_types.csv, and a skeleton zip.

    Layout: 3 glomeruli (DA1, DM1, VA1), each with 2 uniglomerular PNs
    centred in distinct regions of the AL bbox.
    """
    data_dir = tmp_path / "flywire_local"
    data_dir.mkdir()

    # Classification: 6 PNs marked as ALPNs + a couple of distractors.
    classification_rows = []
    for rid in (100, 101, 200, 201, 300, 301):
        classification_rows.append({"root_id": rid, "class": "ALPN"})
    # distractor rows that must NOT appear in the output
    classification_rows.append({"root_id": 999, "class": "Kenyon_Cell"})
    pd.DataFrame(classification_rows).to_csv(
        data_dir / "classification.csv", index=False
    )

    # consolidated_cell_types: primary_type encodes glomerulus.
    cell_types_rows = [
        {"root_id": 100, "primary_type": "DA1_adPN", "additional_type(s)": ""},
        {"root_id": 101, "primary_type": "DA1_lPN",  "additional_type(s)": ""},
        {"root_id": 200, "primary_type": "DM1_adPN", "additional_type(s)": ""},
        {"root_id": 201, "primary_type": "DM1_lPN",  "additional_type(s)": ""},
        {"root_id": 300, "primary_type": "VA1_adPN", "additional_type(s)": ""},
        {"root_id": 301, "primary_type": "VA1_lPN",  "additional_type(s)": ""},
        # multiglomerular — must be filtered out
        {"root_id": 500, "primary_type": "M_vPNml53", "additional_type(s)": ""},
        # unnamed CB — must be filtered out
        {"root_id": 501, "primary_type": "CB1234",    "additional_type(s)": ""},
    ]
    pd.DataFrame(cell_types_rows).to_csv(
        data_dir / "consolidated_cell_types.csv", index=False
    )

    # Skeletons: centres in FAFB14 nanometres inside the AL bbox.
    # DA1 → (400_000, 150_000, 80_000), DM1 → (420_000, 160_000, 70_000), etc.
    centres = {
        100: (400_000, 150_000, 80_000),
        101: (400_000, 150_000, 80_000),
        200: (420_000, 160_000, 70_000),
        201: (420_000, 160_000, 70_000),
        300: (440_000, 170_000, 90_000),
        301: (440_000, 170_000, 90_000),
    }
    zip_path = data_dir / "sk_lod1_783_healed.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for rid, centre in centres.items():
            _write_synthetic_skeleton(zf, rid, centre)

    return data_dir


@pytest.mark.atlas_align
@pytest.mark.integration
class TestBuilderLocalPath:

    def test_annotation_parsing(self, synthetic_local_dump: Path) -> None:
        annotations = build_atlas.fetch_upn_annotations_local(
            synthetic_local_dump
        )
        names = {a.name for a in annotations}
        assert names == {"DA1", "DM1", "VA1"}
        for a in annotations:
            assert len(a.root_ids) == 2
            assert a.modality == "olfactory"

    def test_pointcloud_reads_local_swc(
        self, synthetic_local_dump: Path
    ) -> None:
        zip_path = synthetic_local_dump / "sk_lod1_783_healed.zip"
        bbox = np.array(
            [[380_000, 130_000, 50_000], [620_000, 280_000, 150_000]],
            dtype=np.float64,
        )
        with zipfile.ZipFile(zip_path) as zf:
            pts = build_atlas.fetch_dendrite_pointcloud_local(
                zf, [100, 101], bbox
            )
        assert pts.shape[1] == 3
        # 2 PNs × 64 nodes each, all inside bbox → 128 points
        assert len(pts) == 128

    def test_build_labelmap_end_to_end(
        self, synthetic_local_dump: Path, tmp_path: Path, monkeypatch
    ) -> None:
        # Replace xform + large JRC2018F grid with something fast.
        monkeypatch.setattr(
            build_atlas, "xform_mesh_to_jrc2018f", _identity_xform
        )
        monkeypatch.setattr(build_atlas, "JRC2018F_SHAPE", (200, 300, 500))
        monkeypatch.setattr(build_atlas, "JRC2018F_SPACING_UM", (1.0, 1.0, 1.0))

        out = tmp_path / "atlas_local"
        result = build_atlas.build_labelmap(
            output_dir=out,
            local_flywire=synthetic_local_dump,
            alpha=5000.0,  # looser for tiny synthetic cluster
        )
        assert result["labelmap"].is_file()
        labels = result["labels"].read_text()
        assert "DA1" in labels and "DM1" in labels and "VA1" in labels

        manifest = result["manifest"].read_text()
        assert '"local": true' in manifest
        assert '"mock": false' in manifest

    def test_mock_and_local_are_mutually_exclusive(
        self, synthetic_local_dump: Path, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError):
            build_atlas.build_labelmap(
                output_dir=tmp_path / "x",
                mock_flywire=True,
                local_flywire=synthetic_local_dump,
            )

    def test_missing_local_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            build_atlas.build_labelmap(
                output_dir=tmp_path / "x",
                local_flywire=tmp_path / "does-not-exist",
            )


@pytest.mark.atlas_align
class TestParseGlomerulus:
    """Direct unit tests for the primary_type parser."""

    @pytest.mark.parametrize(
        "primary_type, expected",
        [
            ("DA1_lPN", "DA1"),
            ("DA1_adPN", "DA1"),
            ("VA1v_adPN", "VA1v"),
            ("DL2d_adPN", "DL2d"),
            ("VM5d_adPN", "VM5d"),
            ("M_vPNml53", None),
            ("CB1296", None),
            ("VP5+_l2PN", None),  # multi-VP, we deliberately skip
            ("", None),
            (None, None),
            (float("nan"), None),
        ],
    )
    def test_parser_cases(self, primary_type, expected) -> None:
        assert build_atlas._parse_glomerulus(primary_type) == expected
