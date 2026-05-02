"""Tests for :mod:`door_toolkit.atlas_align.io.roi_exporter` and
:mod:`door_toolkit.atlas_align.io.pose_io`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from door_toolkit.atlas_align.core.iou_matcher import (
    Assignment,
    AssignmentResult,
)
from door_toolkit.atlas_align.core.volume_transform import Pose
from door_toolkit.atlas_align.io.pose_io import (
    file_sha256,
    load_pose,
    save_pose,
)
from door_toolkit.atlas_align.io.roi_exporter import (
    CSV_FIELDS,
    export_assignments_csv,
    export_roi_zip,
)
from door_toolkit.atlas_align.io.roi_loader import ROI, load_rois


def _make_roi(name: str, x0: float, y0: float, size: float = 10.0) -> ROI:
    xs = np.array([x0, x0 + size, x0 + size, x0], dtype=np.float32)
    ys = np.array([y0, y0, y0 + size, y0 + size], dtype=np.float32)
    return ROI(name=name, x=xs, y=ys, roi_type="polygon")


def _make_result(
    n: int = 3,
) -> tuple[list[ROI], AssignmentResult]:
    rois = [_make_roi(f"r{i}", i * 20.0, 0.0) for i in range(n)]
    assignments = [
        Assignment(
            roi_index=0,
            roi_name="r0",
            glomerulus_id=1,
            glomerulus_name="DM1",
            iou=0.85,
            alternates=[(2, "DM2", 0.10)],
            above_threshold=True,
        ),
        Assignment(
            roi_index=1,
            roi_name="r1",
            glomerulus_id=2,
            glomerulus_name="DM2",
            iou=0.60,
            alternates=[(1, "DM1", 0.20), (3, "DA1", 0.05)],
            above_threshold=True,
        ),
        Assignment(
            roi_index=2,
            roi_name="r2",
            glomerulus_id=0,
            glomerulus_name="UNK_r2",
            iou=0.05,
            alternates=[],
            above_threshold=False,
        ),
    ]
    result = AssignmentResult(
        assignments=assignments[:n],
        iou_matrix=np.zeros((n, 3), dtype=np.float32),
        glomerulus_ids=[1, 2, 3],
        mean_iou=0.5,
        n_above_threshold=2 if n == 3 else 0,
        n_below_threshold=1 if n == 3 else 0,
        iterations=1,
    )
    return rois[:n], result


@pytest.mark.atlas_align
class TestROIExporter:

    def test_roi_zip_roundtrip(self, tmp_path: Path) -> None:
        rois, result = _make_result()
        out = tmp_path / "out.zip"
        export_roi_zip(out, rois, result)
        assert out.is_file()
        roundtripped = load_rois(out)
        names = roundtripped.names
        assert "DM1" in names
        assert "DM2" in names
        # Unassigned → prefixed UNK_
        assert any(n.startswith("UNK_") for n in names)

    def test_unique_names_when_conflict(self, tmp_path: Path) -> None:
        rois, _ = _make_result(n=2)
        # Force both ROIs to be "DM1" — exporter must de-duplicate.
        from door_toolkit.atlas_align.core.iou_matcher import (
            Assignment, AssignmentResult,
        )

        result = AssignmentResult(
            assignments=[
                Assignment(0, "a", 1, "DM1", 0.9, [], True),
                Assignment(1, "b", 1, "DM1", 0.5, [], True),
            ],
            iou_matrix=np.zeros((2, 1), dtype=np.float32),
            glomerulus_ids=[1],
            mean_iou=0.7,
            n_above_threshold=2,
            n_below_threshold=0,
            iterations=1,
        )
        out = tmp_path / "dupes.zip"
        export_roi_zip(out, rois, result)
        rt = load_rois(out)
        assert len(rt) == 2
        assert len(set(rt.names)) == 2  # unique

    def test_length_mismatch_raises(self, tmp_path: Path) -> None:
        rois, result = _make_result(n=3)
        with pytest.raises(ValueError):
            export_roi_zip(tmp_path / "bad.zip", rois[:2], result)


@pytest.mark.atlas_align
class TestAssignmentsCSV:

    def test_csv_has_all_rois(self, tmp_path: Path) -> None:
        rois, result = _make_result()
        csv_path = tmp_path / "assignments.csv"
        export_assignments_csv(csv_path, result)
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(result.assignments)
        assert list(rows[0].keys()) == CSV_FIELDS

    def test_csv_records_unassigned(self, tmp_path: Path) -> None:
        _, result = _make_result()
        out = tmp_path / "assignments.csv"
        export_assignments_csv(out, result)
        with out.open() as fh:
            rows = list(csv.DictReader(fh))
        last = rows[-1]
        assert last["assigned_glomerulus"] == ""
        assert last["above_threshold"] == "0"


@pytest.mark.atlas_align
class TestPoseIO:

    def test_save_and_load_roundtrip(self, tmp_path: Path) -> None:
        pose = Pose(tx=3.5, rx=10.0, sy=1.25, flip_x=True)
        out = tmp_path / "pose.json"
        save_pose(out, pose, threshold=0.4, atlas_hash="abc", reference_hash="xyz")
        loaded_pose, meta = load_pose(out)
        assert loaded_pose == pose
        assert meta["threshold"] == pytest.approx(0.4)
        assert meta["atlas_hash"] == "abc"
        assert meta["reference_hash"] == "xyz"
        assert meta["timestamp_utc"]

    def test_load_missing_flags_defaults(self, tmp_path: Path) -> None:
        out = tmp_path / "pose.json"
        out.write_text(json.dumps({"tx": 1.0, "threshold": 0.2}))
        pose, meta = load_pose(out)
        assert pose.tx == pytest.approx(1.0)
        # Defaults applied for everything else.
        assert pose.ty == 0.0
        assert meta["atlas_hash"] is None

    def test_file_sha256(self, tmp_path: Path) -> None:
        f = tmp_path / "x.bin"
        f.write_bytes(b"hello")
        import hashlib

        assert file_sha256(f) == hashlib.sha256(b"hello").hexdigest()
