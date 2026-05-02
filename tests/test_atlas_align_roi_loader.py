"""Tests for :mod:`door_toolkit.atlas_align.io.roi_loader`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from door_toolkit.atlas_align.io.roi_loader import ROI, ROISet, load_rois


def _write_sample_zip(path: Path, rois_xy: list[tuple[str, np.ndarray]]) -> None:
    """Helper: build a FIJI-compatible ROI zip from (name, (N,2) xy) tuples."""
    from roifile import ImagejRoi, roiwrite

    rois = []
    for name, xy in rois_xy:
        r = ImagejRoi.frompoints(xy.tolist())
        r.name = name
        rois.append(r)
    roiwrite(str(path), rois, mode="w")


@pytest.fixture
def sample_zip(tmp_path: Path) -> Path:
    zip_path = tmp_path / "rois.zip"
    _write_sample_zip(
        zip_path,
        [
            ("DM1_roi", np.array([[10, 10], [30, 10], [30, 30], [10, 30]])),
            ("DA1_roi", np.array([[40, 40], [60, 40], [60, 60], [40, 60]])),
            ("triangle", np.array([[70, 70], [90, 70], [80, 90]])),
        ],
    )
    return zip_path


@pytest.mark.atlas_align
class TestROILoader:

    def test_load_roi_count(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        assert isinstance(roiset, ROISet)
        assert len(roiset) == 3

    def test_roi_names_preserved(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        assert roiset.names == ["DM1_roi", "DA1_roi", "triangle"]

    def test_roi_coordinates(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        roi = roiset[0]
        assert isinstance(roi, ROI)
        assert roi.x.dtype == np.float32
        assert roi.y.dtype == np.float32
        # bbox of the first ROI (rectangle 10-30)
        x0, y0, x1, y1 = roi.bbox
        assert x0 == pytest.approx(10.0)
        assert x1 == pytest.approx(30.0)
        assert y0 == pytest.approx(10.0)
        assert y1 == pytest.approx(30.0)

    def test_centroid(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        cx, cy = roiset[0].centroid
        assert cx == pytest.approx(20.0)
        assert cy == pytest.approx(20.0)

    def test_to_mask_shape_and_dtype(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        mask = roiset[0].to_mask((100, 100))
        assert mask.shape == (100, 100)
        assert mask.dtype == bool
        # rectangle 10..30 in both axes, exclusive upper bound via skimage
        assert mask[15, 15]
        assert not mask[5, 5]

    def test_to_mask_area_reasonable(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        # Square with vertices at (10,10),(30,10),(30,30),(10,30): area = 400 ideally.
        # skimage.draw.polygon is inclusive-exclusive, so allow a tolerance band.
        mask = roiset[0].to_mask((100, 100))
        assert 350 <= mask.sum() <= 500

    def test_triangle_mask(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        tri = roiset[2]
        mask = tri.to_mask((100, 100))
        # Triangle interior point should be filled, outside point should not.
        assert mask[80, 80]
        assert not mask[50, 50]

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_rois(tmp_path / "missing.zip")

    def test_empty_zip_raises(self, tmp_path: Path) -> None:
        import zipfile

        empty_zip = tmp_path / "empty.zip"
        with zipfile.ZipFile(empty_zip, "w"):
            pass
        with pytest.raises(ValueError):
            load_rois(empty_zip)

    def test_single_roi_file(self, tmp_path: Path) -> None:
        from roifile import ImagejRoi, roiwrite

        r = ImagejRoi.frompoints([[0, 0], [10, 0], [10, 10], [0, 10]])
        r.name = "solo"
        single = tmp_path / "solo.roi"
        roiwrite(str(single), r, mode="w")
        roiset = load_rois(single)
        assert len(roiset) == 1
        assert roiset[0].name == "solo"

    def test_index_preserved(self, sample_zip: Path) -> None:
        roiset = load_rois(sample_zip)
        for i, roi in enumerate(roiset):
            assert roi.index == i
