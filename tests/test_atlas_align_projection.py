"""Tests for :mod:`door_toolkit.atlas_align.core.projection`."""

from __future__ import annotations

import numpy as np
import pytest

from door_toolkit.atlas_align.core.projection import (
    ProjectedAtlas,
    project_atlas,
)


@pytest.mark.atlas_align
class TestGrayscaleMIP:

    def test_mip_is_elementwise_max_along_z(self) -> None:
        gray = np.zeros((5, 4, 4), dtype=np.float32)
        gray[2, 1, 1] = 3.0
        gray[4, 1, 1] = 7.0
        gray[0, 2, 2] = 1.0

        lm = np.zeros_like(gray, dtype=np.uint16)
        lm[2, 1, 1] = 1
        lm[0, 2, 2] = 1

        proj = project_atlas(gray, lm)
        assert proj.grayscale_mip.shape == (4, 4)
        assert proj.grayscale_mip[1, 1] == 7.0
        assert proj.grayscale_mip[2, 2] == 1.0
        assert proj.grayscale_mip[0, 0] == 0.0

    def test_mip_dtype_is_float32(self) -> None:
        gray = np.ones((3, 4, 4), dtype=np.float32)
        lm = np.zeros_like(gray, dtype=np.uint16)
        proj = project_atlas(gray, lm)
        assert proj.grayscale_mip.dtype == np.float32


@pytest.mark.atlas_align
class TestLabelProjection:

    def test_empty_labelmap_returns_zeros(self) -> None:
        gray = np.zeros((5, 4, 4), dtype=np.float32)
        lm = np.zeros_like(gray, dtype=np.uint16)
        proj = project_atlas(gray, lm)
        assert proj.labelmap_projection.shape == (4, 4)
        assert proj.labelmap_projection.dtype == np.uint16
        assert (proj.labelmap_projection == 0).all()

    def test_single_label_majority(self) -> None:
        gray = np.zeros((5, 3, 3), dtype=np.float32)
        lm = np.zeros_like(gray, dtype=np.uint16)
        lm[:, 1, 1] = 5  # whole column is label 5
        proj = project_atlas(gray, lm)
        assert proj.labelmap_projection[1, 1] == 5
        # Other columns remain background.
        assert proj.labelmap_projection[0, 0] == 0

    def test_zero_label_ignored_as_winner(self) -> None:
        """Even if 0 is the most-frequent value, we pick the top non-zero label."""
        gray = np.zeros((5, 2, 2), dtype=np.float32)
        lm = np.zeros_like(gray, dtype=np.uint16)
        # Column (0, 0): 4 zeros and one 3 → projection should be 3, not 0.
        lm[2, 0, 0] = 3
        proj = project_atlas(gray, lm)
        assert proj.labelmap_projection[0, 0] == 3

    def test_label_mode_resolves_ties_deterministically(self) -> None:
        """When two labels tie, ``np.argmax`` picks the lowest label id."""
        gray = np.zeros((5, 1, 1), dtype=np.float32)
        lm = np.zeros_like(gray, dtype=np.uint16)
        lm[0, 0, 0] = 2
        lm[1, 0, 0] = 2
        lm[2, 0, 0] = 7
        lm[3, 0, 0] = 7
        proj = project_atlas(gray, lm)
        # Both 2 and 7 have 2 voxels; argmax returns the lowest index.
        assert proj.labelmap_projection[0, 0] == 2

    def test_multi_label_separate_columns(self) -> None:
        gray = np.zeros((4, 3, 3), dtype=np.float32)
        lm = np.zeros_like(gray, dtype=np.uint16)
        lm[:, 0, 0] = 1
        lm[:, 1, 1] = 2
        lm[:, 2, 2] = 3
        proj = project_atlas(gray, lm)
        assert proj.labelmap_projection[0, 0] == 1
        assert proj.labelmap_projection[1, 1] == 2
        assert proj.labelmap_projection[2, 2] == 3


@pytest.mark.atlas_align
class TestProjectAtlasAxis:

    def test_project_along_y_axis(self) -> None:
        gray = np.zeros((3, 5, 4), dtype=np.float32)
        lm = np.zeros_like(gray, dtype=np.uint16)
        lm[1, :, 2] = 9  # label 9 stretches along Y at X=2, Z=1
        proj = project_atlas(gray, lm, project_axis=1)
        # After moveaxis(1 → 0), remaining axes are (Z, X), shape (3, 4).
        assert proj.labelmap_projection.shape == (3, 4)
        assert proj.labelmap_projection[1, 2] == 9

    def test_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            project_atlas(
                np.zeros((3, 4, 4), dtype=np.float32),
                np.zeros((3, 4, 5), dtype=np.uint16),
            )

    def test_non_3d_raises(self) -> None:
        with pytest.raises(ValueError):
            project_atlas(
                np.zeros((4, 4), dtype=np.float32),
                np.zeros((4, 4), dtype=np.uint16),
            )

    def test_invalid_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            project_atlas(
                np.zeros((4, 4, 4), dtype=np.float32),
                np.zeros((4, 4, 4), dtype=np.uint16),
                project_axis=3,
            )


@pytest.mark.atlas_align
class TestPerformance:

    def test_mock_shape_fast(self) -> None:
        """Projection of the mock-atlas shape should be <20 ms."""
        gray = np.random.default_rng(0).random((30, 60, 60)).astype(
            np.float32
        )
        lm = np.zeros((30, 60, 60), dtype=np.uint16)
        lm[5:15, 10:30, 10:30] = 1
        lm[20:28, 40:55, 40:55] = 2
        proj = project_atlas(gray, lm)
        assert isinstance(proj, ProjectedAtlas)
        assert proj.elapsed_ms < 500.0  # generous ceiling for CI jitter
