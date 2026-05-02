"""Tests for :mod:`door_toolkit.atlas_align.core.volume_transform`."""

from __future__ import annotations

import numpy as np
import pytest

from door_toolkit.atlas_align.core.volume_transform import (
    Pose,
    TransformedAtlas,
    build_affine_matrix,
    clear_cache,
    transform_atlas,
)


@pytest.fixture(autouse=True)
def _clean_cache():
    clear_cache()
    yield
    clear_cache()


@pytest.fixture
def small_volume():
    rng = np.random.default_rng(42)
    gray = rng.random((20, 30, 30), dtype=np.float32)
    lm = np.zeros_like(gray, dtype=np.uint16)
    lm[5:15, 10:20, 10:20] = 1
    lm[2:8, 5:10, 20:28] = 2
    return gray, lm


@pytest.mark.atlas_align
class TestPose:

    def test_identity_dict_roundtrip(self) -> None:
        p = Pose()
        p2 = Pose.from_dict(p.to_dict())
        assert p == p2

    def test_digest_is_deterministic(self) -> None:
        a = Pose(tx=1, ry=2.5, flip_x=True)
        b = Pose(tx=1, ry=2.5, flip_x=True)
        assert a.digest() == b.digest()

    def test_digest_changes_with_dofs(self) -> None:
        a = Pose().digest()
        b = Pose(tx=0.01).digest()
        assert a != b


@pytest.mark.atlas_align
class TestAffineMatrix:

    def test_identity_matrix(self) -> None:
        m, off = build_affine_matrix(Pose(), (10, 20, 30))
        np.testing.assert_allclose(m, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(off, np.zeros(3), atol=1e-9)

    def test_pure_scale_changes_matrix(self) -> None:
        m, _ = build_affine_matrix(Pose(sx=2.0, sy=2.0, sz=2.0), (10, 10, 10))
        # inverse of 2x scale is 0.5
        np.testing.assert_allclose(
            np.diag(m), [0.5, 0.5, 0.5], atol=1e-9
        )

    def test_degenerate_pose_is_clamped_not_singular(self) -> None:
        """Zero-scale sliders must not produce NaN/inf inverse matrices."""
        m, off = build_affine_matrix(Pose(sx=0.0), (10, 10, 10))
        assert np.all(np.isfinite(m))
        assert np.all(np.isfinite(off))


@pytest.mark.atlas_align
class TestTransformAtlas:

    def test_identity_pose_returns_same_volumes(self, small_volume) -> None:
        gray, lm = small_volume
        result = transform_atlas(gray, lm, Pose(), use_cache=False)
        assert isinstance(result, TransformedAtlas)
        np.testing.assert_allclose(result.grayscale, gray, atol=1e-5)
        np.testing.assert_array_equal(result.labelmap, lm)

    def test_labelmap_dtype_preserved(self, small_volume) -> None:
        gray, lm = small_volume
        result = transform_atlas(
            gray, lm, Pose(tx=2, rz=30.0), use_cache=False
        )
        assert result.labelmap.dtype == np.uint16

    def test_labelmap_values_are_integer_after_rotation(
        self, small_volume
    ) -> None:
        """Nearest-neighbour sampling must preserve the integer label set."""
        gray, lm = small_volume
        original_labels = set(int(v) for v in np.unique(lm))
        result = transform_atlas(gray, lm, Pose(rz=45.0), use_cache=False)
        new_labels = set(int(v) for v in np.unique(result.labelmap))
        # We should not invent fractional labels. new_labels ⊆ original_labels.
        assert new_labels.issubset(original_labels)

    def test_translation_moves_content(self, small_volume) -> None:
        gray, lm = small_volume
        baseline = transform_atlas(gray, lm, Pose(), use_cache=False)
        shifted = transform_atlas(gray, lm, Pose(tx=5), use_cache=False)
        assert not np.array_equal(baseline.labelmap, shifted.labelmap)
        # After shifting +5 in X, the shape should have moved ~5 voxels
        orig_cx = np.where(baseline.labelmap == 1)[2].mean()
        new_cx = np.where(shifted.labelmap == 1)[2].mean()
        assert new_cx == pytest.approx(orig_cx + 5.0, abs=1.0)

    def test_flip_x_mirrors_volume(self, small_volume) -> None:
        gray, lm = small_volume
        flipped = transform_atlas(
            gray, lm, Pose(flip_x=True), use_cache=False
        )
        expected = lm[:, :, ::-1]
        np.testing.assert_array_equal(flipped.labelmap, expected)

    def test_rotation_by_360_is_identity(self, small_volume) -> None:
        gray, lm = small_volume
        rotated = transform_atlas(gray, lm, Pose(rz=360.0), use_cache=False)
        # Label locations should be near-identical after full revolution.
        orig_count = int((lm == 1).sum())
        new_count = int((rotated.labelmap == 1).sum())
        assert abs(orig_count - new_count) / max(orig_count, 1) < 0.05

    def test_rotation_by_90_preserves_voxel_count(
        self, small_volume
    ) -> None:
        """A 90° rotation about the Z axis should preserve integer voxel counts
        (within rasterisation noise)."""
        gray, lm = small_volume
        rot = transform_atlas(gray, lm, Pose(rz=90.0), use_cache=False)
        for label_id in (1, 2):
            orig = int((lm == label_id).sum())
            new = int((rot.labelmap == label_id).sum())
            # Allow small rasterisation drift because centre is non-integer
            assert abs(new - orig) / max(orig, 1) < 0.1

    def test_cache_hit_returns_same_object(self, small_volume) -> None:
        gray, lm = small_volume
        pose = Pose(tx=1, ry=5.0)
        first = transform_atlas(gray, lm, pose, use_cache=True)
        second = transform_atlas(gray, lm, pose, use_cache=True)
        # On a cache hit we get the exact same object.
        assert first is second

    def test_shape_mismatch_raises(self, small_volume) -> None:
        gray, lm = small_volume
        with pytest.raises(ValueError):
            transform_atlas(gray, lm[:-1], Pose())

    def test_elapsed_ms_set(self, small_volume) -> None:
        gray, lm = small_volume
        result = transform_atlas(gray, lm, Pose(rz=10.0), use_cache=False)
        assert result.elapsed_ms >= 0.0
