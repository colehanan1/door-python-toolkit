"""Tests for :mod:`door_toolkit.atlas_align.core.iou_matcher`."""

from __future__ import annotations

import numpy as np
import pytest

from door_toolkit.atlas_align.core.iou_matcher import (
    Assignment,
    AssignmentResult,
    assign_rois,
    iou_matrix,
)


def _mask(shape, rr_cc_ranges):
    m = np.zeros(shape, dtype=bool)
    for (r0, r1), (c0, c1) in rr_cc_ranges:
        m[r0:r1, c0:c1] = True
    return m


@pytest.mark.atlas_align
class TestIoUMatrix:

    def test_identical_masks_give_iou_1(self) -> None:
        lp = np.zeros((10, 10), dtype=np.uint16)
        lp[2:8, 2:8] = 1
        iou, ids = iou_matrix([lp == 1], lp)
        assert ids == [1]
        assert iou.shape == (1, 1)
        assert iou[0, 0] == pytest.approx(1.0)

    def test_disjoint_masks_give_iou_0(self) -> None:
        lp = np.zeros((10, 10), dtype=np.uint16)
        lp[0:3, 0:3] = 1
        roi = np.zeros((10, 10), dtype=bool)
        roi[7:10, 7:10] = True
        iou, _ = iou_matrix([roi], lp)
        assert iou[0, 0] == 0.0

    def test_partial_overlap_is_fractional(self) -> None:
        lp = np.zeros((10, 10), dtype=np.uint16)
        lp[2:6, 2:6] = 1  # 4x4 = 16 pixels
        roi = np.zeros((10, 10), dtype=bool)
        roi[4:8, 4:8] = True  # overlap = 2x2 = 4, union = 16 + 16 - 4 = 28
        iou, _ = iou_matrix([roi], lp)
        assert iou[0, 0] == pytest.approx(4.0 / 28.0)

    def test_multiple_rois_and_glomeruli(self) -> None:
        lp = np.zeros((20, 20), dtype=np.uint16)
        lp[2:8, 2:8] = 1  # label 1
        lp[12:18, 12:18] = 2  # label 2
        roi_a = _mask((20, 20), [((2, 8), (2, 8))])  # exact match for label 1
        roi_b = _mask((20, 20), [((12, 18), (12, 18))])  # exact match for label 2
        iou, ids = iou_matrix([roi_a, roi_b], lp)
        assert set(ids) == {1, 2}
        # Order of ids should be sorted.
        assert ids == [1, 2]
        assert iou[0, 0] == pytest.approx(1.0)
        assert iou[0, 1] == 0.0
        assert iou[1, 0] == 0.0
        assert iou[1, 1] == pytest.approx(1.0)

    def test_empty_roi_gives_zero_row(self) -> None:
        lp = np.zeros((5, 5), dtype=np.uint16)
        lp[1, 1] = 1
        iou, _ = iou_matrix([np.zeros((5, 5), dtype=bool)], lp)
        assert (iou == 0.0).all()

    def test_shape_mismatch_raises(self) -> None:
        lp = np.zeros((5, 5), dtype=np.uint16)
        lp[0, 0] = 1  # need at least one glomerulus so the shape check fires
        wrong = np.zeros((6, 5), dtype=bool)
        with pytest.raises(ValueError):
            iou_matrix([wrong], lp)


@pytest.mark.atlas_align
class TestAssignROIs:

    def test_unambiguous_assignment(self) -> None:
        lp = np.zeros((20, 20), dtype=np.uint16)
        lp[2:8, 2:8] = 1
        lp[12:18, 12:18] = 2
        labels = {1: "DM1", 2: "DM2"}
        rois = [
            _mask((20, 20), [((2, 8), (2, 8))]),
            _mask((20, 20), [((12, 18), (12, 18))]),
        ]
        result = assign_rois(rois, ["r1", "r2"], lp, labels, threshold=0.3)

        assert isinstance(result, AssignmentResult)
        assert result.assignments[0].glomerulus_name == "DM1"
        assert result.assignments[0].iou == pytest.approx(1.0)
        assert result.assignments[1].glomerulus_name == "DM2"
        assert result.assignments[0].above_threshold
        assert result.assignments[1].above_threshold
        assert result.n_above_threshold == 2
        assert result.n_below_threshold == 0

    def test_below_threshold_unassigned(self) -> None:
        lp = np.zeros((20, 20), dtype=np.uint16)
        lp[2:8, 2:8] = 1
        # Tiny overlap → IoU very small.
        roi = _mask((20, 20), [((6, 7), (6, 7))])
        result = assign_rois(
            [roi], ["r"], lp, {1: "DM1"}, threshold=0.5
        )
        a = result.assignments[0]
        assert a.glomerulus_id == 0
        assert a.glomerulus_name.startswith("UNK_")
        assert a.above_threshold is False

    def test_conflict_resolution_highest_iou_wins(self) -> None:
        """Two ROIs argmax on the same glomerulus; the one with higher IoU keeps it."""
        lp = np.zeros((20, 20), dtype=np.uint16)
        lp[2:12, 2:12] = 1  # single big glomerulus
        lp[14:20, 14:20] = 2  # smaller second glomerulus (6x6=36)

        roi_big = _mask((20, 20), [((2, 12), (2, 12))])  # perfect IoU with 1
        roi_small = _mask((20, 20), [((5, 9), (5, 9))])  # also overlaps 1 but tiny

        result = assign_rois(
            [roi_big, roi_small],
            ["big", "small"],
            lp,
            {1: "DM1", 2: "DM2"},
            threshold=0.01,
        )
        # "big" should keep DM1.
        assert result.assignments[0].glomerulus_name == "DM1"
        # "small" should fall to its next-best — here only DM2 is left but
        # it has no overlap, so it ends up below threshold = unassigned.
        assert result.assignments[1].glomerulus_id != 1
        assert result.iterations >= 1

    def test_top_3_alternates_returned(self) -> None:
        lp = np.zeros((20, 20), dtype=np.uint16)
        lp[2:8, 2:8] = 1
        lp[9:13, 2:8] = 2
        lp[14:18, 2:8] = 3
        labels = {1: "DM1", 2: "DM2", 3: "DM3"}
        # A single ROI covering all three stripes → IoU to each > 0.
        roi = _mask((20, 20), [((2, 18), (2, 8))])
        result = assign_rois(
            [roi], ["r"], lp, labels, threshold=0.01
        )
        a = result.assignments[0]
        assert len(a.alternates) <= 3
        # Alternates exclude the assigned glomerulus.
        for alt_id, alt_name, alt_iou in a.alternates:
            assert alt_id != a.glomerulus_id
            assert alt_iou > 0.0

    def test_mean_iou_is_computed(self) -> None:
        lp = np.zeros((10, 10), dtype=np.uint16)
        lp[0:5, 0:5] = 1
        roi = lp == 1
        result = assign_rois(
            [roi, roi], ["a", "b"], lp, {1: "DM1"}, threshold=0.01
        )
        # First ROI wins at IoU=1, second is evicted in conflict resolution
        # and ends up unassigned → mean iou = (1.0 + 1.0) / 2 (both have IoU 1,
        # but the loser is flagged below_threshold yet its iou field = max IoU).
        assert 0.0 < result.mean_iou <= 1.0

    def test_length_mismatch_raises(self) -> None:
        lp = np.zeros((5, 5), dtype=np.uint16)
        with pytest.raises(ValueError):
            assign_rois([np.zeros((5, 5), dtype=bool)], [], lp, {})

    def test_iou_matrix_shape(self) -> None:
        lp = np.zeros((10, 10), dtype=np.uint16)
        lp[0:3, 0:3] = 1
        lp[6:9, 6:9] = 2
        rois = [
            _mask((10, 10), [((0, 3), (0, 3))]),
            _mask((10, 10), [((6, 9), (6, 9))]),
            _mask((10, 10), [((4, 5), (4, 5))]),
        ]
        result = assign_rois(
            rois, ["a", "b", "c"], lp, {1: "DM1", 2: "DM2"}, threshold=0.3
        )
        assert result.iou_matrix.shape == (3, 2)
        assert result.glomerulus_ids == [1, 2]
