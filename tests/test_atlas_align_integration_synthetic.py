"""End-to-end integration test using the mock atlas.

Builds a mock atlas (``--mock-flywire``), projects it to 2D, synthesises
ROIs from that same projection (so ground truth is known), then runs
the full ``transform_atlas → project_atlas → assign_rois`` pipeline
and verifies recovery accuracy.

Also exercises the GUI's ``export_to_directory`` path in a headless
pytest-qt context to confirm file output works end-to-end.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from door_toolkit.atlas_align.atlas_builder import build_labelmap
from door_toolkit.atlas_align.core.iou_matcher import assign_rois
from door_toolkit.atlas_align.core.projection import project_atlas
from door_toolkit.atlas_align.core.volume_transform import Pose, transform_atlas
from door_toolkit.atlas_align.io.atlas_loader import load_atlas_bundle
from door_toolkit.atlas_align.io.roi_loader import ROI, ROISet, load_rois
from door_toolkit.atlas_align.io.roi_exporter import (
    export_assignments_csv,
    export_roi_zip,
)


def _mask_to_polygon(mask: np.ndarray) -> np.ndarray | None:
    """Return an (N, 2) (x, y) float32 polygon tracing the mask boundary."""
    from skimage.measure import find_contours

    contours = find_contours(mask.astype(np.float32), level=0.5)
    if not contours:
        return None
    # Pick the longest contour; skimage returns (row, col) = (y, x).
    contour = max(contours, key=len)
    return np.column_stack([contour[:, 1], contour[:, 0]]).astype(np.float32)


def _build_synthetic_reference_and_rois(
    atlas_dir: Path, reference_tif: Path, roi_zip: Path
) -> dict[int, str]:
    """Project the mock atlas, build per-glomerulus ROIs, save to disk.

    Returns a ``{roi_index → true_glomerulus_name}`` ground-truth map.
    """
    from roifile import ImagejRoi, roiwrite
    import tifffile

    bundle = load_atlas_bundle(atlas_dir)
    # Identity transform → same volumes → their projection is the ground truth.
    transformed = transform_atlas(
        bundle.grayscale, bundle.labelmap, Pose(), use_cache=False
    )
    projection = project_atlas(transformed.grayscale, transformed.labelmap)

    tifffile.imwrite(str(reference_tif), projection.grayscale_mip)

    rois: list[ImagejRoi] = []
    ground_truth: dict[int, str] = {}
    for label_id in sorted(bundle.labels.keys()):
        label_name = bundle.labels[label_id]
        mask = projection.labelmap_projection == label_id
        if not mask.any():
            continue
        polygon_xy = _mask_to_polygon(mask)
        if polygon_xy is None or len(polygon_xy) < 3:
            continue
        r = ImagejRoi.frompoints(polygon_xy.tolist())
        r.name = f"roi_{len(rois):03d}"
        rois.append(r)
        ground_truth[len(rois) - 1] = label_name

    roiwrite(str(roi_zip), rois, mode="w")
    return ground_truth


@pytest.mark.atlas_align
@pytest.mark.integration
class TestEndToEndPipeline:

    def test_recovered_assignment_accuracy(self, tmp_path: Path) -> None:
        """≥90 % of ROIs must recover the correct glomerulus name."""
        atlas_dir = tmp_path / "atlas"
        build_labelmap(output_dir=atlas_dir, mock_flywire=True)

        reference_tif = tmp_path / "reference.tif"
        roi_zip = tmp_path / "rois.zip"
        ground_truth = _build_synthetic_reference_and_rois(
            atlas_dir, reference_tif, roi_zip
        )
        assert len(ground_truth) >= 6  # Make sure we have enough ROIs to test.

        bundle = load_atlas_bundle(atlas_dir)
        transformed = transform_atlas(
            bundle.grayscale, bundle.labelmap, Pose(), use_cache=False
        )
        projection = project_atlas(transformed.grayscale, transformed.labelmap)

        roi_set = load_rois(roi_zip)
        roi_masks = [roi.to_mask(bundle.shape[1:]) for roi in roi_set]

        result = assign_rois(
            roi_masks,
            [roi.name for roi in roi_set],
            projection.labelmap_projection,
            bundle.labels,
            threshold=0.3,
        )

        # Score only ROIs above threshold.
        above = [a for a in result.assignments if a.above_threshold]
        correct = sum(
            1 for a in above
            if a.glomerulus_name == ground_truth.get(a.roi_index)
        )
        accuracy = correct / max(len(above), 1)
        assert len(above) >= int(0.8 * len(ground_truth))
        assert accuracy >= 0.9, (
            f"accuracy {accuracy:.3f} < 0.9; "
            f"above={len(above)} correct={correct}"
        )

    def test_export_writes_all_three_files(self, tmp_path: Path) -> None:
        atlas_dir = tmp_path / "atlas"
        build_labelmap(output_dir=atlas_dir, mock_flywire=True)

        reference_tif = tmp_path / "reference.tif"
        roi_zip = tmp_path / "rois.zip"
        _build_synthetic_reference_and_rois(
            atlas_dir, reference_tif, roi_zip
        )

        bundle = load_atlas_bundle(atlas_dir)
        transformed = transform_atlas(
            bundle.grayscale, bundle.labelmap, Pose(), use_cache=False
        )
        projection = project_atlas(transformed.grayscale, transformed.labelmap)

        roi_set = load_rois(roi_zip)
        roi_masks = [roi.to_mask(bundle.shape[1:]) for roi in roi_set]
        result = assign_rois(
            roi_masks,
            [roi.name for roi in roi_set],
            projection.labelmap_projection,
            bundle.labels,
        )

        out = tmp_path / "out"
        out.mkdir()
        export_roi_zip(out / "rois_assigned.zip", list(roi_set), result)
        export_assignments_csv(out / "assignments.csv", result)

        assert (out / "rois_assigned.zip").is_file()
        assert (out / "assignments.csv").is_file()

        with (out / "assignments.csv").open() as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == len(roi_set)


@pytest.mark.atlas_align
@pytest.mark.integration
class TestGUISmoke:
    """Headless instantiation of the main window + export round-trip."""

    def test_main_window_constructs_and_exports(
        self, qtbot, tmp_path: Path
    ) -> None:
        from door_toolkit.atlas_align.gui.main_window import (
            AtlasAlignMainWindow,
        )
        import tifffile

        atlas_dir = tmp_path / "atlas"
        build_labelmap(output_dir=atlas_dir, mock_flywire=True)
        reference_tif = tmp_path / "reference.tif"
        roi_zip = tmp_path / "rois.zip"
        _build_synthetic_reference_and_rois(
            atlas_dir, reference_tif, roi_zip
        )

        bundle = load_atlas_bundle(atlas_dir)
        reference = tifffile.imread(reference_tif)
        roi_set = load_rois(roi_zip)

        window = AtlasAlignMainWindow(
            bundle=bundle,
            reference_image=reference.astype(np.float32),
            roi_set=roi_set,
            reference_path=reference_tif,
        )
        qtbot.addWidget(window)

        # Wait until the first projection cycle has completed.
        def _has_payload() -> bool:
            return window._last_payload is not None

        qtbot.waitUntil(_has_payload, timeout=5000)

        out = tmp_path / "export"
        out.mkdir()
        window.export_to_directory(out)

        assert (out / "rois_assigned.zip").is_file()
        assert (out / "assignments.csv").is_file()
        assert (out / "pose.json").is_file()
