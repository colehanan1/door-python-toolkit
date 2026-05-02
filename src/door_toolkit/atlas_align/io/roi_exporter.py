"""
ROI exporter
============

Turn :class:`~door_toolkit.atlas_align.core.iou_matcher.AssignmentResult`
plus the original :class:`~door_toolkit.atlas_align.io.roi_loader.ROISet`
into:

* ``rois_assigned.zip`` — a FIJI-readable archive with ROI names
  replaced by their assigned glomerulus. Unassigned ROIs get a
  ``UNK_`` prefix.
* ``assignments.csv`` — tabular summary matching the spec's column
  set.

No data is silently dropped; every input ROI appears in both outputs.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Sequence

from door_toolkit.atlas_align.config import get_logger
from door_toolkit.atlas_align.core.iou_matcher import AssignmentResult
from door_toolkit.atlas_align.io.roi_loader import ROI

logger = get_logger(__name__)


CSV_FIELDS = [
    "roi_index",
    "roi_original_name",
    "assigned_glomerulus",
    "iou",
    "alt1_glomerulus",
    "alt1_iou",
    "alt2_glomerulus",
    "alt2_iou",
    "alt3_glomerulus",
    "alt3_iou",
    "above_threshold",
]


def export_roi_zip(
    output_zip: Path,
    rois: Sequence[ROI],
    result: AssignmentResult,
) -> Path:
    """Write an assigned-ROI ``.zip`` compatible with FIJI's RoiManager.

    Args:
        output_zip: Destination ``.zip`` path.
        rois: The original ROIs in source order (must be same length as
            ``result.assignments``).
        result: Assignment result from :func:`assign_rois`.

    Returns:
        The written path.
    """
    from roifile import ImagejRoi, roiwrite

    if len(rois) != len(result.assignments):
        raise ValueError(
            f"roi count {len(rois)} != assignment count "
            f"{len(result.assignments)}"
        )

    new_rois = []
    used_names: set[str] = set()
    for roi, assignment in zip(rois, result.assignments):
        if assignment.above_threshold:
            new_name = assignment.glomerulus_name
        else:
            new_name = f"UNK_{roi.name}"
        # Ensure unique names so FIJI does not silently merge entries.
        base = new_name
        suffix = 2
        while new_name in used_names:
            new_name = f"{base}_{suffix}"
            suffix += 1
        used_names.add(new_name)

        points = list(zip(roi.x.tolist(), roi.y.tolist()))
        new_roi = ImagejRoi.frompoints(points)
        new_roi.name = new_name
        new_rois.append(new_roi)

    output_zip = Path(output_zip)
    roiwrite(str(output_zip), new_rois, mode="w")
    logger.info("Wrote labelled ROI zip: %s (%d ROIs)", output_zip, len(new_rois))
    return output_zip


def export_assignments_csv(
    output_csv: Path, result: AssignmentResult
) -> Path:
    """Write the ``assignments.csv`` summary.

    The column layout matches the project spec exactly.
    """
    output_csv = Path(output_csv)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for a in result.assignments:
            alt = (a.alternates + [(0, "", 0.0)] * 3)[:3]
            writer.writerow(
                {
                    "roi_index": a.roi_index,
                    "roi_original_name": a.roi_name,
                    "assigned_glomerulus": (
                        a.glomerulus_name if a.above_threshold else ""
                    ),
                    "iou": f"{a.iou:.6f}",
                    "alt1_glomerulus": alt[0][1],
                    "alt1_iou": f"{alt[0][2]:.6f}",
                    "alt2_glomerulus": alt[1][1],
                    "alt2_iou": f"{alt[1][2]:.6f}",
                    "alt3_glomerulus": alt[2][1],
                    "alt3_iou": f"{alt[2][2]:.6f}",
                    "above_threshold": (
                        "1" if a.above_threshold else "0"
                    ),
                }
            )
    logger.info("Wrote assignments CSV: %s", output_csv)
    return output_csv
