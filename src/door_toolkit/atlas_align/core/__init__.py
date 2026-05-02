"""Core algorithms for the atlas_align subpackage.

Modules:

* :mod:`~.volume_transform` — 10-DOF affine posing of 3D volumes.
* :mod:`~.projection` — 3D → 2D projection (MIP + argmax-of-bincount).
* :mod:`~.iou_matcher` — ROI ↔ glomerulus IoU + greedy assignment.
"""

from __future__ import annotations

from door_toolkit.atlas_align.core.iou_matcher import (
    Assignment,
    AssignmentResult,
    assign_rois,
    iou_matrix,
)
from door_toolkit.atlas_align.core.projection import (
    ProjectedAtlas,
    project_atlas,
)
from door_toolkit.atlas_align.core.volume_transform import (
    Pose,
    TransformedAtlas,
    build_affine_matrix,
    transform_atlas,
)

__all__ = [
    "Pose",
    "TransformedAtlas",
    "build_affine_matrix",
    "transform_atlas",
    "ProjectedAtlas",
    "project_atlas",
    "Assignment",
    "AssignmentResult",
    "assign_rois",
    "iou_matrix",
]
