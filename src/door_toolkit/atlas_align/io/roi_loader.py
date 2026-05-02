"""
FIJI RoiManager loader
======================

Parses a FIJI / ImageJ ``.zip`` (or single ``.roi``) produced by
``RoiManager → More → Save…``. Each ROI is materialised into a lightweight
:class:`ROI` dataclass carrying polygon coordinates in image pixel
coordinates (x, y with origin at the top-left), the original ROI type, and
utility methods for rendering a binary mask.

The underlying parser is the `roifile <https://pypi.org/project/roifile>`_
package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from door_toolkit.atlas_align.config import get_logger

logger = get_logger(__name__)


@dataclass
class ROI:
    """A single region-of-interest in 2D image-pixel coordinates.

    Attributes:
        name: Original ROI name from ImageJ.
        x: 1D array of polygon vertex x-coordinates (float32).
        y: 1D array of polygon vertex y-coordinates (float32).
        roi_type: Original ImageJ ROI type name (``"polygon"``, ``"oval"``,
            ``"freehand"``, ...). Informational only; we always store a
            polygon approximation.
        index: Position in the source file (0-based).
    """

    name: str
    x: np.ndarray
    y: np.ndarray
    roi_type: str = "polygon"
    index: int = 0

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        """``(x_min, y_min, x_max, y_max)`` in image pixel coords."""
        return (
            float(self.x.min()),
            float(self.y.min()),
            float(self.x.max()),
            float(self.y.max()),
        )

    @property
    def centroid(self) -> Tuple[float, float]:
        """Unweighted centroid of the polygon vertices."""
        return float(self.x.mean()), float(self.y.mean())

    def to_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Rasterize to a 2D boolean mask.

        Args:
            shape: Destination ``(H, W)`` (rows, columns). Matches the
                convention of ``numpy`` image arrays.

        Returns:
            Boolean array with ``True`` inside the polygon.
        """
        from skimage.draw import polygon

        mask = np.zeros(shape, dtype=bool)
        if self.x.size < 3:
            # Degenerate ROI (point / line). Mark the single pixel/rasterised
            # line so IoU still evaluates deterministically.
            rr = np.clip(np.round(self.y).astype(int), 0, shape[0] - 1)
            cc = np.clip(np.round(self.x).astype(int), 0, shape[1] - 1)
            mask[rr, cc] = True
            return mask
        rr, cc = polygon(self.y, self.x, shape=shape)
        mask[rr, cc] = True
        return mask


@dataclass
class ROISet:
    """Ordered collection of ROIs plus the source path."""

    rois: List[ROI] = field(default_factory=list)
    source: Path = field(default_factory=Path)

    def __len__(self) -> int:
        return len(self.rois)

    def __iter__(self):
        return iter(self.rois)

    def __getitem__(self, idx: int) -> ROI:
        return self.rois[idx]

    @property
    def names(self) -> List[str]:
        return [r.name for r in self.rois]


def _coerce_coords(imagej_roi) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (x, y) vertex arrays from an :class:`roifile.ImagejRoi`."""
    # ImagejRoi.coordinates() returns (N, 2) as [x, y].
    coords = np.asarray(imagej_roi.coordinates(), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2:
        # Some ROI types (e.g. composite) may return a list of arrays; the
        # robust fallback is to concatenate all polygons so the union is
        # still rasterised as one ROI.
        parts: list[np.ndarray] = []
        for item in imagej_roi.coordinates():
            arr = np.asarray(item, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] == 2 and arr.size:
                parts.append(arr)
        if not parts:
            raise ValueError(
                f"ROI {imagej_roi.name!r}: cannot extract polygon coordinates."
            )
        coords = np.vstack(parts)
    return coords[:, 0], coords[:, 1]


def load_rois(path: Path) -> ROISet:
    """Load ROIs from a FIJI ``.zip`` or single ``.roi`` file.

    Args:
        path: Path to the ROI archive.

    Returns:
        :class:`ROISet` of parsed ROIs in source order.

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the archive is empty or every ROI is unusable.
    """
    from roifile import ImagejRoi, roiread

    path = Path(path).expanduser().resolve()
    logger.debug("load_rois(path=%s)", path)
    if not path.is_file():
        raise FileNotFoundError(f"ROI file does not exist: {path}")

    raw = roiread(str(path))
    if isinstance(raw, ImagejRoi):
        raw_list: Sequence = [raw]
    else:
        raw_list = list(raw)

    if not raw_list:
        raise ValueError(f"No ROIs found in {path}")

    parsed: List[ROI] = []
    for idx, roi in enumerate(raw_list):
        try:
            x, y = _coerce_coords(roi)
        except Exception as e:  # noqa: BLE001 — we want a clean fallback path
            logger.warning(
                "Skipping ROI %d (%s): %s", idx, getattr(roi, "name", "?"), e
            )
            continue

        name = getattr(roi, "name", None) or f"roi_{idx + 1:03d}"
        roi_type = getattr(roi, "roitype", None)
        type_str = roi_type.name.lower() if roi_type is not None else "polygon"

        parsed.append(
            ROI(
                name=str(name),
                x=np.ascontiguousarray(x, dtype=np.float32),
                y=np.ascontiguousarray(y, dtype=np.float32),
                roi_type=type_str,
                index=idx,
            )
        )

    if not parsed:
        raise ValueError(f"No usable ROIs in {path}")

    logger.info("Loaded %d ROIs from %s", len(parsed), path)
    return ROISet(rois=parsed, source=path)
