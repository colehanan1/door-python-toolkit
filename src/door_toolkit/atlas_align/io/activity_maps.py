"""
Activity-map loader
===================

Reads a folder of per-odor activity maps (e.g. ``PNR_per_trial.tif``)
produced by the upstream imaging pipeline and exposes a per-ROI
response-vector computation so user-drawn polygons can be compared
against DoOR using the same machinery as the ImageJ-derived ΔF/F CSVs.

Expected folder:

* ``PNR_per_trial.tif`` — ``(n_odors, H, W)`` float32 stack.
* ``PNR_mean_fullframe.tif`` — optional ``(H_full, W_full)`` anatomical
  reference for drawing ROIs on. When present the GUI uses this as the
  display canvas.
* ``README.txt`` — may document per-slice odor ordering; we default to
  ``A, B, C, H, L, O, E`` (the ordering in
  ``build_activity_masks.py``'s output).

The per-trial stack's odor ordering does **not** match the dff-CSV
alphabetical ordering. We remap here so that everything downstream —
DoOR comparison, Response panel, etc — sees odors in the same canonical
order as the rest of the GUI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

from door_toolkit.atlas_align.config import get_logger
from door_toolkit.atlas_align.io.dff_loader import DEFAULT_ODOR_LETTER_MAP

logger = get_logger(__name__)


#: Canonical Z-slice order of ``PNR_per_trial.tif`` (as documented in the
#: README shipped alongside the file).
DEFAULT_PER_TRIAL_SLICE_LETTERS: Tuple[str, ...] = (
    "A", "B", "C", "H", "L", "O", "E",
)

PER_TRIAL_FILENAME = "PNR_per_trial.tif"
FULLFRAME_REF_FILENAME = "PNR_mean_fullframe.tif"
CROP_REF_FILENAME = "PNR_mean.tif"


@dataclass
class ActivityMaps:
    """Per-odor activity stack + (optional) anatomical reference image.

    Attributes:
        odor_order: Canonical odor names in the order matching the
            :class:`~door_toolkit.atlas_align.io.dff_loader.DFFBundle`
            the rest of the app uses. The per-trial stack has been
            permuted to match.
        stack: ``(n_odors, H_map, W_map)`` float32 array — each Z slice
            is one odor's activity landscape.
        reference_fullframe: Optional ``(H_ref, W_ref)`` anatomical
            reference image (typically 2048×2048 so FIJI ROI pixel
            coordinates line up directly). ``None`` if not on disk.
        map_to_ref_offset: ``(row_offset, col_offset)`` of the cropped
            activity map inside the fullframe reference. Used to map
            ROI pixel coords drawn on the fullframe image → activity-map
            pixel coords. Defaults to ``(0, 0)`` when unknown.
        source_dir: Directory we loaded from.
    """

    odor_order: List[str]
    stack: np.ndarray
    reference_fullframe: Optional[np.ndarray] = None
    map_to_ref_offset: Tuple[int, int] = (0, 0)
    source_dir: Optional[Path] = None

    @property
    def n_odors(self) -> int:
        return int(self.stack.shape[0])

    @property
    def map_shape(self) -> Tuple[int, int]:
        return int(self.stack.shape[1]), int(self.stack.shape[2])

    def response_for_polygon(
        self,
        xs_ref: np.ndarray,
        ys_ref: np.ndarray,
        reduction: str = "mean",
    ) -> np.ndarray:
        """Compute a per-odor response vector for a polygon drawn in the
        fullframe image's pixel coordinates.

        Args:
            xs_ref, ys_ref: Polygon vertex coordinates in the fullframe
                reference's pixel space.
            reduction: ``"mean"`` (default) or ``"max"`` — how to collapse
                the pixels inside the polygon into a single per-odor
                number.

        Returns:
            ``(n_odors,)`` float32 vector, in ``self.odor_order``.
        """
        from skimage.draw import polygon as sk_polygon

        H, W = self.map_shape
        off_r, off_c = self.map_to_ref_offset
        # Shift polygon from ref coords → map coords.
        xs_map = np.asarray(xs_ref, dtype=np.float64) - off_c
        ys_map = np.asarray(ys_ref, dtype=np.float64) - off_r
        rr, cc = sk_polygon(ys_map, xs_map, shape=(H, W))
        out = np.zeros(self.n_odors, dtype=np.float32)
        if rr.size == 0:
            return out
        for z in range(self.n_odors):
            pixels = self.stack[z, rr, cc]
            if reduction == "mean":
                out[z] = float(np.nanmean(pixels)) if pixels.size else 0.0
            elif reduction == "max":
                out[z] = float(np.nanmax(pixels)) if pixels.size else 0.0
            else:
                raise ValueError(f"Unknown reduction {reduction!r}")
        return out


def _resolve_odor_names(
    slice_letters: Tuple[str, ...],
    odor_letter_map: Dict[str, str],
) -> List[str]:
    return [odor_letter_map.get(letter, f"odor_{letter}") for letter in slice_letters]


def _permute_to_canonical_order(
    stack: np.ndarray,
    source_odors: List[str],
    target_odors: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """Reorder ``stack`` Z axis to match ``target_odors``.

    Drops odors that appear in the stack but not in the target list.
    """
    src_index = {name: i for i, name in enumerate(source_odors)}
    keep_target: List[str] = []
    indices: List[int] = []
    for name in target_odors:
        if name in src_index:
            indices.append(src_index[name])
            keep_target.append(name)
    if not indices:
        logger.warning(
            "No overlap between activity-map odors %s and target %s",
            source_odors, target_odors,
        )
        return stack, source_odors
    return stack[indices], keep_target


def _infer_fullframe_offset(
    crop_shape: Tuple[int, int],
    fullframe_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Return ``(row, col)`` offset centring the crop inside the fullframe.

    The pipeline embeds the AL bounding-box crop inside a larger
    fullframe canvas. If we haven't been given the exact offset, assume
    it was centred — consistent with how most crop-embed utilities work.
    """
    ch, cw = crop_shape
    fh, fw = fullframe_shape
    return ((fh - ch) // 2, (fw - cw) // 2)


def load_activity_maps(
    path: Path,
    target_odor_order: Optional[List[str]] = None,
) -> ActivityMaps:
    """Load ``PNR_per_trial.tif`` (and optional fullframe reference) into
    an :class:`ActivityMaps`.

    Args:
        path: Directory containing ``PNR_per_trial.tif``.
        target_odor_order: If given, permute the stack's Z axis to match
            this odor order. Typical callers pass the
            :class:`DFFBundle.odor_order` so the two sources align.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Activity-maps directory not found: {path}")

    per_trial_path = path / PER_TRIAL_FILENAME
    if not per_trial_path.is_file():
        raise FileNotFoundError(
            f"{PER_TRIAL_FILENAME} missing in {path}."
        )
    stack = tifffile.imread(str(per_trial_path)).astype(np.float32, copy=False)
    if stack.ndim != 3:
        raise ValueError(
            f"Expected {PER_TRIAL_FILENAME} to be 3D (Z, H, W); got {stack.shape}"
        )

    source_odor_names = _resolve_odor_names(
        DEFAULT_PER_TRIAL_SLICE_LETTERS, DEFAULT_ODOR_LETTER_MAP
    )
    if target_odor_order is not None:
        stack, source_odor_names = _permute_to_canonical_order(
            stack, source_odor_names, list(target_odor_order)
        )
    logger.info(
        "Activity maps loaded: %d odors × %d × %d (odor order: %s)",
        stack.shape[0], stack.shape[1], stack.shape[2], source_odor_names,
    )

    # Fullframe reference.
    ref_path = path / FULLFRAME_REF_FILENAME
    reference_fullframe: Optional[np.ndarray] = None
    offset = (0, 0)
    if ref_path.is_file():
        reference_fullframe = tifffile.imread(str(ref_path)).astype(
            np.float32, copy=False
        )
        if reference_fullframe.ndim != 2:
            logger.warning(
                "Unexpected shape for %s: %s", ref_path, reference_fullframe.shape
            )
            reference_fullframe = None
        else:
            offset = _infer_fullframe_offset(
                stack.shape[1:], reference_fullframe.shape
            )
            logger.info(
                "Fullframe reference loaded: %s (crop offset %s)",
                reference_fullframe.shape, offset,
            )

    return ActivityMaps(
        odor_order=source_odor_names,
        stack=stack,
        reference_fullframe=reference_fullframe,
        map_to_ref_offset=offset,
        source_dir=path,
    )
