"""
3D → 2D projection
==================

Two projections are produced for every pose update:

* **Grayscale MIP** — the max-intensity projection of the grayscale
  atlas along the Z axis. Used as the anatomical reference the user
  visually compares to their imaging plane.
* **Label projection** — for each ``(y, x)`` column, the most-frequent
  non-zero label along Z. Vectorised as a single ``np.bincount`` over a
  linearised ``(column_index, label)`` pair array so the whole
  projection is one call regardless of the number of glomeruli.

Both arrays are returned in ``(H, W)`` image-pixel order where
``H = Y`` and ``W = X`` (i.e. ``axis=0`` of the volume is the
projection axis, the remaining axes map directly to image rows and
columns).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from door_toolkit.atlas_align.config import get_logger

logger = get_logger(__name__)


@dataclass
class ProjectedAtlas:
    """2D projections of a transformed atlas.

    Attributes:
        grayscale_mip: ``(H, W)`` float32 MIP of the grayscale channel.
        labelmap_projection: ``(H, W)`` uint16 per-column mode of
            non-zero labels.
        elapsed_ms: Wall-clock time spent inside :func:`project_atlas`.
    """

    grayscale_mip: np.ndarray
    labelmap_projection: np.ndarray
    elapsed_ms: float = 0.0


def _mode_along_z_ignore_zero(labelmap: np.ndarray) -> np.ndarray:
    """Per-column mode of non-zero labels along the first axis.

    Args:
        labelmap: ``(Z, H, W)`` integer array.

    Returns:
        ``(H, W)`` array with dtype matching ``labelmap``. Columns with
        no non-zero labels are filled with ``0``.

    Implementation notes:
        We build a single 1D array of ``column_index * K + label`` pairs
        (one per non-zero voxel) and ``np.bincount`` that. Then reshape
        back to ``(H*W, K)`` and take ``argmax``. Labels are small
        integers (uint16 glomerulus ids, typically < 200) so the sparse
        column-index × label grid comfortably fits in memory even for
        the full JRC2018F volume.
    """
    if labelmap.ndim != 3:
        raise ValueError(f"Expected 3D labelmap, got shape {labelmap.shape}")

    Z, H, W = labelmap.shape
    # Fast path: Z=1 → there's no "mode over Z" to compute. Used by the
    # DoOR multi-view path where each frame is one slice and we only want
    # that single slice's values. Skips a huge bincount allocation.
    if Z == 1:
        return labelmap[0].copy()

    flat_labels = labelmap.reshape(Z, -1)  # (Z, N)
    N = flat_labels.shape[1]

    max_label = int(flat_labels.max())
    if max_label == 0:
        return np.zeros((H, W), dtype=labelmap.dtype)

    K = max_label + 1  # includes 0

    # For each (col, z) the label. We want, per col, the label id with
    # the highest count among ids > 0.
    col_idx = np.broadcast_to(
        np.arange(N, dtype=np.int64)[None, :], (Z, N)
    ).reshape(-1)
    label_flat = flat_labels.reshape(-1).astype(np.int64, copy=False)

    # Ignore the ``0`` label — it always wins by volume otherwise.
    nonzero_mask = label_flat > 0
    if not nonzero_mask.any():
        return np.zeros((H, W), dtype=labelmap.dtype)
    col_idx_nz = col_idx[nonzero_mask]
    label_nz = label_flat[nonzero_mask]

    # Encode (col, label) into a single int and bincount.
    code = col_idx_nz * K + label_nz
    counts_flat = np.bincount(code, minlength=N * K)
    counts = counts_flat.reshape(N, K)

    # counts[:, 0] is always zero because we masked the zeros out.
    winners = counts.argmax(axis=1)

    # Columns where every slice was 0 → argmax returns 0 (background).
    projection = winners.reshape(H, W).astype(labelmap.dtype)
    return projection


def project_atlas(
    grayscale: np.ndarray,
    labelmap: np.ndarray,
    *,
    project_axis: int = 0,
) -> ProjectedAtlas:
    """Compute the pair of 2D projections used by the atlas_align GUI.

    Args:
        grayscale: 3D float volume.
        labelmap: 3D uint16 volume, same shape as ``grayscale``.
        project_axis: Volume axis to project along. The two remaining
            axes become the output ``(H, W)``. Default 0 (Z → image
            plane), matching the default imaging geometry.

    Returns:
        :class:`ProjectedAtlas`.
    """
    if grayscale.shape != labelmap.shape:
        raise ValueError(
            f"grayscale shape {grayscale.shape} != labelmap shape {labelmap.shape}"
        )
    if grayscale.ndim != 3:
        raise ValueError(f"Expected 3D volumes, got ndim={grayscale.ndim}")
    if project_axis not in (0, 1, 2):
        raise ValueError(f"project_axis must be 0, 1 or 2; got {project_axis}")

    t0 = time.perf_counter()

    if project_axis == 0:
        gray_oriented = grayscale
        label_oriented = labelmap
    else:
        gray_oriented = np.moveaxis(grayscale, project_axis, 0)
        label_oriented = np.moveaxis(labelmap, project_axis, 0)

    grayscale_mip = gray_oriented.max(axis=0).astype(np.float32, copy=False)
    labelmap_projection = _mode_along_z_ignore_zero(label_oriented)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.debug(
        "project_atlas axis=%d elapsed=%.1fms in_shape=%s out_shape=%s",
        project_axis, elapsed_ms, grayscale.shape, grayscale_mip.shape,
    )

    return ProjectedAtlas(
        grayscale_mip=grayscale_mip,
        labelmap_projection=labelmap_projection,
        elapsed_ms=elapsed_ms,
    )
