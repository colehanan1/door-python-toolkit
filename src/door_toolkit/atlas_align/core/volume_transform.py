"""
10-DOF volume transform
=======================

Apply a 10-parameter rigid-plus-scale affine transform to a 3D atlas:

* 3 translations (``tx``, ``ty``, ``tz``) in voxels
* 3 Euler rotations (``rx``, ``ry``, ``rz``) in degrees (intrinsic ZYX)
* 3 scales (``sx``, ``sy``, ``sz``)
* 3 flip flags (``flip_x``, ``flip_y``, ``flip_z``)

The input (``grayscale``, ``labelmap``) pair is resampled to an output
volume of the same voxel shape.

**Critical**: the labelmap is always sampled with ``order=0`` (nearest
neighbour) and ``prefilter=False`` so integer labels are preserved
exactly.  The grayscale volume is sampled with ``order=1`` (trilinear)
for smooth MIP rendering.

The :func:`transform_atlas` result is cached by pose hash so the GUI does
not recompute when a user fiddles with a spinbox and then reverts.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy import ndimage

from door_toolkit.atlas_align.config import get_logger

logger = get_logger(__name__)


# Maximum number of (pose_hash → volumes) entries kept in the LRU cache.
_CACHE_MAX_ENTRIES = 4


@dataclass(frozen=True)
class Pose:
    """All 10 degrees of freedom for the atlas pose.

    Angle convention: intrinsic ZYX Euler angles in degrees. Apply order:
    ``flip → scale → rotate → translate``.
    """

    tx: float = 0.0
    ty: float = 0.0
    tz: float = 0.0
    rx: float = 0.0
    ry: float = 0.0
    rz: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    sz: float = 1.0
    flip_x: bool = False
    flip_y: bool = False
    flip_z: bool = False

    def to_tuple(self) -> Tuple:
        return (
            self.tx, self.ty, self.tz,
            self.rx, self.ry, self.rz,
            self.sx, self.sy, self.sz,
            self.flip_x, self.flip_y, self.flip_z,
        )

    def to_dict(self) -> dict:
        return {
            "tx": self.tx, "ty": self.ty, "tz": self.tz,
            "rx": self.rx, "ry": self.ry, "rz": self.rz,
            "sx": self.sx, "sy": self.sy, "sz": self.sz,
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
            "flip_z": self.flip_z,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Pose":
        return cls(
            tx=float(data.get("tx", 0.0)),
            ty=float(data.get("ty", 0.0)),
            tz=float(data.get("tz", 0.0)),
            rx=float(data.get("rx", 0.0)),
            ry=float(data.get("ry", 0.0)),
            rz=float(data.get("rz", 0.0)),
            sx=float(data.get("sx", 1.0)),
            sy=float(data.get("sy", 1.0)),
            sz=float(data.get("sz", 1.0)),
            flip_x=bool(data.get("flip_x", False)),
            flip_y=bool(data.get("flip_y", False)),
            flip_z=bool(data.get("flip_z", False)),
        )

    def digest(self) -> str:
        """Stable hex digest keyed on all 10 DOFs (used as cache key)."""
        payload = repr(self.to_tuple()).encode("utf-8")
        return hashlib.sha1(payload).hexdigest()


@dataclass
class TransformedAtlas:
    """Output of :func:`transform_atlas`."""

    grayscale: np.ndarray
    labelmap: np.ndarray
    pose: Pose
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# Affine matrix construction
# ---------------------------------------------------------------------------


def _rotation_zyx(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Intrinsic ZYX Euler rotation (Rz * Ry * Rx) in 3×3."""
    rx, ry, rz = np.deg2rad([rx_deg, ry_deg, rz_deg])
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    Rx = np.array(
        [[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64
    )
    Ry = np.array(
        [[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64
    )
    Rz = np.array(
        [[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64
    )
    return Rz @ Ry @ Rx


def build_affine_matrix(
    pose: Pose, shape_zyx: Tuple[int, int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct the 3×3 linear part and 3-vector offset for
    :func:`scipy.ndimage.affine_transform`.

    ``affine_transform`` uses *output-to-input* mapping. Given an output
    voxel coordinate ``o``, the input voxel it samples from is
    ``matrix @ o + offset``. We compose the desired forward transform
    ``forward = T * R * S * F`` (translate ∘ rotate ∘ scale ∘ flip)
    around the volume centre, then pass its *inverse* to ndimage.

    Args:
        pose: 10-DOF pose.
        shape_zyx: Volume shape ``(Z, Y, X)``.

    Returns:
        ``(matrix_3x3, offset_3)``: the ``order=0`` inputs to
        :func:`scipy.ndimage.affine_transform`. Arrays are in
        ``(Z, Y, X)`` axis order to match the volume.
    """
    shape = np.asarray(shape_zyx, dtype=np.float64)
    center = (shape - 1.0) / 2.0  # ZYX

    # Build forward transform piece-by-piece in (Z, Y, X) axis order.

    # Flip diagonal: -1 where flip is set, +1 otherwise
    flip_diag = np.array(
        [
            -1.0 if pose.flip_z else 1.0,
            -1.0 if pose.flip_y else 1.0,
            -1.0 if pose.flip_x else 1.0,
        ],
        dtype=np.float64,
    )
    F = np.diag(flip_diag)

    # Scale diagonal — note pose uses (sx, sy, sz) but arrays are (Z, Y, X).
    # sz scales the Z axis, sy scales Y, sx scales X.
    scale_diag = np.array(
        [max(pose.sz, 1e-6), max(pose.sy, 1e-6), max(pose.sx, 1e-6)],
        dtype=np.float64,
    )
    S = np.diag(scale_diag)

    # Rotation. The spec says Euler ZYX in (rx, ry, rz); we build the
    # rotation in world-axis order (X, Y, Z) and then permute to the
    # (Z, Y, X) storage order that ndimage uses.
    R_xyz = _rotation_zyx(pose.rx, pose.ry, pose.rz)  # applied in XYZ world
    perm = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.float64)  # swap XYZ<->ZYX
    R = perm @ R_xyz @ perm.T  # rotation in ZYX axis order

    # Forward linear part
    A_forward = R @ S @ F

    # Translation (pose t* is in world XYZ pixels); reorder to ZYX
    t_zyx = np.array([pose.tz, pose.ty, pose.tx], dtype=np.float64)

    # forward: out = A_forward @ (in - center) + center + t
    # So in terms of (in, out): out = A_forward @ in + (center + t - A_forward @ center)
    # Inverse (needed by ndimage): in = A_inv @ (out - center - t) + center
    #                                  = A_inv @ out + (center - A_inv @ (center + t))
    try:
        A_inv = np.linalg.inv(A_forward)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Non-invertible pose: {pose}") from e

    offset = center - A_inv @ (center + t_zyx)
    return A_inv, offset


# ---------------------------------------------------------------------------
# Pose → resampled volumes
# ---------------------------------------------------------------------------


def _affine_transform_fast(
    volume: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    order: int,
) -> np.ndarray:
    """Thin wrapper around :func:`scipy.ndimage.affine_transform`."""
    return ndimage.affine_transform(
        volume,
        matrix=matrix,
        offset=offset,
        order=order,
        mode="constant",
        cval=0,
        prefilter=(order >= 2),
    )


_cache: list[tuple[str, TransformedAtlas]] = []


def clear_cache() -> None:
    """Drop all cached transformed volumes. Mainly used by tests."""
    _cache.clear()


def _cache_get(key: str) -> TransformedAtlas | None:
    for entry_key, value in _cache:
        if entry_key == key:
            return value
    return None


def _cache_put(key: str, value: TransformedAtlas) -> None:
    for i, (entry_key, _) in enumerate(_cache):
        if entry_key == key:
            del _cache[i]
            break
    _cache.append((key, value))
    while len(_cache) > _CACHE_MAX_ENTRIES:
        _cache.pop(0)


def transform_atlas(
    grayscale: np.ndarray,
    labelmap: np.ndarray,
    pose: Pose,
    use_cache: bool = True,
) -> TransformedAtlas:
    """Resample (grayscale, labelmap) under the given pose.

    Args:
        grayscale: 3D float volume, ``(Z, Y, X)``.
        labelmap: 3D uint16 volume, same shape as ``grayscale``.
        pose: 10-DOF pose.
        use_cache: If True, return a cached result when the pose hash matches.

    Returns:
        :class:`TransformedAtlas` containing the resampled pair.
    """
    if grayscale.shape != labelmap.shape:
        raise ValueError(
            f"grayscale shape {grayscale.shape} != labelmap shape {labelmap.shape}"
        )
    if grayscale.ndim != 3:
        raise ValueError(f"Expected 3D volume, got ndim={grayscale.ndim}")

    # Cache key must include the underlying data location — callers often
    # pass different slices of the same bundle (each has the same pose +
    # shape but different content). Using ``ctypes.data`` (the raw memory
    # pointer) differentiates slice views without hashing 8 MB of data.
    try:
        data_id = int(labelmap.ctypes.data)
    except AttributeError:
        data_id = id(labelmap)
    key = f"{pose.digest()}:{grayscale.shape}:{data_id}"
    if use_cache:
        cached = _cache_get(key)
        if cached is not None:
            logger.debug("transform_atlas cache hit: %s", pose)
            return cached

    t0 = time.perf_counter()
    matrix, offset = build_affine_matrix(pose, grayscale.shape)

    grayscale_out = _affine_transform_fast(
        grayscale.astype(np.float32, copy=False), matrix, offset, order=1
    )
    # CRITICAL: nearest-neighbour for the labelmap so labels stay integer.
    labelmap_out = _affine_transform_fast(
        labelmap.astype(np.uint16, copy=False), matrix, offset, order=0
    ).astype(np.uint16, copy=False)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    logger.debug(
        "transform_atlas pose=%s elapsed=%.1fms shape=%s",
        pose, elapsed_ms, grayscale.shape,
    )

    result = TransformedAtlas(
        grayscale=grayscale_out,
        labelmap=labelmap_out,
        pose=pose,
        elapsed_ms=elapsed_ms,
    )
    if use_cache:
        _cache_put(key, result)
    return result
