"""
Pose JSON I/O
=============

Save / load a 10-DOF :class:`~door_toolkit.atlas_align.core.volume_transform.Pose`
with sidecar metadata (threshold, timestamps, content hashes).
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Optional

from door_toolkit.atlas_align.config import get_logger
from door_toolkit.atlas_align.core.volume_transform import Pose

logger = get_logger(__name__)


def save_pose(
    path: Path,
    pose: Pose,
    threshold: float,
    atlas_hash: Optional[str] = None,
    reference_hash: Optional[str] = None,
) -> Path:
    """Write a pose JSON capturing the user's alignment.

    Args:
        path: Output ``.json`` path.
        pose: Current pose.
        threshold: IoU threshold at save time.
        atlas_hash: Optional SHA-256 of the atlas labelmap TIF.
        reference_hash: Optional SHA-256 of the reference image.

    Returns:
        The path that was written.
    """
    path = Path(path)
    payload = {
        **pose.to_dict(),
        "threshold": float(threshold),
        "atlas_hash": atlas_hash,
        "reference_hash": reference_hash,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved pose JSON: %s", path)
    return path


def load_pose(path: Path) -> tuple[Pose, dict]:
    """Load a pose JSON previously produced by :func:`save_pose`.

    Args:
        path: Path to the JSON file.

    Returns:
        ``(pose, metadata)`` where ``metadata`` carries the non-pose
        fields (threshold, hashes, timestamp).
    """
    path = Path(path)
    data = json.loads(path.read_text())
    pose = Pose.from_dict(data)
    metadata = {
        k: data.get(k)
        for k in ("threshold", "atlas_hash", "reference_hash", "timestamp_utc")
    }
    logger.info("Loaded pose JSON: %s", path)
    return pose, metadata


def file_sha256(path: Path) -> str:
    """Helper: SHA-256 hex digest of a file's contents."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
