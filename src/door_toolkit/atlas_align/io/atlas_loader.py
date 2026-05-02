"""
Atlas bundle loader
===================

Reads the directory produced by
:func:`door_toolkit.atlas_align.atlas_builder.build_labelmap` into a
single :class:`AtlasBundle` that the GUI and core modules can consume.

Expected files inside the atlas directory::

    flywire_al_labelmap.tif     (uint16, required)
    flywire_al_labels.json      (dict[int, str], required)
    flywire_al_grayscale.tif    (optional; MIP of labelmap used as fallback)
    build_manifest.json         (optional but recommended)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import tifffile

from door_toolkit.atlas_align.config import get_logger

logger = get_logger(__name__)

LABELMAP_FILENAME = "flywire_al_labelmap.tif"
LABELS_FILENAME = "flywire_al_labels.json"
GRAYSCALE_FILENAME = "flywire_al_grayscale.tif"
MANIFEST_FILENAME = "build_manifest.json"


@dataclass
class AtlasBundle:
    """Container for the atlas artefacts the GUI needs at runtime.

    Attributes:
        labelmap: 3D uint16 array, shape ``(Z, Y, X)``. ``0`` = background.
        grayscale: 3D float32 array, same shape as ``labelmap``. When the
            builder does not emit a grayscale TIF, this is synthesised as
            a binary mask from the labelmap (non-zero → 1.0) so the GUI
            can still show a meaningful anatomical reference.
        labels: Mapping from integer label id → glomerulus name.
        spacing_um: Voxel spacing (Z, Y, X) in microns; ``None`` if the
            source TIF did not carry that metadata.
        manifest: Raw contents of ``build_manifest.json`` if present.
        atlas_dir: Absolute path to the directory we loaded from.
        grayscale_synthesised: True when the grayscale channel was built
            from the labelmap because no dedicated grayscale TIF was
            found.
    """

    labelmap: np.ndarray
    grayscale: np.ndarray
    labels: Dict[int, str]
    spacing_um: Optional[Tuple[float, float, float]] = None
    manifest: Dict = field(default_factory=dict)
    atlas_dir: Optional[Path] = None
    grayscale_synthesised: bool = False

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Volume shape ``(Z, Y, X)``."""
        return tuple(int(d) for d in self.labelmap.shape)  # type: ignore[return-value]

    @property
    def n_labels(self) -> int:
        """Number of glomeruli (distinct non-zero labels)."""
        return len(self.labels)

    def label_name(self, idx: int) -> str:
        """Return the name for label id ``idx``, or ``UNK_{idx}`` if missing."""
        return self.labels.get(int(idx), f"UNK_{int(idx)}")


def _read_labels_json(path: Path) -> Dict[int, str]:
    """Read the labels JSON and normalise keys to :class:`int`."""
    raw = json.loads(path.read_text())
    if not isinstance(raw, Mapping):
        raise ValueError(f"Labels JSON must be a dict, got {type(raw).__name__}.")
    parsed: Dict[int, str] = {}
    for key, value in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Invalid label key {key!r} in {path}: must be int-castable."
            ) from e
        parsed[idx] = str(value)
    return parsed


def _read_tif_with_spacing(
    path: Path,
) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
    """Read a 3D TIF and try to pull (Z, Y, X) spacing from metadata."""
    with tifffile.TiffFile(str(path)) as tf:
        array = tf.asarray()
        spacing: Optional[Tuple[float, float, float]] = None
        try:
            tags = tf.pages[0].tags
            if "ImageDescription" in tags:
                desc = tags["ImageDescription"].value
                meta = json.loads(desc) if isinstance(desc, str) else {}
                if isinstance(meta, dict) and "spacing_um" in meta:
                    s = meta["spacing_um"]
                    if len(s) == 3:
                        spacing = (float(s[0]), float(s[1]), float(s[2]))
        except Exception:  # noqa: BLE001 — metadata parsing is best-effort
            spacing = None
    return array, spacing


def _grayscale_from_labelmap(labelmap: np.ndarray) -> np.ndarray:
    """Synthesise a grayscale volume when no dedicated channel exists.

    The fallback is simply a float32 binary mask (``labelmap > 0``). It's
    not a realistic anatomical reference, but it gives the GUI something
    to project for alignment when the user only has the labelmap.
    """
    return (labelmap > 0).astype(np.float32)


def load_atlas_bundle(atlas_dir: Path) -> AtlasBundle:
    """Load an :class:`AtlasBundle` from the builder's output directory.

    Args:
        atlas_dir: Directory containing at minimum ``flywire_al_labelmap.tif``
            and ``flywire_al_labels.json``.

    Returns:
        A fully-populated :class:`AtlasBundle`.

    Raises:
        FileNotFoundError: if either required file is missing.
        ValueError: if the labelmap is not 3D or the labels JSON is malformed.
    """
    atlas_dir = Path(atlas_dir).expanduser().resolve()
    logger.debug("load_atlas_bundle(atlas_dir=%s)", atlas_dir)
    if not atlas_dir.is_dir():
        raise FileNotFoundError(f"Atlas directory does not exist: {atlas_dir}")

    labelmap_path = atlas_dir / LABELMAP_FILENAME
    labels_path = atlas_dir / LABELS_FILENAME
    grayscale_path = atlas_dir / GRAYSCALE_FILENAME
    manifest_path = atlas_dir / MANIFEST_FILENAME

    if not labelmap_path.is_file():
        raise FileNotFoundError(f"Required labelmap missing: {labelmap_path}")
    if not labels_path.is_file():
        raise FileNotFoundError(f"Required labels JSON missing: {labels_path}")

    labelmap, spacing = _read_tif_with_spacing(labelmap_path)
    if labelmap.ndim != 3:
        raise ValueError(
            f"Expected 3D labelmap, got shape {labelmap.shape} from {labelmap_path}"
        )
    labelmap = np.ascontiguousarray(labelmap, dtype=np.uint16)

    labels = _read_labels_json(labels_path)

    grayscale_synthesised = False
    if grayscale_path.is_file():
        logger.debug("Reading grayscale from %s", grayscale_path)
        grayscale_raw, gray_spacing = _read_tif_with_spacing(grayscale_path)
        if grayscale_raw.shape != labelmap.shape:
            logger.warning(
                "grayscale shape %s != labelmap shape %s; falling back to MIP.",
                grayscale_raw.shape, labelmap.shape,
            )
            grayscale = _grayscale_from_labelmap(labelmap)
            grayscale_synthesised = True
        else:
            grayscale = grayscale_raw.astype(np.float32, copy=False)
            if spacing is None and gray_spacing is not None:
                spacing = gray_spacing
    else:
        logger.info(
            "No %s found in %s; synthesising grayscale from labelmap.",
            GRAYSCALE_FILENAME, atlas_dir,
        )
        grayscale = _grayscale_from_labelmap(labelmap)
        grayscale_synthesised = True

    manifest: Dict = {}
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
            if spacing is None and "template_spacing_um_zyx" in manifest:
                s = manifest["template_spacing_um_zyx"]
                if len(s) == 3:
                    spacing = (float(s[0]), float(s[1]), float(s[2]))
        except json.JSONDecodeError as e:
            logger.warning("Manifest JSON malformed at %s: %s", manifest_path, e)

    bundle = AtlasBundle(
        labelmap=labelmap,
        grayscale=grayscale,
        labels=labels,
        spacing_um=spacing,
        manifest=manifest,
        atlas_dir=atlas_dir,
        grayscale_synthesised=grayscale_synthesised,
    )
    logger.info(
        "Loaded atlas: shape=%s, %d labels, grayscale=%s, spacing_um=%s",
        bundle.shape,
        bundle.n_labels,
        "synth" if grayscale_synthesised else "file",
        spacing,
    )
    return bundle
