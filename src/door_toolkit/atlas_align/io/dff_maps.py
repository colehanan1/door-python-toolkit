"""
Per-pixel ΔF/F map loader
==========================

Walks a trials directory ``fly_X/trial_NNN_OFM_L/images/caiman_pnr_preproc_p70/
al_cropped_stack.tif`` for each odor, computes two per-pixel summary
maps from the raw cropped movie, and caches them so the GUI can look up
real ΔF/F for any user-drawn polygon:

* ``F0_maps[o]`` : mean intensity per pixel across the baseline window
* ``peak_maps[o]``: max intensity per pixel across the odor-presentation window

ΔF/F for a polygon at odor ``o`` is then

    ΔF/F = (mean(peak_maps[o] inside poly) − mean(F0_maps[o] inside poly))
           /  mean(F0_maps[o] inside poly)

Loading 7 × (340, 1269, 1009) float32 movies over the network is slow,
so the maps are cached to ``~/.atlas_align/cache/<fly>_dff_maps.npz``
on first run and reloaded instantly on subsequent launches.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

from door_toolkit.atlas_align.config import USER_STATE_DIR, get_logger
from door_toolkit.atlas_align.io.dff_loader import DEFAULT_ODOR_LETTER_MAP

logger = get_logger(__name__)

#: Filename of the pre-cropped movie stack inside each trial folder.
CROPPED_STACK_REL = Path("images") / "caiman_pnr_preproc_p70" / "al_cropped_stack.tif"

#: Regex that recognises a trial folder like ``trial_001_OFM_A``.
_TRIAL_DIR_RE = re.compile(r"^trial_(?P<num>\d+)_OFM_(?P<letter>[A-Za-z])$")

DEFAULT_CACHE_DIR = USER_STATE_DIR / "cache"


@dataclass
class DffMaps:
    """Per-odor baseline and peak intensity maps ready for polygon lookup.

    Attributes:
        odor_order: Odor names in the canonical order (matches whatever
            ordering the caller passed in; typically the dff/activity
            bundle's).
        F0_maps: ``(n_odors, H, W)`` float32 per-pixel baseline mean.
        peak_maps: ``(n_odors, H, W)`` float32 per-pixel odor-window max.
        map_to_ref_offset: ``(row, col)`` offset mapping the cropped stack
            coords → the fullframe reference image's pixel coords. Used
            to translate polygon coords drawn on the fullframe into the
            cropped map's coordinate system.
        source_dir: Path to the trials directory we loaded from.
        baseline_window: ``(start, end)`` frame range (end exclusive) used
            for the F0 average.
        odor_window: ``(start, end)`` frame range (end exclusive) used for
            the peak extraction.
    """

    odor_order: List[str]
    F0_maps: np.ndarray
    peak_maps: np.ndarray
    map_to_ref_offset: Tuple[int, int] = (0, 0)
    source_dir: Optional[Path] = None
    baseline_window: Tuple[int, int] = (0, 150)
    odor_window: Tuple[int, int] = (150, 190)

    @property
    def n_odors(self) -> int:
        return int(self.F0_maps.shape[0])

    @property
    def map_shape(self) -> Tuple[int, int]:
        return int(self.F0_maps.shape[1]), int(self.F0_maps.shape[2])

    def response_for_polygon(
        self,
        xs_ref: np.ndarray,
        ys_ref: np.ndarray,
    ) -> np.ndarray:
        """Compute a real ΔF/F vector for a polygon in fullframe coords."""
        from skimage.draw import polygon as sk_polygon

        H, W = self.map_shape
        off_r, off_c = self.map_to_ref_offset
        xs_map = np.asarray(xs_ref, dtype=np.float64) - off_c
        ys_map = np.asarray(ys_ref, dtype=np.float64) - off_r
        rr, cc = sk_polygon(ys_map, xs_map, shape=(H, W))
        out = np.zeros(self.n_odors, dtype=np.float32)
        if rr.size == 0:
            return out
        for z in range(self.n_odors):
            f0 = float(np.nanmean(self.F0_maps[z, rr, cc]))
            peak = float(np.nanmean(self.peak_maps[z, rr, cc]))
            if not np.isfinite(f0) or f0 == 0.0:
                out[z] = 0.0
            else:
                out[z] = (peak - f0) / f0
        return out


def _cache_path_for(trials_dir: Path, cache_dir: Path) -> Path:
    """Deterministic cache path per trials-dir."""
    digest = hashlib.md5(str(trials_dir.resolve()).encode()).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9_]+", "_", trials_dir.name)
    return cache_dir / f"dff_maps_{safe}_{digest}.npz"


def _find_trial_for_letter(trials_dir: Path, letter: str) -> Optional[Path]:
    """Return the first ``trial_*_OFM_<letter>/images/.../al_cropped_stack.tif``
    found (sorted by trial number)."""
    candidates: List[Tuple[int, Path]] = []
    for sub in trials_dir.iterdir():
        if not sub.is_dir():
            continue
        m = _TRIAL_DIR_RE.match(sub.name)
        if m is None:
            continue
        if m.group("letter").upper() != letter.upper():
            continue
        stack_path = sub / CROPPED_STACK_REL
        if stack_path.is_file():
            candidates.append((int(m.group("num")), stack_path))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def _compute_per_pixel_maps(
    stack: np.ndarray,
    baseline_window: Tuple[int, int],
    odor_window: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(F0_map, peak_map)`` for a single (T, H, W) stack."""
    bs, be = baseline_window
    os_, oe = odor_window
    bs = max(0, bs)
    be = min(stack.shape[0], be)
    os_ = max(0, os_)
    oe = min(stack.shape[0], oe)
    if be <= bs or oe <= os_:
        raise ValueError(
            f"Bad baseline/odor windows: baseline={baseline_window}, "
            f"odor={odor_window}, stack_frames={stack.shape[0]}"
        )
    F0 = stack[bs:be].mean(axis=0, dtype=np.float32)
    peak = stack[os_:oe].max(axis=0).astype(np.float32)
    return F0, peak


def _cache_sidecar(cache_path: Path) -> Path:
    return cache_path.with_suffix(".json")


def load_dff_maps(
    trials_dir: Path,
    odor_order: List[str],
    *,
    baseline_window: Tuple[int, int] = (0, 150),
    odor_window: Tuple[int, int] = (150, 190),
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_recompute: bool = False,
    fullframe_shape: Optional[Tuple[int, int]] = None,
    odor_letter_map: Optional[Dict[str, str]] = None,
) -> DffMaps:
    """Compute (or reload from cache) per-pixel F0 and peak maps.

    Args:
        trials_dir: Directory containing ``trial_NNN_OFM_L`` subfolders.
        odor_order: Canonical odor-name order to produce maps in.
        baseline_window: Frame range (start, end-exclusive) for F₀.
        odor_window: Frame range (start, end-exclusive) for peak.
        cache_dir: Where to cache the per-pixel maps.
        force_recompute: Ignore existing cache and rebuild.
        fullframe_shape: Optional ``(H_full, W_full)`` for the fullframe
            reference — if given, the returned offset centres the cropped
            maps inside that shape.
        odor_letter_map: Override for the letter → canonical-odor map.
    """
    trials_dir = Path(trials_dir).expanduser().resolve()
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not trials_dir.is_dir():
        raise FileNotFoundError(f"--trials-dir not found: {trials_dir}")

    odor_letter_map = dict(odor_letter_map or DEFAULT_ODOR_LETTER_MAP)
    name_to_letter = {v: k for k, v in odor_letter_map.items()}

    cache_path = _cache_path_for(trials_dir, cache_dir)
    sidecar_path = _cache_sidecar(cache_path)
    signature = {
        "odor_order": list(odor_order),
        "baseline_window": list(baseline_window),
        "odor_window": list(odor_window),
    }

    # Try cache hit first. Two files on disk: .npz (arrays) + .json (sig).
    if (
        cache_path.is_file()
        and sidecar_path.is_file()
        and not force_recompute
    ):
        try:
            with np.load(cache_path) as saved:
                sidecar = json.loads(sidecar_path.read_text())
                if sidecar.get("signature") == signature:
                    logger.info("Loaded cached dff maps: %s", cache_path)
                    return DffMaps(
                        odor_order=list(sidecar["odor_order_resolved"]),
                        F0_maps=saved["F0_maps"],
                        peak_maps=saved["peak_maps"],
                        map_to_ref_offset=tuple(saved["offset"].tolist()),
                        source_dir=trials_dir,
                        baseline_window=baseline_window,
                        odor_window=odor_window,
                    )
                logger.info(
                    "Cache at %s has different signature — recomputing.",
                    cache_path,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Could not load dff-maps cache %s (%s) — recomputing.",
                cache_path, e,
            )

    # Compute per odor.
    F0_stack: List[np.ndarray] = []
    peak_stack: List[np.ndarray] = []
    kept_odor_order: List[str] = []
    first_shape: Optional[Tuple[int, int]] = None

    for odor_name in odor_order:
        letter = name_to_letter.get(odor_name)
        if letter is None:
            logger.warning(
                "No odor-letter for %r in DEFAULT_ODOR_LETTER_MAP; skipping.",
                odor_name,
            )
            continue
        stack_path = _find_trial_for_letter(trials_dir, letter)
        if stack_path is None:
            logger.warning(
                "No al_cropped_stack.tif found for odor %s (letter %s) under %s.",
                odor_name, letter, trials_dir,
            )
            continue
        logger.info(
            "Reading raw stack for %s (%s)...",
            odor_name, stack_path.relative_to(trials_dir),
        )
        stack = tifffile.imread(str(stack_path))
        if stack.ndim != 3:
            logger.warning(
                "Unexpected shape for %s: %s — skipping.",
                stack_path, stack.shape,
            )
            continue
        if first_shape is None:
            first_shape = (int(stack.shape[1]), int(stack.shape[2]))
        elif (stack.shape[1], stack.shape[2]) != first_shape:
            logger.warning(
                "Stack %s shape %s doesn't match first %s — skipping.",
                stack_path, stack.shape, first_shape,
            )
            continue
        F0, peak = _compute_per_pixel_maps(
            stack.astype(np.float32, copy=False),
            baseline_window,
            odor_window,
        )
        F0_stack.append(F0)
        peak_stack.append(peak)
        kept_odor_order.append(odor_name)
        logger.info(
            "  %s: F0 mean=%.2f, peak mean=%.2f (shape %s)",
            odor_name, float(F0.mean()), float(peak.mean()), F0.shape,
        )

    if not kept_odor_order:
        raise RuntimeError(
            f"No raw stacks could be loaded from {trials_dir}."
        )

    F0_arr = np.stack(F0_stack, axis=0)
    peak_arr = np.stack(peak_stack, axis=0)

    offset: Tuple[int, int] = (0, 0)
    if fullframe_shape is not None:
        ch, cw = F0_arr.shape[1], F0_arr.shape[2]
        fh, fw = fullframe_shape
        offset = ((fh - ch) // 2, (fw - cw) // 2)

    # Persist to cache: .npz for the numeric arrays, .json for the signature.
    try:
        np.savez_compressed(
            cache_path,
            F0_maps=F0_arr,
            peak_maps=peak_arr,
            offset=np.asarray(offset, dtype=np.int32),
        )
        sidecar_path.write_text(
            json.dumps(
                {
                    "signature": signature,
                    "odor_order_resolved": kept_odor_order,
                    "trials_dir": str(trials_dir),
                },
                indent=2,
            )
        )
        logger.info(
            "Cached dff maps → %s (%.1f MB)",
            cache_path, cache_path.stat().st_size / 1e6,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not write dff-maps cache %s: %s", cache_path, e)

    return DffMaps(
        odor_order=kept_odor_order,
        F0_maps=F0_arr,
        peak_maps=peak_arr,
        map_to_ref_offset=offset,
        source_dir=trials_dir,
        baseline_window=baseline_window,
        odor_window=odor_window,
    )
