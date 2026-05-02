"""
DoOR 2D polygon atlas builder (multi-view)
===========================================

Rasterises the ``door_AL_map.RData`` atlas (shipped with ropensci
``DoOR.data``) into a **stack of 5 slice views**:

* Z=0 — slice1 (most posterior)
* Z=1 — slice2
* Z=2 — slice3
* Z=3 — slice4 (most anterior)
* Z=4 — VP (thermo/hygro glomeruli)

Each slice view is a *separate* 2D labelmap showing only the glomeruli
that appear in that AL depth. The GUI treats the stack as a 5-frame
"slideshow": pressing Space cycles the active frame so the user can
visually compare each DoOR slice to their own imaging plane while the
reference image stays at its native resolution.

The builder accepts ``--reference-shape H,W`` so the rasterisation lands
on the user's reference grid directly — no resizing / aspect distortion
at GUI startup.

Outputs mirror the FlyWire builder:
    flywire_al_labelmap.tif    uint16, shape (5, H, W)
    flywire_al_labels.json     {id: glomerulus_name}
    flywire_al_grayscale.tif   float32, shape (5, H, W); per-slice bg contours
    build_manifest.json        { atlas_type: "door_2d_multiview", ... }
    qc/mip_z.png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tifffile

LOG = logging.getLogger("door_toolkit.atlas_align.atlas_builder.door_2d")

#: Atlas file (GitHub raw URL, ropensci mirror). ~400 kB.
DEFAULT_RDATA_URL = (
    "https://github.com/ropensci/DoOR.data/raw/master/data/door_AL_map.RData"
)

#: Default output canvas size for a single slice view.
DEFAULT_SHAPE_HW = (1024, 1024)

#: Slice-subset x ranges in DoOR atlas coordinates. Matches the four
#: ``background`` panels exactly so each view shows ONE tight cluster of
#: glomeruli (one AL hemisphere / cross-section), not two.
#:
#: Glomeruli that the DoOR atlas draws outside these panels (label area,
#: thermo/hygro annotations) are captured by the "extras" view below.
_SLICE_X_RANGES = {
    "slice4": (-10.0, 62.0),    # most anterior
    "slice3": (62.0, 128.0),
    "slice2": (128.0, 197.0),
    "slice1": (197.0, 247.0),
}
#: Glomeruli with centroid x > 247 are drawn outside the four slice panels
#: in the DoOR map (typically VP + a handful of misc annotations). We group
#: them into an "extras" view rather than force them into slice1.
_EXTRAS_X_MIN = 247.0
#: The VP thermo/hygro set lives in ``unmapped_not.olf`` in the RData file.
_VP_VIEW_NAME = "VP"
#: Catch-all view for glomerulus polygons drawn outside the 4 slice panels.
_EXTRAS_VIEW_NAME = "extras"

#: Canonical ordering used for Z-axis indexing (spacebar cycles this).
VIEW_ORDER = (
    "slice1", "slice2", "slice3", "slice4",
    _VP_VIEW_NAME, _EXTRAS_VIEW_NAME,
)


@dataclass
class DoorAtlasData:
    """Parsed DoOR AL map with per-view polygon groupings."""

    # view_name -> {glomerulus_name -> (N, 2) polygon vertices}
    glomeruli_by_view: Dict[str, Dict[str, np.ndarray]]
    # view_name -> list of background polylines for that slice
    background_by_view: Dict[str, List[np.ndarray]]
    # view_name -> list of cutout polygons for that slice
    bg_cutouts_by_view: Dict[str, List[np.ndarray]]
    # flat glomerulus -> label anchor
    labels: Dict[str, Tuple[float, float]]
    # view_name -> ((xmin, xmax), (ymin, ymax))
    view_ranges: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]]


def _split_polygons_by_gap(
    xs: np.ndarray, ys: np.ndarray, gap_threshold: float = 5.0
) -> List[np.ndarray]:
    """Split a vertex sequence into separate polygons at large coordinate gaps.

    R ``polygon()`` sequences can contain ``NA`` rows to separate disjoint
    parts; after pandas round-tripping those can become large jumps in
    (x, y). We split whenever the per-step displacement exceeds
    ``gap_threshold`` (in DoOR atlas units — the whole atlas is ~300
    units wide, so 5 is a conservative value).
    """
    if len(xs) == 0:
        return []
    diffs = np.abs(np.diff(xs)) + np.abs(np.diff(ys))
    splits = np.where(diffs > gap_threshold)[0] + 1
    parts = np.split(np.column_stack([xs, ys]), splits)
    return [p for p in parts if len(p) >= 3]


def _which_slice(x: float) -> Optional[str]:
    """Return the view a given DoOR-atlas x-coordinate falls in.

    Returns the slice name if ``x`` is inside one of the four panels,
    or ``"extras"`` if it's past them (outside-panel label area).
    """
    for slice_name, (xmin, xmax) in _SLICE_X_RANGES.items():
        if xmin <= x <= xmax:
            return slice_name
    if x >= _EXTRAS_X_MIN:
        return _EXTRAS_VIEW_NAME
    return None


def load_door_rdata(path: Path) -> DoorAtlasData:
    """Parse ``door_AL_map.RData`` into a :class:`DoorAtlasData`."""
    import rdata

    path = Path(path)
    LOG.info("Reading DoOR AL map: %s", path)
    parsed = rdata.parser.parse_file(str(path))
    converted = rdata.conversion.convert(parsed)
    if "door_AL_map" not in converted:
        raise ValueError(
            f"{path} does not contain 'door_AL_map' object; got "
            f"{list(converted.keys())}"
        )
    m = converted["door_AL_map"]

    glom_df = m["glomeruli"]
    unmapped_df = m.get("unmapped_not.olf")
    bg_df = m["background"]
    labels_df = m["labels"]
    cutout_df = m.get("bg.cutout")

    glomeruli_by_view: Dict[str, Dict[str, np.ndarray]] = {
        v: {} for v in VIEW_ORDER
    }

    # Main olfactory glomeruli → assigned to slice1..slice4 by x-coord
    for name, sub in glom_df.groupby("glomerulus", observed=True):
        xs = sub["x"].to_numpy(dtype=np.float32)
        ys = sub["y"].to_numpy(dtype=np.float32)
        parts = _split_polygons_by_gap(xs, ys)
        if not parts:
            continue
        largest = max(parts, key=len).astype(np.float32)
        # Determine slice from the centroid x
        cx = float(largest[:, 0].mean())
        view = _which_slice(cx)
        if view is None:
            LOG.debug("  %s centroid x=%.1f outside known slice ranges — skipping",
                      name, cx)
            continue
        glomeruli_by_view[view][str(name)] = largest

    # VP (thermo/hygro) glomeruli live in unmapped_not.olf
    if unmapped_df is not None and len(unmapped_df):
        for name, sub in unmapped_df.groupby("glomerulus", observed=True):
            xs = sub["x"].to_numpy(dtype=np.float32)
            ys = sub["y"].to_numpy(dtype=np.float32)
            parts = _split_polygons_by_gap(xs, ys)
            if not parts:
                continue
            largest = max(parts, key=len).astype(np.float32)
            glomeruli_by_view[_VP_VIEW_NAME][str(name)] = largest

    for v in VIEW_ORDER:
        LOG.info("  %s: %d glomeruli", v, len(glomeruli_by_view[v]))

    # Background polylines by view (group column like "slice1"/"slice2"/...
    # corresponds 1:1 to the VIEW_ORDER olfactory slices; VP has no bg).
    background_by_view: Dict[str, List[np.ndarray]] = {v: [] for v in VIEW_ORDER}
    for group_name, sub in bg_df.groupby("group", observed=True):
        xs = sub["x"].to_numpy(dtype=np.float32)
        ys = sub["y"].to_numpy(dtype=np.float32)
        view = str(group_name)
        if view in background_by_view:
            background_by_view[view].extend(_split_polygons_by_gap(xs, ys))

    bg_cutouts_by_view: Dict[str, List[np.ndarray]] = {v: [] for v in VIEW_ORDER}
    if cutout_df is not None and len(cutout_df):
        for _, sub in cutout_df.groupby("group", observed=True):
            xs = sub["x"].to_numpy(dtype=np.float32)
            ys = sub["y"].to_numpy(dtype=np.float32)
            parts = _split_polygons_by_gap(xs, ys)
            # Cutouts don't carry an explicit view assignment; attach to
            # slice based on centroid x.
            for poly in parts:
                cx = float(poly[:, 0].mean())
                view = _which_slice(cx) or _VP_VIEW_NAME
                bg_cutouts_by_view.setdefault(view, []).append(poly)

    labels: Dict[str, Tuple[float, float]] = {}
    for _, row in labels_df.iterrows():
        labels[str(row["glomerulus"])] = (
            float(row["x"]),
            float(row["y"]),
        )

    # Per-view coordinate ranges
    view_ranges: Dict[str, Tuple[Tuple[float, float], Tuple[float, float]]] = {}
    for view in VIEW_ORDER:
        polys = list(glomeruli_by_view[view].values())
        if not polys:
            view_ranges[view] = ((0.0, 1.0), (0.0, 1.0))
            continue
        xs = np.concatenate([p[:, 0] for p in polys])
        ys = np.concatenate([p[:, 1] for p in polys])
        view_ranges[view] = (
            (float(xs.min()), float(xs.max())),
            (float(ys.min()), float(ys.max())),
        )

    return DoorAtlasData(
        glomeruli_by_view=glomeruli_by_view,
        background_by_view=background_by_view,
        bg_cutouts_by_view=bg_cutouts_by_view,
        labels=labels,
        view_ranges=view_ranges,
    )


def _rasterize_one_view(
    glomeruli: Dict[str, np.ndarray],
    background: List[np.ndarray],
    cutouts: List[np.ndarray],
    view_range: Tuple[Tuple[float, float], Tuple[float, float]],
    shape_hw: Tuple[int, int],
    label_base: Dict[str, int],
    margin_frac: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rasterize a single view's glomeruli + background into (H, W) arrays.

    Returns:
        ``(labelmap_2d, grayscale_2d)``.
    """
    from skimage.draw import polygon as sk_polygon, polygon_perimeter

    H, W = int(shape_hw[0]), int(shape_hw[1])
    (xmin, xmax), (ymin, ymax) = view_range
    mx = max((xmax - xmin) * margin_frac, 1.0)
    my = max((ymax - ymin) * margin_frac, 1.0)
    xmin -= mx; xmax += mx
    ymin -= my; ymax += my

    # Preserve aspect: scale by the smaller factor, centre the other axis.
    sx = (W - 1) / (xmax - xmin)
    sy = (H - 1) / (ymax - ymin)
    s = min(sx, sy)
    # Centred offset so the scaled content sits in the middle of the canvas
    content_w = (xmax - xmin) * s
    content_h = (ymax - ymin) * s
    off_x = (W - 1 - content_w) / 2.0
    off_y = (H - 1 - content_h) / 2.0

    def to_px(xs: np.ndarray, ys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        px = (xs - xmin) * s + off_x
        py = (ys - ymin) * s + off_y
        return py, px  # (row, col)

    labelmap_2d = np.zeros((H, W), dtype=np.uint16)
    for name, poly in sorted(glomeruli.items()):
        idx = label_base[name]
        py, px = to_px(poly[:, 0], poly[:, 1])
        rr, cc = sk_polygon(py, px, shape=(H, W))
        free = labelmap_2d[rr, cc] == 0
        labelmap_2d[rr[free], cc[free]] = idx

    grayscale_2d = np.zeros((H, W), dtype=np.float32)
    for poly in background:
        if len(poly) < 3:
            continue
        py, px = to_px(poly[:, 0], poly[:, 1])
        rr, cc = polygon_perimeter(py, px, shape=(H, W))
        grayscale_2d[rr, cc] = 1.0
    for poly in cutouts:
        if len(poly) < 3:
            continue
        py, px = to_px(poly[:, 0], poly[:, 1])
        rr, cc = polygon_perimeter(py, px, shape=(H, W))
        grayscale_2d[rr, cc] = 0.6

    return labelmap_2d, grayscale_2d


def rasterize_multiview(
    atlas: DoorAtlasData,
    shape_hw: Tuple[int, int],
    margin_frac: float = 0.08,
) -> Tuple[np.ndarray, Dict[int, str], np.ndarray, List[str]]:
    """Rasterize each view into its own Z slice.

    The 5 views are stacked along axis 0 in :data:`VIEW_ORDER`.

    Returns:
        ``(labelmap, label_lookup, grayscale, view_names)``:

        * ``labelmap``: ``(5, H, W)`` uint16
        * ``label_lookup``: global ``{id -> glomerulus_name}`` (ids are
          consistent across views so the GUI palette stays stable)
        * ``grayscale``: ``(5, H, W)`` float32 background contours
        * ``view_names``: the slice-name at each Z index
    """
    # Build a global id scheme so the same glomerulus gets the same colour
    # regardless of which slice is currently visible.
    all_names: List[str] = []
    seen: set = set()
    for view in VIEW_ORDER:
        for name in sorted(atlas.glomeruli_by_view[view].keys()):
            if name not in seen:
                seen.add(name)
                all_names.append(name)
    label_base: Dict[str, int] = {n: i + 1 for i, n in enumerate(all_names)}
    label_lookup: Dict[int, str] = {i + 1: n for i, n in enumerate(all_names)}
    if len(all_names) > np.iinfo(np.uint16).max:
        raise ValueError("More glomeruli than uint16 can encode.")

    H, W = int(shape_hw[0]), int(shape_hw[1])
    labelmap_stack = np.zeros((len(VIEW_ORDER), H, W), dtype=np.uint16)
    grayscale_stack = np.zeros((len(VIEW_ORDER), H, W), dtype=np.float32)

    for z, view in enumerate(VIEW_ORDER):
        glomeruli = atlas.glomeruli_by_view[view]
        if not glomeruli:
            LOG.warning("  view %s: no glomeruli — skipping Z=%d", view, z)
            continue
        lm2, gr2 = _rasterize_one_view(
            glomeruli=glomeruli,
            background=atlas.background_by_view.get(view, []),
            cutouts=atlas.bg_cutouts_by_view.get(view, []),
            view_range=atlas.view_ranges[view],
            shape_hw=(H, W),
            label_base=label_base,
            margin_frac=margin_frac,
        )
        labelmap_stack[z] = lm2
        grayscale_stack[z] = gr2
        LOG.info(
            "  view %s at Z=%d: %d glomeruli, %.2f%% pixels filled",
            view, z, len(glomeruli),
            100.0 * np.count_nonzero(lm2) / lm2.size,
        )

    return labelmap_stack, label_lookup, grayscale_stack, list(VIEW_ORDER)


def write_qc_mips(
    labelmap: np.ndarray,
    label_lookup: Dict[int, str],
    view_names: List[str],
    qc_dir: Path,
) -> None:
    """Write one PNG per view for quick inspection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    qc_dir.mkdir(parents=True, exist_ok=True)
    for z, name in enumerate(view_names):
        fig, ax = plt.subplots(figsize=(10, 10), dpi=100)
        ax.imshow(labelmap[z], cmap="tab20", interpolation="nearest")
        unique = np.unique(labelmap[z])
        ax.set_title(
            f"DoOR view '{name}' (Z={z}, {len(unique) - 1} glomeruli)"
        )
        ax.axis("off")
        fig.savefig(qc_dir / f"view_{z}_{name}.png", bbox_inches="tight")
        plt.close(fig)


def _color_for_label_rgb(name: str) -> Tuple[int, int, int]:
    """Deterministic RGB triple for a glomerulus name (same palette as GUI)."""
    digest = hashlib.md5(name.encode("utf-8")).digest()
    hue = digest[0] / 255.0 * 360.0
    sat = (180 + digest[1] % 60) / 255.0
    val = (200 + digest[2] % 55) / 255.0
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue / 360.0, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


def write_per_view_tifs(
    labelmap: np.ndarray,
    grayscale: np.ndarray,
    label_lookup: Dict[int, str],
    view_names: List[str],
    views_dir: Path,
) -> None:
    """Write per-view TIFs: raw labelmap + RGB figure with baked labels.

    For each Z-slice, this produces two files::

        views/<view>_labelmap.tif     uint16, IDs only
        views/<view>_labeled.tif      RGB uint8, coloured polygons +
                                      glomerulus name text overlaid

    The "labeled" TIF is self-contained (opens in any image viewer with
    glomerulus names visible on top of each polygon); the "labelmap" TIF
    is the programmatic source for IoU pipelines.
    """
    from PIL import Image, ImageDraw, ImageFont
    from skimage.segmentation import find_boundaries

    views_dir.mkdir(parents=True, exist_ok=True)

    # Try to use a real TTF for nicer text; fall back to Pillow's builtin
    # bitmap font if nothing is installed.
    font = None
    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ):
        try:
            font = ImageFont.truetype(candidate, size=28)
            break
        except (OSError, IOError):
            continue
    if font is None:
        font = ImageFont.load_default()

    for z, name in enumerate(view_names):
        slice_lm = labelmap[z]
        slice_gray = grayscale[z]

        # 1) Raw uint16 labelmap TIF — mirrors flywire_al_labelmap[z]
        tifffile.imwrite(
            views_dir / f"{name}_labelmap.tif",
            slice_lm,
            compression="zlib",
            metadata={"axes": "YX", "glomerulus_labels": {
                str(gid): label_lookup[gid] for gid in np.unique(slice_lm)
                if int(gid) != 0 and int(gid) in label_lookup
            }},
        )

        # 2) RGB figure TIF with coloured polygons + glomerulus name text
        H, W = slice_lm.shape
        rgb = np.zeros((H, W, 3), dtype=np.uint8)

        # Paint the grayscale slice-outline contours as dim lines.
        gray_mask = slice_gray > 0
        rgb[gray_mask] = [80, 80, 80]

        # Fill each glomerulus polygon with its colour.
        unique = np.unique(slice_lm)
        for gid in unique:
            gid_int = int(gid)
            if gid_int == 0:
                continue
            glom_name = label_lookup.get(gid_int, f"UNK_{gid_int}")
            r, g, b = _color_for_label_rgb(glom_name)
            mask = slice_lm == gid_int
            rgb[mask] = [r, g, b]

        # Darken boundaries so neighbouring gloms are distinguishable.
        boundaries = find_boundaries(slice_lm, mode="outer")
        rgb[boundaries] = [255, 255, 255]

        img = Image.fromarray(rgb, mode="RGB")
        draw = ImageDraw.Draw(img)

        # Centroid text label per glomerulus.
        for gid in unique:
            gid_int = int(gid)
            if gid_int == 0:
                continue
            glom_name = label_lookup.get(gid_int, f"UNK_{gid_int}")
            mask = slice_lm == gid_int
            if mask.sum() < 8:
                continue
            ys, xs = np.where(mask)
            cy = float(ys.mean())
            cx = float(xs.mean())
            # Measure text so we can centre it.
            try:
                bbox = draw.textbbox((0, 0), glom_name, font=font)
                tw = bbox[2] - bbox[0]
                th = bbox[3] - bbox[1]
            except AttributeError:
                tw, th = draw.textsize(glom_name, font=font)
            # Draw a dark outline by offsetting 2 px in each direction.
            for dx in (-2, 0, 2):
                for dy in (-2, 0, 2):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text(
                        (cx - tw / 2 + dx, cy - th / 2 + dy),
                        glom_name, fill=(0, 0, 0), font=font,
                    )
            draw.text(
                (cx - tw / 2, cy - th / 2),
                glom_name, fill=(255, 255, 255), font=font,
            )

        out_rgb = np.asarray(img)
        tifffile.imwrite(
            views_dir / f"{name}_labeled.tif",
            out_rgb,
            compression="zlib",
            photometric="rgb",
            metadata={"axes": "YXC"},
        )
        LOG.info(
            "  wrote views/%s_labelmap.tif + %s_labeled.tif", name, name
        )


def _infer_shape_from_reference(ref_path: Path) -> Tuple[int, int]:
    """Return ``(H, W)`` by reading the reference TIF/PNG header."""
    arr = tifffile.imread(str(ref_path))
    if arr.ndim == 2:
        return int(arr.shape[0]), int(arr.shape[1])
    if arr.ndim == 3:
        # assume ZYX or YXC; in both cases take the last two dims
        return int(arr.shape[-2]), int(arr.shape[-1])
    raise ValueError(f"Cannot infer reference shape from array of shape {arr.shape}")


def build_door_atlas(
    rdata_path: Path,
    output_dir: Path,
    shape_hw: Tuple[int, int] = DEFAULT_SHAPE_HW,
    reference_path: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """Library-callable equivalent of the CLI.

    Args:
        rdata_path: Path to ``door_AL_map.RData``.
        output_dir: Destination directory.
        shape_hw: ``(H, W)`` raster shape per view. Ignored if
            ``reference_path`` is given.
        reference_path: Optional TIF/PNG whose shape is used as the raster
            canvas so the atlas lands directly on the user's reference
            grid — no resize at GUI startup.
        verbose: Verbose logging.
    """
    from door_toolkit.atlas_align.config import configure_logging

    configure_logging(verbose=verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if reference_path is not None:
        shape_hw = _infer_shape_from_reference(Path(reference_path))
        LOG.info("Using reference shape %s for atlas canvas.", shape_hw)

    t0 = time.time()
    atlas = load_door_rdata(rdata_path)
    labelmap, label_lookup, grayscale, view_names = rasterize_multiview(
        atlas, shape_hw
    )

    lm_path = output_dir / "flywire_al_labelmap.tif"
    tifffile.imwrite(
        lm_path,
        labelmap,
        compression="zlib",
        metadata={"axes": "ZYX", "spacing_um": [1.0, 1.0, 1.0]},
    )

    gray_path = output_dir / "flywire_al_grayscale.tif"
    tifffile.imwrite(
        gray_path,
        grayscale,
        compression="zlib",
        metadata={"axes": "ZYX", "spacing_um": [1.0, 1.0, 1.0]},
    )

    labels_path = output_dir / "flywire_al_labels.json"
    labels_path.write_text(
        json.dumps({str(k): v for k, v in label_lookup.items()}, indent=2)
    )

    qc_dir = output_dir / "qc"
    write_qc_mips(labelmap, label_lookup, view_names, qc_dir)

    views_dir = output_dir / "views"
    write_per_view_tifs(
        labelmap, grayscale, label_lookup, view_names, views_dir
    )

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "atlas_type": "door_2d_multiview",
        "source_rdata": str(Path(rdata_path).resolve()),
        "n_glomeruli": len(label_lookup),
        "view_names": view_names,
        "template": "DoOR AL map (5 slice views stacked along Z)",
        "template_shape_zyx": list(labelmap.shape),
        "template_spacing_um_zyx": [1.0, 1.0, 1.0],
        "reference_shape_hw": list(shape_hw) if reference_path is None
            else list(shape_hw),
        "reference_path": str(Path(reference_path).resolve())
            if reference_path is not None else None,
        "labelmap_sha256": hashlib.sha256(lm_path.read_bytes()).hexdigest(),
        "mock": False,
        "local": True,
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    LOG.info(
        "DoOR 2D multiview atlas built in %.1fs. Outputs in %s",
        time.time() - t0, output_dir,
    )
    return {
        "labelmap": lm_path,
        "grayscale": gray_path,
        "labels": labels_path,
        "manifest": manifest_path,
        "qc_dir": qc_dir,
        "views_dir": views_dir,
    }


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a 2D DoOR-polygon multi-view atlas for atlas_align."
    )
    parser.add_argument(
        "--rdata",
        type=Path,
        required=True,
        help=(
            "Path to door_AL_map.RData (download from "
            f"{DEFAULT_RDATA_URL} if you don't have it)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Destination directory for the atlas bundle.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help=(
            "Optional path to the user's reference TIF/PNG. If given, the "
            "atlas canvas is built at the same (H, W) so the GUI loads it "
            "without resizing anything."
        ),
    )
    parser.add_argument(
        "--height",
        type=int,
        default=DEFAULT_SHAPE_HW[0],
        help=(
            f"Per-view raster height (default: {DEFAULT_SHAPE_HW[0]}). "
            "Ignored if --reference is given."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=DEFAULT_SHAPE_HW[1],
        help=(
            f"Per-view raster width (default: {DEFAULT_SHAPE_HW[1]}). "
            "Ignored if --reference is given."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        build_door_atlas(
            rdata_path=args.rdata,
            output_dir=args.output_dir,
            shape_hw=(args.height, args.width),
            reference_path=args.reference,
            verbose=args.verbose,
        )
    except Exception:  # noqa: BLE001
        LOG.exception("DoOR 2D atlas build failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
