#!/usr/bin/env python
"""
Fit DoOR atlas slices to manually-labelled ROIs.
=================================================

Given:

* a DoOR 2D multi-view atlas built with ``door-atlas-builder-2d``,
* the user's FIJI RoiManager zip,
* the reference image the ROIs were drawn on,
* a CSV of manual glomerulus assignments
  (``manually_assigned_glomerulus`` or legacy ``assigned_glomerulus``
  column),

this script iterates over the 4 anatomical slice views (slice1..slice4)
plus the "extras" view — **skipping VP** — and for each view fits a 4-DOF
similarity transform (translation + rotation + uniform scale + optional
X/Y flip) that maps the atlas's glomerulus polygon centroids onto the
user's labelled ROI centroids. If fewer than 2 matching anchors exist
for that view, the view is skipped.

One PNG per view is written with the user's reference underneath, the
**fitted atlas polygons** drawn below the user's ROI outlines (so ROIs
stay legible), and glomerulus names labelled at each fitted polygon
centroid. A ``fit_report.txt`` summarises anchors, residuals, and
candidate labels for currently-unlabelled ROIs.

Intended invocation::

    python scripts/fit_atlas_to_labels.py \\
        --labels data/assignments.csv \\
        --rois /home/ramanlab/Lightsheet/fly_4/ROIACV.zip \\
        --reference /home/ramanlab/Lightsheet/fly_4/trial_001_OFM_A/STD_project_odor.tif \\
        --atlas ~/door_al_atlas \\
        --output-dir out/atlas_fit
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.patches import Polygon as MplPolygon
from scipy import ndimage
from skimage import measure


# Views to skip. VP is thermo/hygro and not useful for olfactory fits.
SKIP_VIEWS = {"VP"}


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class ViewGlomeruli:
    """Per-view glomerulus geometry extracted from the atlas labelmap."""

    view_name: str
    view_z: int
    polygons: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (M, 2) (x, y)
    centroids: Dict[str, np.ndarray] = field(default_factory=dict)  # name -> (x, y)


@dataclass
class FitResult:
    matrix: np.ndarray  # 3×3 homogeneous
    residual_px: float
    rotation_deg: float
    scale: float
    flip_x: bool
    flip_y: bool
    n_anchors: int


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def load_labels_csv(path: Path) -> Dict[int, str]:
    """Return ``{roi_index → glomerulus_name}`` for manually-labelled rows.

    Accepts either the simplified manual CSV (column
    ``manually_assigned_glomerulus``) or the legacy full export
    (``assigned_glomerulus``). Rows with blank / NaN glomerulus are
    skipped.
    """
    df = pd.read_csv(path)
    col = None
    for candidate in (
        "manually_assigned_glomerulus",
        "manual_glomerulus",
        "assigned_glomerulus",
        "glomerulus",
    ):
        if candidate in df.columns:
            col = candidate
            break
    if col is None:
        raise ValueError(
            f"No recognised glomerulus column in {path}; "
            f"expected one of "
            "'manually_assigned_glomerulus', 'assigned_glomerulus', "
            "'glomerulus'. Got: " + ", ".join(df.columns)
        )
    if "roi_index" not in df.columns:
        raise ValueError(f"{path} must have a 'roi_index' column.")
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        name = str(row[col]).strip()
        if not name or name.lower() in {"nan", ""}:
            continue
        out[int(row["roi_index"])] = name
    return out


def extract_view_geometry(
    labelmap: np.ndarray,
    labels: Dict[int, str],
    view_names: List[str],
) -> List[ViewGlomeruli]:
    """Per-view polygon + centroid extraction from the 6-frame labelmap."""
    result: List[ViewGlomeruli] = []
    if labelmap.ndim != 3:
        raise ValueError(f"labelmap must be (Z, H, W); got {labelmap.shape}")

    for z, name in enumerate(view_names):
        view = ViewGlomeruli(view_name=name, view_z=z)
        slice_lm = labelmap[z]
        for label_id, glom_name in labels.items():
            gid = int(label_id)
            if gid == 0:
                continue
            mask = slice_lm == gid
            if not mask.any():
                continue
            # Largest contour in (row, col) = (y, x) order.
            contours = measure.find_contours(mask.astype(np.float32), 0.5)
            if not contours:
                continue
            contour = max(contours, key=len)
            poly_xy = np.column_stack(
                [contour[:, 1], contour[:, 0]]
            ).astype(np.float32)  # → (x, y)
            view.polygons[glom_name] = poly_xy
            # centroid via mean position
            ys, xs = np.where(mask)
            view.centroids[glom_name] = np.array(
                [float(xs.mean()), float(ys.mean())], dtype=np.float32
            )
        result.append(view)
    return result


def _centroid_from_roi(roi) -> Tuple[float, float]:
    try:
        return roi.centroid  # (x, y)
    except Exception:  # noqa: BLE001
        x0, y0, x1, y1 = roi.bbox
        return ((x0 + x1) / 2.0, (y0 + y1) / 2.0)


# --------------------------------------------------------------------------- #
# Transform fitting (Umeyama similarity + flip search)
# --------------------------------------------------------------------------- #
def fit_similarity_with_flips(
    src_xy: np.ndarray, dst_xy: np.ndarray
) -> FitResult:
    """Find the (tx, ty, rotation, uniform-scale, ±flip_x, ±flip_y) transform
    that best maps ``src_xy`` onto ``dst_xy``. Uses Umeyama (1991) for the
    similarity part and brute-forces the 4 flip combinations.
    """
    if len(src_xy) < 2 or len(dst_xy) != len(src_xy):
        raise ValueError("need ≥ 2 matched point pairs")
    best: Optional[FitResult] = None
    for fx in (1.0, -1.0):
        for fy in (1.0, -1.0):
            flipped = src_xy * np.array([fx, fy], dtype=np.float64)
            mp = flipped.mean(axis=0)
            mq = dst_xy.mean(axis=0)
            pc = flipped - mp
            qc = dst_xy - mq
            H = (pc.T @ qc) / len(src_xy)
            U, S, Vt = np.linalg.svd(H)
            D = np.eye(2)
            if np.linalg.det(U @ Vt) < 0:
                D[1, 1] = -1.0
            R = Vt.T @ D @ U.T
            var_src = (pc * pc).sum() / len(pc)
            s = float(np.sum(S * np.diag(D)) / var_src) if var_src > 0 else 1.0
            t = mq - s * R @ mp
            predicted = (s * R @ flipped.T).T + t
            residual = float(np.linalg.norm(predicted - dst_xy, axis=1).mean())

            matrix = np.eye(3)
            matrix[:2, :2] = (s * R) * np.array([fx, fy])
            matrix[:2, 2] = t
            rotation_deg = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))

            candidate = FitResult(
                matrix=matrix,
                residual_px=residual,
                rotation_deg=rotation_deg,
                scale=s,
                flip_x=(fx == -1.0),
                flip_y=(fy == -1.0),
                n_anchors=len(src_xy),
            )
            if best is None or candidate.residual_px < best.residual_px:
                best = candidate
    assert best is not None
    return best


def apply_transform(points_xy: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    n = len(points_xy)
    h = np.hstack([points_xy, np.ones((n, 1), dtype=np.float64)])
    out = (matrix @ h.T).T
    return out[:, :2].astype(np.float32)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _color_for_name(name: str) -> Tuple[float, float, float]:
    import hashlib, colorsys

    d = hashlib.md5(name.encode()).digest()
    h = d[0] / 255.0
    s = 0.65 + (d[1] % 20) / 100.0
    v = 0.85 + (d[2] % 15) / 100.0
    return colorsys.hsv_to_rgb(h, s, v)


def render_view_fit(
    reference: np.ndarray,
    view: ViewGlomeruli,
    transform: Optional[FitResult],
    rois: List,
    manual_labels: Dict[int, str],
    roi_to_glom: Dict[int, str],
    output_path: Path,
) -> None:
    H, W = reference.shape[:2]
    fig, ax = plt.subplots(figsize=(14, 14 * H / W), dpi=150)
    ax.imshow(reference, cmap="gray", origin="upper")

    # Fitted atlas polygons UNDER the ROI outlines.
    if transform is not None:
        for name, poly in view.polygons.items():
            fitted = apply_transform(poly, transform.matrix)
            color = _color_for_name(name)
            patch = MplPolygon(
                fitted, closed=True,
                facecolor=(*color, 0.35),
                edgecolor=(*color, 0.95),
                linewidth=1.5,
                zorder=5,
            )
            ax.add_patch(patch)
            # label at fitted centroid
            cx, cy = apply_transform(
                view.centroids[name][None, :], transform.matrix
            )[0]
            ax.text(
                cx, cy, name,
                fontsize=9, fontweight="bold",
                ha="center", va="center", color="white",
                zorder=12,
                path_effects=[
                    _text_outline(1.5),
                ],
            )

    # ROI outlines on TOP.
    for i, roi in enumerate(rois):
        # Close polygon for drawing.
        x = np.concatenate([roi.x, roi.x[:1]])
        y = np.concatenate([roi.y, roi.y[:1]])
        label = manual_labels.get(i)
        is_anchor = label in roi_to_glom.values()
        color = "yellow" if is_anchor else "cyan"
        lw = 2.2 if is_anchor else 1.4
        ax.plot(x, y, color=color, lw=lw, zorder=20)
        if label:
            cx, cy = _centroid_from_roi(roi)
            ax.text(
                cx, cy, f"[{label}]",
                fontsize=8, color="yellow", ha="center", va="center",
                zorder=25,
                path_effects=[_text_outline(1.5)],
            )

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_aspect("equal")
    title = f"View '{view.view_name}'"
    if transform is None:
        title += "   —   skipped (fewer than 2 matching anchors)"
    else:
        title += (
            f"   —   fit: {transform.n_anchors} anchors, "
            f"residual {transform.residual_px:.1f} px, "
            f"scale {transform.scale:.2f}, "
            f"rot {transform.rotation_deg:+.1f}°"
        )
        if transform.flip_x:
            title += " + flipX"
        if transform.flip_y:
            title += " + flipY"
    ax.set_title(title, fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def _text_outline(linewidth: float):
    from matplotlib import patheffects

    return patheffects.withStroke(linewidth=linewidth, foreground="black")


# --------------------------------------------------------------------------- #
# Per-view fit driver
# --------------------------------------------------------------------------- #
def fit_and_report_view(
    view: ViewGlomeruli,
    roi_to_glom: Dict[int, str],
    rois_list: List,
) -> Tuple[Optional[FitResult], List[str]]:
    """Fit this view against the manual labels. Returns (fit, report_lines)."""
    lines: List[str] = []
    # Match: manual labels that land on a glomerulus polygon in THIS view.
    anchors_src: List[np.ndarray] = []
    anchors_dst: List[np.ndarray] = []
    anchor_names: List[str] = []
    for roi_idx, glom in roi_to_glom.items():
        if glom not in view.centroids:
            continue
        atlas_c = view.centroids[glom]
        roi = rois_list[roi_idx]
        rx, ry = _centroid_from_roi(roi)
        anchors_src.append(atlas_c)
        anchors_dst.append(np.array([rx, ry], dtype=np.float32))
        anchor_names.append(glom)

    lines.append(f"View '{view.view_name}':")
    lines.append(f"  Glomeruli in this view: {sorted(view.centroids.keys())}")
    lines.append(f"  Matching anchors: {len(anchors_src)}")
    if len(anchors_src) < 2:
        lines.append("  -> SKIPPED (need ≥ 2 anchors).")
        return None, lines

    src = np.vstack(anchors_src).astype(np.float64)
    dst = np.vstack(anchors_dst).astype(np.float64)
    fit = fit_similarity_with_flips(src, dst)
    lines.append(
        f"  Fit: residual={fit.residual_px:.1f}px, "
        f"scale={fit.scale:.3f}, rot={fit.rotation_deg:+.2f}°, "
        f"flipX={fit.flip_x}, flipY={fit.flip_y}"
    )
    lines.append(f"  Anchors used: {anchor_names}")
    # Suggest candidate labels for OTHER glomeruli in this view: after the
    # fit, the predicted centroid for each unlabelled glomerulus points to a
    # place on the image — find the nearest *unlabelled* ROI.
    labelled_roi_idxs = set(roi_to_glom.keys())
    unlabelled_rois = [
        (i, _centroid_from_roi(r))
        for i, r in enumerate(rois_list)
        if i not in labelled_roi_idxs
    ]
    if unlabelled_rois:
        lines.append("  Candidate labels (unlabelled ROI nearest to each predicted glomerulus):")
        for glom, centroid in view.centroids.items():
            if glom in anchor_names:
                continue
            predicted = apply_transform(centroid[None, :], fit.matrix)[0]
            # nearest ROI
            best_i, best_d = -1, float("inf")
            for ri, (rx, ry) in unlabelled_rois:
                d = float(np.hypot(rx - predicted[0], ry - predicted[1]))
                if d < best_d:
                    best_d = d
                    best_i = ri
            if best_i >= 0:
                lines.append(
                    f"    {glom:8s} → ROI #{best_i} "
                    f"(distance {best_d:.1f} px)"
                )
    return fit, lines


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="CSV with manual labels (roi_index + glomerulus column).")
    parser.add_argument("--rois", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--atlas", type=Path, required=True,
                        help="door-atlas-builder-2d output directory.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "out" / "atlas_fit",
        help="Where to write PNGs + fit_report.txt.",
    )
    args = parser.parse_args(argv)

    # Deferred imports for speed / optional deps.
    from door_toolkit.atlas_align.io.roi_loader import load_rois
    from door_toolkit.atlas_align.io.atlas_loader import load_atlas_bundle

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels_csv(args.labels)
    print(f"Loaded {len(labels)} manual labels from {args.labels}")

    roi_set = load_rois(args.rois)
    rois_list = list(roi_set)
    print(f"Loaded {len(rois_list)} FIJI ROIs from {args.rois}")

    bundle = load_atlas_bundle(args.atlas)
    view_names = list(bundle.manifest.get("view_names") or [])
    if len(view_names) != bundle.labelmap.shape[0]:
        view_names = [f"view_{i}" for i in range(bundle.labelmap.shape[0])]
    print(f"Atlas views: {view_names}")

    reference = tifffile.imread(str(args.reference))
    if reference.ndim == 3:
        reference = reference.max(axis=0) if reference.shape[0] <= 8 else reference[..., :3].mean(axis=-1)
    reference = reference.astype(np.float32)

    views = extract_view_geometry(bundle.labelmap, bundle.labels, view_names)

    report_lines: List[str] = []
    report_lines.append(f"fit_atlas_to_labels report")
    report_lines.append(f"==========================")
    report_lines.append(f"labels:    {args.labels}")
    report_lines.append(f"rois:      {args.rois}")
    report_lines.append(f"reference: {args.reference}")
    report_lines.append(f"atlas:     {args.atlas}")
    report_lines.append(f"views:     {view_names} (skipping {sorted(SKIP_VIEWS)})")
    report_lines.append("")

    for view in views:
        if view.view_name in SKIP_VIEWS:
            report_lines.append(f"View '{view.view_name}': SKIPPED (requested).\n")
            continue
        fit, view_lines = fit_and_report_view(view, labels, rois_list)
        report_lines.extend(view_lines)
        png_path = output_dir / f"{view.view_name}_fitted.png"
        render_view_fit(
            reference=reference,
            view=view,
            transform=fit,
            rois=rois_list,
            manual_labels=labels,
            roi_to_glom=labels,
            output_path=png_path,
        )
        report_lines.append(f"  -> wrote {png_path}")
        report_lines.append("")

    report_path = output_dir / "fit_report.txt"
    report_path.write_text("\n".join(report_lines))
    print("\n".join(report_lines))
    print(f"\nReport: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
