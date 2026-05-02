#!/usr/bin/env python
"""
build_atlas.py
==============

One-shot builder: construct a 3D labeled antennal lobe atlas from FlyWire
projection neuron data, registered to the JRC2018F unisex template brain.
Output is a uint16 TIF labelmap suitable for direct consumption by the
atlas_align GUI.

Pipeline:
  1. Authenticate to FlyWire via CAVE token.
  2. Query Schlegel et al. 2024 annotations for all uniglomerular PNs
     (olfactory + thermo/hygrosensory), grouped by target glomerulus.
  3. For each glomerulus, fetch the PN skeletons, restrict to dendrite
     arbor within the AL, pool all node positions into a point cloud.
  4. Compute an alpha-shape mesh per glomerulus.
  5. Transform all meshes from FlyWire (FAFB14) space to JRC2018F via
     the navis-flybrains CMTK bridge.
  6. Rasterize meshes onto the JRC2018F voxel grid (1210x566x174 @
     0.519x0.519x1.000 um), assigning unique integer labels.
  7. Write labelmap TIF + label-name JSON + QC screenshots.

When ``--mock-flywire`` is passed, steps 1–5 are skipped and a
deterministic synthetic 8-glomerulus ellipsoid atlas is rasterized onto
a small 30x60x60 grid instead. This path is the only one exercised by
the automated test suite.

Outputs (in --output-dir):
  - flywire_al_labelmap.tif         uint16, JRC2018F grid, 0 = background
  - flywire_al_labels.json          {"1": "DA1", "2": "DA2", ...}
  - flywire_al_meshes/              individual PLY per glomerulus (optional)
  - qc/                             per-glomerulus MIP PNGs for visual inspection
  - build_manifest.json             versions, param values, checksums

Tested with: Python 3.11, navis 1.8.0, fafbseg 3.0.10, flybrains 0.2.13,
trimesh 4.0.5, tifffile 2024.1.30.

Cole Drayna, Raman Lab, 2026
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tifffile

# Heavy imports deferred to main() / build_labelmap() so --help is fast
# and missing-dep errors are reported with a clear action item.

LOG = logging.getLogger("door_toolkit.atlas_align.atlas_builder")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def setup_logging(verbose: bool = False) -> None:
    """Configure the atlas_align logging pipeline for stand-alone use.

    When the builder is invoked as part of a larger application that has
    already configured logging (e.g. via
    :func:`door_toolkit.atlas_align.config.configure_logging`), this call
    is a no-op apart from the verbosity bump.
    """
    from door_toolkit.atlas_align.config import configure_logging

    configure_logging(verbose=verbose)


# --------------------------------------------------------------------------- #
# Step 1 - authentication
# --------------------------------------------------------------------------- #
def ensure_flywire_auth() -> None:
    """Verify CAVE token is configured; raise with clear message if not."""
    from fafbseg import flywire

    try:
        token = flywire.get_chunkedgraph_secret()
        if not token:
            raise RuntimeError("Empty token")
        LOG.info("FlyWire CAVE token detected.")
    except Exception as e:
        LOG.error(
            "No FlyWire CAVE token found. Run this once interactively:\n"
            "  from fafbseg import flywire\n"
            "  flywire.set_chunkedgraph_secret('YOUR_TOKEN')\n"
            "Get your token at: https://global.daf-apis.com/auth/api/v1/user/token"
        )
        raise SystemExit(1) from e


# --------------------------------------------------------------------------- #
# Step 2 - fetch uPN annotations, grouped by glomerulus
# --------------------------------------------------------------------------- #
@dataclass
class GlomerulusAnnotation:
    name: str
    root_ids: list[int]
    modality: str  # "olfactory" | "thermo" | "hygro"


def fetch_upn_annotations() -> list[GlomerulusAnnotation]:
    """Query flytable for all uniglomerular PNs grouped by target glomerulus."""
    from fafbseg import flywire

    LOG.info("Querying FlyWire flytable for uniglomerular PNs...")
    t0 = time.time()

    # The Schlegel 2024 annotation schema tags uPNs with fields:
    #   cell_class, super_class, cell_type, glomerulus
    # uPN cell_class is one of {"ALPN", "ALLN" (exclude), "ALON"}
    # We want super_class == "neuron" and cell_class == "ALPN" with
    # exactly one glomerulus entry.
    try:
        # Try the newer flytable query path first
        df = flywire.search_annotations(
            "cell_class:ALPN super_class:neuron", dataset="production"
        )
    except AttributeError:
        # Fall back to the older API
        df = flywire.get_annotations(
            table="neuron_information", dataset="production"
        )
        df = df.query("cell_class == 'ALPN' and super_class == 'neuron'")

    LOG.debug("Raw ALPN annotations: %d rows", len(df))

    # Filter to uniglomerular (single-glomerulus label, no ampersands or commas)
    df = df[df["glomerulus"].notna()].copy()
    df = df[~df["glomerulus"].str.contains(r"[,&+/]", na=False)]
    df["glomerulus"] = df["glomerulus"].str.strip().str.upper()

    # Infer modality from glomerulus name
    THERMO = {"VP1", "VP1D", "VP1L", "VP1M", "VP2", "VP3", "VP4", "VP5"}
    HYGRO = {"VP1D", "VP2", "VP3"}  # some overlap is expected
    modality_of = lambda g: (
        "thermo" if g in THERMO else ("hygro" if g in HYGRO else "olfactory")
    )

    grouped: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        grouped.setdefault(row["glomerulus"], []).append(int(row["root_id"]))

    annotations = [
        GlomerulusAnnotation(
            name=glom,
            root_ids=sorted(set(rids)),
            modality=modality_of(glom),
        )
        for glom, rids in sorted(grouped.items())
    ]

    LOG.info(
        "Fetched uPNs for %d glomeruli in %.1fs (total %d neurons).",
        len(annotations),
        time.time() - t0,
        sum(len(a.root_ids) for a in annotations),
    )
    for a in annotations:
        LOG.debug("  %-8s [%s] %d PNs", a.name, a.modality, len(a.root_ids))

    return annotations


# --------------------------------------------------------------------------- #
# Step 3 - fetch skeletons, restrict to AL dendrite arbor
# --------------------------------------------------------------------------- #
def fetch_dendrite_pointcloud(
    root_ids: Sequence[int],
    al_bbox_fafb: np.ndarray,
) -> np.ndarray:
    """Fetch L2 skeletons for uPNs, keep nodes inside the AL bounding box."""
    from fafbseg import flywire

    LOG.debug("  Fetching %d skeletons...", len(root_ids))
    t0 = time.time()

    skeletons = flywire.get_l2_skeletons(
        list(root_ids), omit_failures=True, progress=False
    )

    if not len(skeletons):
        LOG.warning("  No skeletons returned for %s", root_ids[:3])
        return np.empty((0, 3), dtype=np.float32)

    # Merge all node positions
    all_nodes = np.vstack(
        [sk.nodes[["x", "y", "z"]].values for sk in skeletons]
    ).astype(np.float32)

    # Restrict to AL bounding box (nanometers, FAFB14 coords)
    x0, y0, z0 = al_bbox_fafb[0]
    x1, y1, z1 = al_bbox_fafb[1]
    mask = (
        (all_nodes[:, 0] >= x0) & (all_nodes[:, 0] <= x1) &
        (all_nodes[:, 1] >= y0) & (all_nodes[:, 1] <= y1) &
        (all_nodes[:, 2] >= z0) & (all_nodes[:, 2] <= z1)
    )
    filtered = all_nodes[mask]

    LOG.debug(
        "  %d/%d nodes within AL bbox (%.1fs)",
        len(filtered), len(all_nodes), time.time() - t0,
    )
    return filtered


# --------------------------------------------------------------------------- #
# Step 4 - alpha-shape mesh from point cloud
# --------------------------------------------------------------------------- #
def pointcloud_to_mesh(
    points: np.ndarray, alpha: float = 2500.0
) -> "trimesh.Trimesh":
    """Build a watertight mesh around a point cloud via alpha shape.

    alpha is the characteristic length scale in nanometers. 2500 nm (~2.5 um)
    empirically gives tight glomerular boundaries in FAFB; tune if output
    is too loose (lower alpha) or fragmented (higher alpha).
    """
    import trimesh

    if len(points) < 50:
        raise ValueError(f"Too few points ({len(points)}) for mesh reconstruction.")

    # Use trimesh's convex hull as robust fallback; for tighter fit use pyvista's
    # delaunay_3d with alpha, but that introduces a heavier dependency.
    # Alpha shape via scipy Delaunay + simplex filtering:
    from scipy.spatial import Delaunay

    tri = Delaunay(points)
    # Keep simplices whose circumradius < 1/alpha
    tetra = points[tri.simplices]
    # Compute circumradii of tetrahedra
    a = np.linalg.norm(tetra[:, 1] - tetra[:, 0], axis=1)
    b = np.linalg.norm(tetra[:, 2] - tetra[:, 1], axis=1)
    c = np.linalg.norm(tetra[:, 0] - tetra[:, 2], axis=1)
    d = np.linalg.norm(tetra[:, 3] - tetra[:, 0], axis=1)
    e = np.linalg.norm(tetra[:, 3] - tetra[:, 1], axis=1)
    f = np.linalg.norm(tetra[:, 3] - tetra[:, 2], axis=1)
    # Use product-of-edges / 6V as a proxy for circumradius scale
    s = (a + b + c) / 2.0
    area = np.sqrt(np.maximum(s * (s - a) * (s - b) * (s - c), 0))
    vol = np.abs(
        np.einsum(
            "ij,ij->i",
            np.cross(tetra[:, 1] - tetra[:, 0], tetra[:, 2] - tetra[:, 0]),
            tetra[:, 3] - tetra[:, 0],
        )
    ) / 6.0
    # Avoid divide-by-zero
    vol = np.maximum(vol, 1e-9)
    circumradius_proxy = (a * b * c * d * e * f) ** (1 / 6) / np.cbrt(vol)
    keep = circumradius_proxy < alpha * 2

    kept_simplices = tri.simplices[keep]

    # Extract boundary faces of the retained tetrahedra
    faces_all = np.vstack(
        [
            kept_simplices[:, [0, 1, 2]],
            kept_simplices[:, [0, 1, 3]],
            kept_simplices[:, [0, 2, 3]],
            kept_simplices[:, [1, 2, 3]],
        ]
    )
    # Sort vertex indices within each face for dedup
    faces_sorted = np.sort(faces_all, axis=1)
    _, counts = np.unique(faces_sorted, axis=0, return_counts=True)
    unique_faces, first_idx = np.unique(
        faces_sorted, axis=0, return_index=True
    )
    # Boundary faces appear exactly once
    boundary_mask = counts == 1
    boundary_faces = unique_faces[boundary_mask]

    mesh = trimesh.Trimesh(
        vertices=points, faces=boundary_faces, process=True
    )
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    return mesh


# --------------------------------------------------------------------------- #
# Step 5 - transform meshes FlyWire -> JRC2018F
# --------------------------------------------------------------------------- #
def xform_mesh_to_jrc2018f(mesh: "trimesh.Trimesh") -> "trimesh.Trimesh":
    """Transform mesh vertices from FlyWire (FAFB14, nm) to JRC2018F (um).

    Uses ``source="FAFB14"`` rather than ``source="FLYWIRE"`` so navis
    routes through the CMTK FAFB14 → JRC2018F bridge (shipped with
    ``flybrains``) instead of picking a 4-hop path through BANC that
    requires elastix binaries on PATH.
    """
    import navis
    import trimesh

    # FlyWire skeleton coords are FAFB14 nanometres; target is JRC2018F um.
    verts_jrc = navis.xform_brain(
        mesh.vertices, source="FAFB14", target="JRC2018F"
    )
    return trimesh.Trimesh(
        vertices=verts_jrc, faces=mesh.faces, process=False
    )


# --------------------------------------------------------------------------- #
# Local-data path: Schlegel 2024 flat files + SWC skeleton archive
# --------------------------------------------------------------------------- #
# Expected layout under ``--local-flywire <dir>``:
#   classification.csv[.gz]              Schlegel class/super_class/side
#   consolidated_cell_types.csv[.gz]     primary_type (used to parse glomerulus)
#   sk_lod1_783_healed.zip               SWC skeletons named "<root_id>.swc"
#
# All three ship with the Codex v783 release and are ~14 GB for the skeleton
# zip; the CSVs are small.

_LOCAL_CLASSIFICATION_CANDIDATES = (
    "classification.csv",
    "classification.csv.gz",
)
_LOCAL_CELL_TYPES_CANDIDATES = (
    "consolidated_cell_types.csv",
    "consolidated_cell_types.csv.gz",
)
_LOCAL_SKELETON_ZIP_CANDIDATES = (
    "sk_lod1_783_healed.zip",
    "skeletons_783_healed.zip",
)

# Matches the Schlegel primary_type naming convention for uniglomerular PNs:
# ``<glomerulus>_<PN-type>PN``. Explicitly rejects ``M_`` (multiglomerular),
# ``CB`` (unnamed central-brain types), and anything with ``+`` (multi-VP).
_UPN_PRIMARY_TYPE = re.compile(r"^([A-Z][A-Za-z0-9]*)_[a-z0-9]*PN$")


def _resolve_local_file(
    data_dir: Path, candidates: Sequence[str]
) -> Path:
    """Return the first existing path in ``data_dir`` from ``candidates``."""
    for name in candidates:
        p = data_dir / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        f"None of {candidates} found in {data_dir}"
    )


def _parse_glomerulus(primary_type: object) -> Optional[str]:
    """Extract a glomerulus name from a Schlegel PN primary_type string."""
    if not isinstance(primary_type, str):
        return None
    if primary_type.startswith("M_") or primary_type.startswith("CB"):
        return None
    m = _UPN_PRIMARY_TYPE.match(primary_type)
    if not m:
        return None
    glom = m.group(1)
    if glom == "M":
        return None
    return glom


def fetch_upn_annotations_local(
    data_dir: Path,
) -> list[GlomerulusAnnotation]:
    """Parse uPN root_ids grouped by glomerulus from local Schlegel CSVs.

    Args:
        data_dir: Directory containing ``classification.csv`` and
            ``consolidated_cell_types.csv``.

    Returns:
        Same shape as :func:`fetch_upn_annotations`.
    """
    import pandas as pd

    classification_path = _resolve_local_file(
        data_dir, _LOCAL_CLASSIFICATION_CANDIDATES
    )
    cell_types_path = _resolve_local_file(
        data_dir, _LOCAL_CELL_TYPES_CANDIDATES
    )

    LOG.info(
        "Loading local Schlegel annotations: %s + %s",
        classification_path.name, cell_types_path.name,
    )
    t0 = time.time()

    classification = pd.read_csv(
        classification_path, usecols=["root_id", "class"]
    )
    cell_types = pd.read_csv(cell_types_path)
    df = classification.merge(cell_types, on="root_id", how="inner")
    df = df[df["class"] == "ALPN"].copy()
    df["glomerulus"] = df["primary_type"].apply(_parse_glomerulus)
    df = df.dropna(subset=["glomerulus"])

    # Modality: the Schlegel VP* glomeruli are thermo/hygrosensory. Everything
    # else in the uniglomerular set is olfactory.
    THERMO_HYGRO = {"VP1d", "VP1l", "VP1m", "VP2", "VP3", "VP4", "VP5"}

    def modality_of(g: str) -> str:
        return "thermo_hygro" if g in THERMO_HYGRO else "olfactory"

    grouped: dict[str, list[int]] = {}
    for _, row in df.iterrows():
        grouped.setdefault(row["glomerulus"], []).append(int(row["root_id"]))

    annotations = [
        GlomerulusAnnotation(
            name=glom,
            root_ids=sorted(set(rids)),
            modality=modality_of(glom),
        )
        for glom, rids in sorted(grouped.items())
    ]

    LOG.info(
        "Local Schlegel annotations: %d glomeruli / %d uPNs in %.1fs.",
        len(annotations),
        sum(len(a.root_ids) for a in annotations),
        time.time() - t0,
    )
    return annotations


def _swc_nodes_from_zip(
    zip_handle,
    root_id: int,
) -> Optional[np.ndarray]:
    """Return an ``(N, 3)`` XYZ array of skeleton nodes, or None if missing.

    SWC files in Schlegel's healed v783 zip follow the standard NeuroMorpho
    layout: whitespace-separated ``id  type  x  y  z  r  parent`` per node.
    Coordinates are in FAFB14 nanometers.
    """
    try:
        info = zip_handle.getinfo(f"{root_id}.swc")
    except KeyError:
        return None
    with zip_handle.open(info) as fh:
        # Skip comment lines starting with '#'.
        rows = []
        for line in fh:
            line = line.strip()
            if not line or line.startswith(b"#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            rows.append((float(parts[2]), float(parts[3]), float(parts[4])))
    if not rows:
        return None
    return np.asarray(rows, dtype=np.float32)


def fetch_dendrite_pointcloud_local(
    zip_handle,
    root_ids: Sequence[int],
    al_bbox_fafb: np.ndarray,
) -> np.ndarray:
    """Equivalent of :func:`fetch_dendrite_pointcloud` but reads SWC locally.

    Args:
        zip_handle: An open :class:`zipfile.ZipFile` over the healed
            skeleton archive.
        root_ids: Neuron IDs to pool.
        al_bbox_fafb: 2×3 ``[[x0,y0,z0],[x1,y1,z1]]`` bbox in FAFB14 nm.

    Returns:
        ``(N, 3)`` float32 array of node coords inside the bbox.
    """
    t0 = time.time()
    all_nodes: list[np.ndarray] = []
    missing = 0
    for rid in root_ids:
        pts = _swc_nodes_from_zip(zip_handle, rid)
        if pts is None:
            missing += 1
            continue
        all_nodes.append(pts)
    if not all_nodes:
        LOG.warning("  No skeletons found locally for any of %d root_ids",
                    len(root_ids))
        return np.empty((0, 3), dtype=np.float32)

    stacked = np.vstack(all_nodes)
    x0, y0, z0 = al_bbox_fafb[0]
    x1, y1, z1 = al_bbox_fafb[1]
    mask = (
        (stacked[:, 0] >= x0) & (stacked[:, 0] <= x1) &
        (stacked[:, 1] >= y0) & (stacked[:, 1] <= y1) &
        (stacked[:, 2] >= z0) & (stacked[:, 2] <= z1)
    )
    filtered = stacked[mask]
    LOG.debug(
        "  local: %d/%d nodes in AL bbox (missing=%d, %.1fs)",
        len(filtered), len(stacked), missing, time.time() - t0,
    )
    return filtered


# --------------------------------------------------------------------------- #
# Step 6 - rasterize meshes onto JRC2018F voxel grid
# --------------------------------------------------------------------------- #
# JRC2018F canonical dimensions (unisex 20x, high-res)
JRC2018F_SHAPE = (174, 566, 1210)  # Z, Y, X
JRC2018F_SPACING_UM = (1.000, 0.519, 0.519)  # Z, Y, X

# Mock atlas geometry (fast, deterministic, for tests/CI)
MOCK_SHAPE = (30, 60, 60)  # Z, Y, X
MOCK_SPACING_UM = (1.0, 1.0, 1.0)  # Z, Y, X

# 8 deterministic synthetic glomeruli: (name, center_xyz_um, radii_xyz_um).
# Centers are chosen so that every glomerulus has a distinct (x, y) footprint
# under a Z-axis MIP (so all eight are recoverable from the 2D projection
# used by the integration test), and all ellipsoids fit inside the
# MOCK_SHAPE grid with margin.
_MOCK_GLOMERULI = [
    ("DM1",  (12.0, 14.0, 10.0), (3.5, 3.5, 3.0)),
    ("DM2",  (24.0, 14.0, 10.0), (3.5, 3.5, 3.0)),
    ("DA1",  (36.0, 14.0, 10.0), (3.5, 3.5, 3.0)),
    ("DA2",  (48.0, 14.0, 10.0), (3.5, 3.5, 3.0)),
    ("VA1",  (12.0, 40.0, 20.0), (3.5, 3.5, 3.0)),
    ("VA2",  (24.0, 40.0, 20.0), (3.5, 3.5, 3.0)),
    ("VM1",  (36.0, 40.0, 20.0), (3.5, 3.5, 3.0)),
    ("VM7d", (48.0, 40.0, 20.0), (3.5, 3.5, 3.0)),
]


def _build_mock_meshes() -> dict[str, "trimesh.Trimesh"]:
    """Return 8 deterministic ellipsoid meshes in JRC2018F-like coords (um).

    Uses ``trimesh.creation.icosphere(subdivisions=3)`` (162 vertices) as
    the base geometry so rasterization is smooth but cheap.
    """
    import trimesh

    meshes: dict[str, "trimesh.Trimesh"] = {}
    for name, center, radii in _MOCK_GLOMERULI:
        m = trimesh.creation.icosphere(subdivisions=3)
        m.apply_scale(np.asarray(radii, dtype=np.float64))
        m.apply_translation(np.asarray(center, dtype=np.float64))
        meshes[name] = m
        LOG.debug(
            "  Mock %-5s center=%s radii=%s verts=%d",
            name, center, radii, len(m.vertices),
        )
    return meshes


def rasterize_meshes_to_labelmap(
    meshes: dict[str, "trimesh.Trimesh"],
    shape: tuple[int, int, int] = JRC2018F_SHAPE,
    spacing_um: tuple[float, float, float] = JRC2018F_SPACING_UM,
) -> tuple[np.ndarray, dict[int, str]]:
    """Voxelize each mesh and composite into a single integer labelmap."""
    labelmap = np.zeros(shape, dtype=np.uint16)
    label_lookup: dict[int, str] = {}

    for idx, (name, mesh) in enumerate(sorted(meshes.items()), start=1):
        if idx > np.iinfo(np.uint16).max:
            raise ValueError("More glomeruli than uint16 can encode.")

        LOG.debug("  Rasterizing %s -> label %d", name, idx)

        # Voxelize in mesh space (microns), then map to integer indices
        pitch = min(spacing_um)  # isotropic voxelization, then resample
        voxelgrid = mesh.voxelized(pitch=pitch).fill()
        occupied_um = voxelgrid.points  # (N, 3) in microns, XYZ order

        # Convert (x, y, z) in microns to (z_idx, y_idx, x_idx) integers
        ix = np.round(occupied_um[:, 0] / spacing_um[2]).astype(int)
        iy = np.round(occupied_um[:, 1] / spacing_um[1]).astype(int)
        iz = np.round(occupied_um[:, 2] / spacing_um[0]).astype(int)

        in_bounds = (
            (ix >= 0) & (ix < shape[2]) &
            (iy >= 0) & (iy < shape[1]) &
            (iz >= 0) & (iz < shape[0])
        )
        ix, iy, iz = ix[in_bounds], iy[in_bounds], iz[in_bounds]

        # Write-once policy: do not overwrite existing labels (boundary ambiguity)
        free = labelmap[iz, iy, ix] == 0
        labelmap[iz[free], iy[free], ix[free]] = idx
        label_lookup[idx] = name

    LOG.info(
        "Rasterized %d glomeruli into labelmap (%.2f%% voxels filled).",
        len(label_lookup),
        100.0 * np.count_nonzero(labelmap) / labelmap.size,
    )
    return labelmap, label_lookup


# --------------------------------------------------------------------------- #
# Step 7 - QC screenshots
# --------------------------------------------------------------------------- #
def write_qc_mips(
    labelmap: np.ndarray, label_lookup: dict[int, str], qc_dir: Path
) -> None:
    """Write per-axis MIPs with labels for visual inspection."""
    import matplotlib
    matplotlib.use("Agg")  # headless-safe
    import matplotlib.pyplot as plt

    qc_dir.mkdir(parents=True, exist_ok=True)
    for axis, axname in zip((0, 1, 2), ("z", "y", "x")):
        mip = labelmap.max(axis=axis)
        fig, ax = plt.subplots(figsize=(12, 6), dpi=120)
        ax.imshow(mip, cmap="tab20", interpolation="nearest")
        ax.set_title(f"Labelmap MIP along {axname}-axis ({len(label_lookup)} glomeruli)")
        ax.axis("off")
        fig.savefig(qc_dir / f"mip_{axname}.png", bbox_inches="tight")
        plt.close(fig)
    LOG.info("Wrote QC MIPs to %s", qc_dir)


# --------------------------------------------------------------------------- #
# Library-callable entry point
# --------------------------------------------------------------------------- #
def build_labelmap(
    output_dir: Path,
    alpha: float = 2500.0,
    save_meshes: bool = False,
    mock_flywire: bool = False,
    local_flywire: Optional[Path] = None,
    verbose: bool = False,
) -> dict:
    """Build a glomerular labelmap and write outputs.

    Three data-source modes are supported:

    * ``mock_flywire=True`` — deterministic 8-glomerulus synthetic atlas,
      used by the test suite. No network, no local files needed.
    * ``local_flywire=<dir>`` — read Schlegel 2024 annotations and v783
      healed SWC skeletons from a local directory. Bypasses CAVE entirely.
      The directory must contain ``classification.csv``,
      ``consolidated_cell_types.csv``, and ``sk_lod1_783_healed.zip``.
    * neither — query FlyWire via CAVE (original behaviour). Requires a
      token with ``view`` permission on the ``fafb`` production dataset.

    Args:
        output_dir: Destination directory; created if it doesn't exist.
        alpha: Alpha-shape length scale (nm). Ignored in mock mode.
        save_meshes: If True, also write per-glomerulus PLY meshes.
        mock_flywire: If True, use the deterministic synthetic atlas.
        local_flywire: If given, read annotations + skeletons from this
            local directory instead of calling CAVE.
        verbose: If True, route DEBUG-level logs to console.

    Returns:
        Dict with keys ``labelmap``, ``labels``, ``manifest``, ``qc_dir``
        mapping to :class:`pathlib.Path`. When ``save_meshes`` is True a
        ``meshes_dir`` key is also included.
    """
    setup_logging(verbose=verbose)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if mock_flywire and local_flywire is not None:
        raise ValueError(
            "--mock-flywire and --local-flywire are mutually exclusive."
        )

    # Defer heavy imports
    import trimesh

    if mock_flywire:
        LOG.info("Building MOCK atlas (8 synthetic glomeruli, %s grid).", MOCK_SHAPE)
        meshes_jrc = _build_mock_meshes()
        failed: list[str] = []
        raster_shape = MOCK_SHAPE
        raster_spacing = MOCK_SPACING_UM
        navis_version = "mock"
    elif local_flywire is not None:
        import navis
        import flybrains  # noqa: F401  registers transforms into navis

        local_dir = Path(local_flywire)
        if not local_dir.is_dir():
            raise FileNotFoundError(
                f"--local-flywire directory does not exist: {local_dir}"
            )

        al_bbox_fafb = np.array(
            [[380_000, 130_000, 50_000], [620_000, 280_000, 150_000]],
            dtype=np.float64,
        )

        annotations = fetch_upn_annotations_local(local_dir)

        skeleton_zip = _resolve_local_file(
            local_dir, _LOCAL_SKELETON_ZIP_CANDIDATES
        )
        LOG.info("Opening local skeleton archive: %s", skeleton_zip.name)

        meshes_fw: dict[str, "trimesh.Trimesh"] = {}
        failed = []
        with zipfile.ZipFile(skeleton_zip) as zf:
            for ann in annotations:
                LOG.info(
                    "  %s (%s, %d PNs)",
                    ann.name, ann.modality, len(ann.root_ids),
                )
                try:
                    pts = fetch_dendrite_pointcloud_local(
                        zf, ann.root_ids, al_bbox_fafb
                    )
                    if len(pts) < 50:
                        LOG.warning("    Skipping: insufficient dendrite nodes")
                        failed.append(ann.name)
                        continue
                    mesh = pointcloud_to_mesh(pts, alpha=alpha)
                    meshes_fw[ann.name] = mesh
                    LOG.debug(
                        "    Mesh: %d verts, %d faces, volume=%.1e nm^3",
                        len(mesh.vertices), len(mesh.faces), mesh.volume,
                    )
                except Exception as e:  # noqa: BLE001
                    LOG.error("    Failed: %s", e)
                    failed.append(ann.name)

        if failed:
            LOG.warning(
                "Failed to build meshes for %d glomeruli: %s",
                len(failed), ", ".join(failed),
            )

        LOG.info("Transforming %d meshes FLYWIRE -> JRC2018F...", len(meshes_fw))
        meshes_jrc = {
            name: xform_mesh_to_jrc2018f(m) for name, m in meshes_fw.items()
        }
        raster_shape = JRC2018F_SHAPE
        raster_spacing = JRC2018F_SPACING_UM
        navis_version = navis.__version__
    else:
        import navis
        import flybrains  # noqa: F401  registers transforms into navis

        ensure_flywire_auth()

        # AL bounding box in FAFB14 nanometer coords (generous margin;
        # both hemispheres).
        al_bbox_fafb = np.array(
            [[380_000, 130_000, 50_000], [620_000, 280_000, 150_000]],
            dtype=np.float64,
        )

        annotations = fetch_upn_annotations()

        LOG.info("Building glomerular meshes...")
        meshes_fw: dict[str, "trimesh.Trimesh"] = {}
        failed = []
        for ann in annotations:
            LOG.info(
                "  %s (%s, %d PNs)", ann.name, ann.modality, len(ann.root_ids)
            )
            try:
                pts = fetch_dendrite_pointcloud(ann.root_ids, al_bbox_fafb)
                if len(pts) < 50:
                    LOG.warning("    Skipping: insufficient dendrite nodes")
                    failed.append(ann.name)
                    continue
                mesh = pointcloud_to_mesh(pts, alpha=alpha)
                meshes_fw[ann.name] = mesh
                LOG.debug(
                    "    Mesh: %d verts, %d faces, volume=%.1e nm^3",
                    len(mesh.vertices), len(mesh.faces), mesh.volume,
                )
            except Exception as e:
                LOG.error("    Failed: %s", e)
                failed.append(ann.name)

        if failed:
            LOG.warning(
                "Failed to build meshes for %d glomeruli: %s",
                len(failed), ", ".join(failed),
            )

        LOG.info("Transforming %d meshes FLYWIRE -> JRC2018F...", len(meshes_fw))
        meshes_jrc = {
            name: xform_mesh_to_jrc2018f(m) for name, m in meshes_fw.items()
        }
        raster_shape = JRC2018F_SHAPE
        raster_spacing = JRC2018F_SPACING_UM
        navis_version = navis.__version__

    if save_meshes:
        mesh_dir = output_dir / "flywire_al_meshes"
        mesh_dir.mkdir(exist_ok=True)
        for name, m in meshes_jrc.items():
            m.export(mesh_dir / f"{name}.ply")
        LOG.info("Saved individual meshes to %s", mesh_dir)

    LOG.info(
        "Rasterizing to grid %s @ %s um...", raster_shape, raster_spacing
    )
    labelmap, label_lookup = rasterize_meshes_to_labelmap(
        meshes_jrc, shape=raster_shape, spacing_um=raster_spacing
    )

    # Write outputs
    lm_path = output_dir / "flywire_al_labelmap.tif"
    tifffile.imwrite(
        lm_path,
        labelmap,
        compression="zlib",
        metadata={"axes": "ZYX", "spacing_um": list(raster_spacing)},
    )
    LOG.info(
        "Wrote labelmap: %s (%.1f MB)", lm_path, lm_path.stat().st_size / 1e6
    )

    labels_path = output_dir / "flywire_al_labels.json"
    labels_path.write_text(
        json.dumps({str(k): v for k, v in label_lookup.items()}, indent=2)
    )
    LOG.info("Wrote label lookup: %s", labels_path)

    qc_dir = output_dir / "qc"
    write_qc_mips(labelmap, label_lookup, qc_dir)

    # Build manifest
    if mock_flywire:
        source = "synthetic ellipsoids (mock-flywire mode)"
    elif local_flywire is not None:
        source = (
            f"local Schlegel 2024 flat files + healed v783 SWC archive "
            f"at {Path(local_flywire).resolve()}"
        )
    else:
        source = "FlyWire production dataset (Schlegel 2024 annotations)"

    manifest = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mock": bool(mock_flywire),
        "local": local_flywire is not None,
        "alpha_nm": alpha,
        "n_glomeruli": len(label_lookup),
        "failed_glomeruli": failed,
        "labelmap_sha256": hashlib.sha256(lm_path.read_bytes()).hexdigest(),
        "template": "MOCK" if mock_flywire else "JRC2018F",
        "template_shape_zyx": list(raster_shape),
        "template_spacing_um_zyx": list(raster_spacing),
        "source": source,
        "package_versions": {
            "navis": navis_version,
            "trimesh": trimesh.__version__,
        },
    }
    manifest_path = output_dir / "build_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    LOG.info("Done. Outputs in %s", output_dir.resolve())
    LOG.info(
        "Next: point the atlas_align GUI at this directory via "
        "`--atlas %s`.",
        output_dir,
    )

    result: dict = {
        "labelmap": lm_path,
        "labels": labels_path,
        "manifest": manifest_path,
        "qc_dir": qc_dir,
    }
    if save_meshes:
        result["meshes_dir"] = output_dir / "flywire_al_meshes"
    return result


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FlyWire-derived AL labelmap in JRC2018F space."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./flywire_al_atlas"),
        help="Output directory (default: ./flywire_al_atlas)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=2500.0,
        help="Alpha-shape length scale in nm (default: 2500)",
    )
    parser.add_argument(
        "--save-meshes",
        action="store_true",
        help="Also save individual PLY files per glomerulus",
    )
    parser.add_argument(
        "--mock-flywire",
        action="store_true",
        help=(
            "Skip FlyWire/navis queries and build a deterministic synthetic "
            "8-glomerulus toy atlas on a 30x60x60 grid. Intended for tests "
            "and smoke checks only."
        ),
    )
    parser.add_argument(
        "--local-flywire",
        type=Path,
        default=None,
        help=(
            "Path to a directory holding a local Schlegel 2024 dump: "
            "classification.csv, consolidated_cell_types.csv, and "
            "sk_lod1_783_healed.zip. Using this flag bypasses the CAVE "
            "API entirely and avoids the 403 if your token lacks FAFB "
            "production access."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for ``door-atlas-builder``.

    Returns:
        0 on success; non-zero on unrecoverable failure.
    """
    args = _parse_args(argv)
    try:
        build_labelmap(
            output_dir=args.output_dir,
            alpha=args.alpha,
            save_meshes=args.save_meshes,
            mock_flywire=args.mock_flywire,
            local_flywire=args.local_flywire,
            verbose=args.verbose,
        )
    except SystemExit:
        raise
    except Exception:
        LOG.exception("Atlas build failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
