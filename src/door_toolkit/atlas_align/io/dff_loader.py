"""
ΔF/F signal loader
==================

Reads a directory of per-odor ΔF/F CSV files produced by the imaging
analysis pipeline (one CSV per odor trial, e.g. ``dff_A_trial_001_OFM_A.csv``)
and computes a per-ROI max / min ΔF/F table across odors.

Expected CSV format:

* First column: ``frame`` (time index; ignored beyond sanity checking).
* Remaining columns: one per ROI, with column name encoding the ROI
  centroid like ``026_a2245_y841_x617`` (index / area / y / x). We
  parse the ``y`` and ``x`` fields to match CSV ROIs to FIJI RoiManager
  ROIs by nearest centroid.

Odor letter encoded in the filename (``dff_<L>_trial_...``) is mapped
to the canonical DoOR odorant name via :data:`DEFAULT_ODOR_LETTER_MAP`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from door_toolkit.atlas_align.config import get_logger

logger = get_logger(__name__)


#: Maps the letter in a filename like ``dff_A_trial_001_OFM_A.csv`` to the
#: canonical DoOR odorant name. Covers the 7 odors used in the standard
#: OFM panel; extend as needed.
DEFAULT_ODOR_LETTER_MAP: Dict[str, str] = {
    "A": "Apple_Cider_Vinegar",
    "B": "Benzaldehyde",
    "C": "Citral",
    "E": "Ethyl_Butyrate",
    "H": "Hexanol",
    "L": "Linalool",
    "O": "3-Octanol",
}

_FILENAME_RE = re.compile(r"dff_([A-Za-z])_trial_.*\.csv$")
_ROI_HEADER_RE = re.compile(
    r"^(?P<idx>\d+)_a(?P<area>\d+)_y(?P<y>\d+)_x(?P<x>\d+)$"
)


@dataclass
class DFFROIColumn:
    """One ROI column from a dff CSV, parsed into usable fields."""

    header: str
    index: int         # 1-based ROI number from the header
    area: int          # pixel area (unused but retained)
    y: float           # centroid y in image pixel coords
    x: float           # centroid x in image pixel coords


@dataclass
class DFFBundle:
    """Loaded ΔF/F data across all odors.

    Attributes:
        odor_order: Odor names (canonical) in the order they were loaded.
        roi_columns: ROI metadata parsed from CSV column names — one list,
            assumed identical across all odors (verified on load).
        max_matrix: ``(n_odors, n_rois)`` peak ΔF/F per (odor, ROI).
        min_matrix: ``(n_odors, n_rois)`` trough ΔF/F per (odor, ROI).
        traces: Per-odor raw traces as DataFrames with ROI columns +
            ``frame`` index. Useful if the user wants sparklines later.
        source_dir: The directory we loaded from.
    """

    odor_order: List[str]
    roi_columns: List[DFFROIColumn]
    max_matrix: np.ndarray
    min_matrix: np.ndarray
    traces: Dict[str, pd.DataFrame] = field(default_factory=dict)
    source_dir: Optional[Path] = None

    @property
    def n_odors(self) -> int:
        return len(self.odor_order)

    @property
    def n_rois(self) -> int:
        return len(self.roi_columns)

    def roi_centroids(self) -> np.ndarray:
        """``(n_rois, 2)`` float array of ``(x, y)`` centroids."""
        return np.array(
            [(c.x, c.y) for c in self.roi_columns], dtype=np.float32
        )

    def summary_for_roi(self, roi_index: int) -> Dict[str, Tuple[float, float]]:
        """Return ``{odor: (min, max)}`` for the given ROI index."""
        if not (0 <= roi_index < self.n_rois):
            raise IndexError(
                f"roi_index {roi_index} out of range 0..{self.n_rois - 1}"
            )
        return {
            odor: (
                float(self.min_matrix[i, roi_index]),
                float(self.max_matrix[i, roi_index]),
            )
            for i, odor in enumerate(self.odor_order)
        }

    def max_vector_for_roi(self, roi_index: int) -> np.ndarray:
        """``(n_odors,)`` peak-ΔF/F vector for one ROI."""
        return self.max_matrix[:, roi_index].copy()


def _parse_roi_header(header: str) -> Optional[DFFROIColumn]:
    m = _ROI_HEADER_RE.match(header)
    if m is None:
        return None
    return DFFROIColumn(
        header=header,
        index=int(m.group("idx")),
        area=int(m.group("area")),
        y=float(m.group("y")),
        x=float(m.group("x")),
    )


def load_dff_directory(
    path: Path,
    odor_letter_map: Optional[Dict[str, str]] = None,
) -> DFFBundle:
    """Load every ``dff_<letter>_*.csv`` in ``path`` into a :class:`DFFBundle`.

    Args:
        path: Directory containing the CSVs.
        odor_letter_map: Override for the letter → canonical-odor mapping.
            Defaults to :data:`DEFAULT_ODOR_LETTER_MAP`.

    Returns:
        :class:`DFFBundle` with per-ROI max/min tables.

    Raises:
        FileNotFoundError: if ``path`` doesn't exist or has no matching CSVs.
        ValueError: if CSVs have mismatched ROI column sets.
    """
    path = Path(path).expanduser().resolve()
    logger.debug("load_dff_directory(path=%s)", path)
    if not path.is_dir():
        raise FileNotFoundError(f"dff directory does not exist: {path}")

    odor_letter_map = dict(odor_letter_map or DEFAULT_ODOR_LETTER_MAP)

    csv_files = sorted(path.glob("dff_*_trial_*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No files matching 'dff_<letter>_trial_*.csv' in {path}"
        )

    traces: Dict[str, pd.DataFrame] = {}
    odor_order: List[str] = []
    reference_columns: Optional[List[str]] = None

    for csv_path in csv_files:
        m = _FILENAME_RE.search(csv_path.name)
        if m is None:
            logger.warning(
                "Skipping %s: name doesn't match dff_<letter>_trial_*", csv_path
            )
            continue
        letter = m.group(1).upper()
        if letter not in odor_letter_map:
            logger.warning(
                "Unknown odor letter %r in %s — mapping to 'odor_%s'",
                letter, csv_path.name, letter,
            )
            odor_name = f"odor_{letter}"
        else:
            odor_name = odor_letter_map[letter]

        df = pd.read_csv(csv_path)
        if "frame" not in df.columns:
            logger.warning(
                "%s has no 'frame' column — assuming first col is time",
                csv_path.name,
            )
        roi_cols = [c for c in df.columns if c != "frame"]
        if reference_columns is None:
            reference_columns = list(roi_cols)
        elif list(roi_cols) != reference_columns:
            # Columns don't exactly match across odors; take the intersection
            # in reference order to stay aligned. Common when a trial had a
            # missing ROI.
            shared = [c for c in reference_columns if c in roi_cols]
            logger.warning(
                "%s has %d ROI columns vs reference %d; keeping %d shared.",
                csv_path.name, len(roi_cols), len(reference_columns),
                len(shared),
            )
            reference_columns = shared

        traces[odor_name] = df
        odor_order.append(odor_name)
        logger.info(
            "Loaded dff %s (%s): %d frames × %d ROIs",
            odor_name, csv_path.name,
            df.shape[0], len(roi_cols),
        )

    if reference_columns is None or not reference_columns:
        raise ValueError(f"No usable ROI columns found in any CSV under {path}")

    # Parse ROI column headers.
    roi_columns: List[DFFROIColumn] = []
    for h in reference_columns:
        parsed = _parse_roi_header(h)
        if parsed is None:
            logger.warning(
                "ROI column %r doesn't match expected pattern — dropping",
                h,
            )
            continue
        roi_columns.append(parsed)

    if not roi_columns:
        raise ValueError(
            f"No ROI columns matched the expected "
            f"'<idx>_a<area>_y<y>_x<x>' pattern in {path}"
        )

    # Build the max / min matrices in (odor, roi) shape.
    n_odors = len(odor_order)
    n_rois = len(roi_columns)
    max_m = np.zeros((n_odors, n_rois), dtype=np.float32)
    min_m = np.zeros((n_odors, n_rois), dtype=np.float32)
    for oi, odor in enumerate(odor_order):
        df = traces[odor]
        for ri, roi in enumerate(roi_columns):
            if roi.header not in df.columns:
                # This odor lost that ROI; leave the cell at zero and flag.
                logger.debug(
                    "  odor %s missing ROI %s — setting max/min to 0",
                    odor, roi.header,
                )
                continue
            col = df[roi.header].to_numpy(dtype=np.float32)
            # Treat NaN as "not observed"; fall back to 0 if whole column is NaN.
            col = col[np.isfinite(col)]
            if col.size == 0:
                continue
            max_m[oi, ri] = float(col.max())
            min_m[oi, ri] = float(col.min())

    logger.info(
        "DFFBundle: %d odors × %d ROIs. Max-ΔF/F range %.3f..%.3f",
        n_odors, n_rois, float(max_m.min()), float(max_m.max()),
    )
    return DFFBundle(
        odor_order=odor_order,
        roi_columns=roi_columns,
        max_matrix=max_m,
        min_matrix=min_m,
        traces=traces,
        source_dir=path,
    )


def match_rois_to_dff(
    fiji_rois,
    dff_bundle: DFFBundle,
    tolerance_px: float = 25.0,
) -> Dict[int, int]:
    """Match FIJI ROIs (from :class:`~.roi_loader.ROISet`) to dff CSV rows
    by nearest centroid within ``tolerance_px``.

    Args:
        fiji_rois: Iterable of :class:`~.roi_loader.ROI` instances (or a
            :class:`~.roi_loader.ROISet`).
        dff_bundle: Loaded DFFBundle.
        tolerance_px: Maximum Euclidean distance (image pixels) to accept a
            match. Larger → more FIJI ROIs get a dff partner; ROIs farther
            than this from any dff ROI stay unmatched.

    Returns:
        ``{fiji_roi_index → dff_roi_index}``. Missing entries mean no
        match was within tolerance.
    """
    if dff_bundle.n_rois == 0:
        return {}
    csv_xy = dff_bundle.roi_centroids()  # (N_dff, 2) as (x, y)
    result: Dict[int, int] = {}
    for i, roi in enumerate(fiji_rois):
        # FIJI ROI centroid in image coords.
        try:
            fx, fy = roi.centroid  # (x, y)
        except Exception:  # noqa: BLE001
            # Fall back to the bbox centre if centroid isn't available.
            x0, y0, x1, y1 = roi.bbox
            fx = (x0 + x1) / 2.0
            fy = (y0 + y1) / 2.0
        dists = np.hypot(csv_xy[:, 0] - fx, csv_xy[:, 1] - fy)
        j = int(np.argmin(dists))
        if dists[j] <= tolerance_px:
            result[i] = j
    logger.info(
        "match_rois_to_dff: %d/%d FIJI ROIs matched within %.1f px",
        len(result), i + 1 if result else 0, tolerance_px,
    )
    return result
