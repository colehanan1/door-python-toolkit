"""
DoOR literature response lookup
===============================

Wraps the DoOR consensus response matrix (693 odorants × 78 receptors)
together with the atlas_align-level mappings required to compare a
user's ΔF/F observations against DoOR per-glomerulus response vectors:

* **odorant name → InChIKey** via
  ``data/mappings/odorant_name_to_inchikey_complete.csv``
* **receptor name → glomerulus** via
  ``data/mappings/receptor_inventory.csv``

On first use the DoOR response matrix is downloaded from the ropensci
DoOR.data GitHub mirror and cached under ``~/.atlas_align/cache/``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from door_toolkit.atlas_align.config import USER_STATE_DIR, get_logger

logger = get_logger(__name__)

#: Download URL for DoOR's non-normalised response matrix (693 × 78).
DOOR_RESPONSE_URL = (
    "https://raw.githubusercontent.com/ropensci/DoOR.data/master/data/"
    "door_response_matrix_non_normalized.csv"
)

#: Where we cache the downloaded matrix so we only grab it once.
DEFAULT_RESPONSE_CACHE = USER_STATE_DIR / "cache" / "door_response_matrix.csv"

#: Default mapping: the canonical odor names the dff loader produces →
#: their DoOR common name. ``apple_cider_vinegar`` maps to acetic acid
#: because ACV is a mixture and acetic acid is the standard DoOR proxy.
DEFAULT_ODOR_TO_DOOR_NAME: Dict[str, str] = {
    "Apple_Cider_Vinegar": "Acetic Acid",
    "Benzaldehyde": "Benzaldehyde",
    "Citral": "Citral",
    "Ethyl_Butyrate": "Ethyl butyrate",
    "Hexanol": "1-Hexanol",
    "Linalool": "Linalool",
    "3-Octanol": "3-Octanol",
}

#: Built-in glomerulus → DoOR-receptor overrides that take priority over
#: whatever ``receptor_inventory.csv`` says. These fill documented gaps:
#:
#: * ``DL2d`` / ``DL2v``: both sit in the ac3 sensillum (Ir75a + Ir75b +
#:   Ir75c). Ir75b/c aren't in DoOR v2.0.0, but the ac3A column captures
#:   the combined sensillum response.
#: * ``VP1`` / ``VP4``: driven by Ir40a (coreceptor Ir25a). Neither
#:   receptor is in DoOR v2.0.0 — we still record the mapping so the
#:   Response dock shows *why* no data comes back.
BUILTIN_GLOMERULUS_RECEPTOR_OVERRIDES: Dict[str, str] = {
    "DL2d": "ac3A",
    "DL2v": "ac3A",
    "VP1": "Ir40a",
    "VP4": "Ir40a",
}

#: Optional user-edited override CSV. If present, its two columns
#: ``glomerulus,receptor`` are merged on top of the built-in overrides.
USER_OVERRIDE_CSV = USER_STATE_DIR / "glomerulus_receptor_override.csv"


@dataclass
class DoorResponseBundle:
    """Per-glomerulus DoOR response vectors for a chosen odor panel.

    Attributes:
        odor_order: The user-facing odor names in a fixed ordering (matches
            the dff bundle's odor order).
        door_names: The resolved DoOR common name for each odor. ``None``
            for odors we couldn't map.
        glomeruli: Glomerulus names covered. Each key has a DoOR response
            entry in ``response_matrix``.
        response_matrix: ``(n_glomeruli, n_odors)`` float32 array of DoOR
            responses, 0.0–1.0. ``NaN`` for receptor / odor pairs where
            DoOR has no data.
        receptor_by_glomerulus: The receptor DoOR uses for each glomerulus.
        unmapped_glomeruli: Names the caller asked about but that don't
            have a receptor in the DoOR mapping.
    """

    odor_order: List[str]
    door_names: List[Optional[str]]
    glomeruli: List[str]
    response_matrix: np.ndarray
    receptor_by_glomerulus: Dict[str, str]
    unmapped_glomeruli: List[str] = field(default_factory=list)

    @property
    def n_glomeruli(self) -> int:
        return len(self.glomeruli)

    @property
    def n_odors(self) -> int:
        return len(self.odor_order)

    def response_for_glomerulus(self, name: str) -> Optional[np.ndarray]:
        """``(n_odors,)`` DoOR response vector for a glomerulus (or None)."""
        try:
            idx = self.glomeruli.index(name)
        except ValueError:
            return None
        return self.response_matrix[idx].copy()


def _ensure_response_matrix_downloaded(
    cache_path: Path = DEFAULT_RESPONSE_CACHE,
) -> Path:
    """Fetch the DoOR response matrix on first use; cache it locally."""
    cache_path = Path(cache_path)
    if cache_path.is_file() and cache_path.stat().st_size > 10_000:
        return cache_path
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading DoOR response matrix: %s", DOOR_RESPONSE_URL)
    import urllib.request

    urllib.request.urlretrieve(DOOR_RESPONSE_URL, str(cache_path))
    logger.info(
        "Cached DoOR response matrix at %s (%.0f KB)",
        cache_path, cache_path.stat().st_size / 1024,
    )
    return cache_path


def _load_odorant_name_to_inchikey(mapping_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(mapping_csv)
    # Exact common_name lookup (case-insensitive) is all we need.
    lut: Dict[str, str] = {}
    for _, row in df.iterrows():
        name = str(row["common_name"]).strip()
        key = str(row["inchikey"]).strip()
        if not name or not key:
            continue
        lut[name.lower()] = key
    return lut


def _load_receptor_to_glomerulus(mapping_csv: Path) -> Dict[str, str]:
    """Return ``{receptor_name → glomerulus_name}`` (strip ORN_ prefix)."""
    df = pd.read_csv(mapping_csv)
    lut: Dict[str, str] = {}
    for _, row in df.iterrows():
        receptor = str(row["receptor_name"]).strip()
        glom_full = str(row.get("flywire_glomerulus", "")).strip()
        if not receptor or not glom_full or glom_full == "nan":
            continue
        glom = re.sub(r"^ORN_", "", glom_full)
        lut[receptor] = glom
    return lut


def _load_user_overrides() -> Dict[str, str]:
    """Read the optional user override CSV at
    :data:`USER_OVERRIDE_CSV`. Two columns: ``glomerulus,receptor``.
    Missing / malformed file is treated as "no overrides".
    """
    overrides: Dict[str, str] = {}
    path = USER_OVERRIDE_CSV
    if not path.is_file():
        return overrides
    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not parse %s: %s", path, e)
        return overrides
    if {"glomerulus", "receptor"}.issubset(df.columns):
        for _, row in df.iterrows():
            g = str(row["glomerulus"]).strip()
            r = str(row["receptor"]).strip()
            if g and r and g.lower() != "nan" and r.lower() != "nan":
                overrides[g] = r
        logger.info("Loaded %d user overrides from %s", len(overrides), path)
    else:
        logger.warning(
            "Override CSV %s must have 'glomerulus' and 'receptor' columns.",
            path,
        )
    return overrides


def _resolve_odor_rows(
    response_df: pd.DataFrame,
    odor_order: List[str],
    odorant_name_to_inchikey: Dict[str, str],
    odor_to_door_name: Dict[str, str],
) -> Tuple[List[Optional[str]], List[Optional[int]]]:
    """Return the DoOR display name + response-row index for each odor."""
    door_names: List[Optional[str]] = []
    rows: List[Optional[int]] = []
    for odor in odor_order:
        door_name = odor_to_door_name.get(odor)
        if door_name is None:
            door_names.append(None)
            rows.append(None)
            continue
        key = odorant_name_to_inchikey.get(door_name.lower())
        if key is None:
            logger.warning(
                "No InChIKey for DoOR name %r (from %s)", door_name, odor
            )
            door_names.append(door_name)
            rows.append(None)
            continue
        # The response CSV's first column is the InChIKey index.
        if key in response_df.index:
            row_idx = response_df.index.get_loc(key)
        else:
            logger.warning(
                "InChIKey %s not in DoOR response matrix", key
            )
            row_idx = None
        door_names.append(door_name)
        rows.append(row_idx)
    return door_names, rows


def load_door_responses(
    odor_order: Iterable[str],
    glomeruli: Iterable[str],
    mappings_dir: Path,
    response_cache: Path = DEFAULT_RESPONSE_CACHE,
    odor_to_door_name: Optional[Dict[str, str]] = None,
) -> DoorResponseBundle:
    """Build a :class:`DoorResponseBundle` for the requested odors + glomeruli.

    Args:
        odor_order: Odor names (as used by the dff loader).
        glomeruli: Glomerulus names to look up.
        mappings_dir: Directory containing
            ``odorant_name_to_inchikey_complete.csv`` and
            ``receptor_inventory.csv``.
        response_cache: Path to the DoOR response matrix CSV (downloaded
            automatically on first use).
        odor_to_door_name: Override for the odor→DoOR-name map. Defaults
            to :data:`DEFAULT_ODOR_TO_DOOR_NAME`.
    """
    odor_order = list(odor_order)
    glomeruli = list(glomeruli)

    csv_path = _ensure_response_matrix_downloaded(response_cache)
    mappings_dir = Path(mappings_dir)

    # Response matrix: row = InChIKey, columns = receptors.
    response_df = pd.read_csv(csv_path, sep=";", decimal=".", index_col=0)
    logger.info(
        "Loaded DoOR response matrix: %d odorants × %d receptors",
        response_df.shape[0], response_df.shape[1],
    )

    odorant_name_to_inchikey = _load_odorant_name_to_inchikey(
        mappings_dir / "odorant_name_to_inchikey_complete.csv"
    )
    receptor_to_glomerulus = _load_receptor_to_glomerulus(
        mappings_dir / "receptor_inventory.csv"
    )
    # Reverse map: glomerulus → receptor (first receptor wins for dups).
    glomerulus_to_receptor: Dict[str, str] = {}
    for receptor, glom in receptor_to_glomerulus.items():
        glomerulus_to_receptor.setdefault(glom, receptor)

    # Built-in overrides fill documented gaps (DL2d/v, VP1, VP4).
    for glom, receptor in BUILTIN_GLOMERULUS_RECEPTOR_OVERRIDES.items():
        glomerulus_to_receptor[glom] = receptor
    # User overrides win over everything.
    for glom, receptor in _load_user_overrides().items():
        glomerulus_to_receptor[glom] = receptor

    odor_to_door_name = dict(
        odor_to_door_name or DEFAULT_ODOR_TO_DOOR_NAME
    )

    door_names, odor_rows = _resolve_odor_rows(
        response_df, odor_order, odorant_name_to_inchikey, odor_to_door_name
    )

    covered_glomeruli: List[str] = []
    covered_receptors: Dict[str, str] = {}
    row_list: List[np.ndarray] = []
    unmapped: List[str] = []
    receptor_not_in_matrix: Dict[str, str] = {}  # glom → receptor name
    for glom in glomeruli:
        receptor = glomerulus_to_receptor.get(glom)
        if receptor is None:
            unmapped.append(glom)
            continue
        if receptor not in response_df.columns:
            # The mapping is known but DoOR v2.0.0 doesn't carry data for
            # this receptor (e.g. Ir40a, Ir25a, Ir75b, Ir75c). Record the
            # link with an all-NaN row so the GUI can still say "no data
            # for receptor X" rather than silently dropping the glomerulus.
            row_list.append(np.full(len(odor_order), np.nan, dtype=np.float32))
            covered_glomeruli.append(glom)
            covered_receptors[glom] = receptor
            receptor_not_in_matrix[glom] = receptor
            continue
        row = np.full(len(odor_order), np.nan, dtype=np.float32)
        for i, row_idx in enumerate(odor_rows):
            if row_idx is None:
                continue
            val = response_df.iat[row_idx, response_df.columns.get_loc(receptor)]
            if pd.isna(val):
                continue
            row[i] = float(val)
        row_list.append(row)
        covered_glomeruli.append(glom)
        covered_receptors[glom] = receptor

    matrix = (
        np.vstack(row_list) if row_list
        else np.zeros((0, len(odor_order)), dtype=np.float32)
    )
    logger.info(
        "DoorResponseBundle: %d glomeruli with receptor mapping "
        "(%d with usable DoOR data, %d receptors not in matrix), %d unmapped",
        len(covered_glomeruli),
        len(covered_glomeruli) - len(receptor_not_in_matrix),
        len(receptor_not_in_matrix),
        len(unmapped),
    )
    return DoorResponseBundle(
        odor_order=odor_order,
        door_names=door_names,
        glomeruli=covered_glomeruli,
        response_matrix=matrix,
        receptor_by_glomerulus=covered_receptors,
        unmapped_glomeruli=unmapped,
    )


def cosine_similarity(
    user_vec: np.ndarray, door_vec: np.ndarray
) -> float:
    """Cosine similarity (uncentred) robust to NaN entries in the DoOR vector.

    Kept around for reference, but :func:`pearson_correlation` is the
    recommended primary metric — cosine is biased upward whenever both
    inputs are all-non-negative, which is exactly the case for raw
    ΔF/F vs DoOR response values.
    """
    u = np.asarray(user_vec, dtype=np.float64)
    d = np.asarray(door_vec, dtype=np.float64)
    mask = np.isfinite(u) & np.isfinite(d)
    if int(mask.sum()) < 2:
        return float("nan")
    u = u[mask]
    d = d[mask]
    nu = float(np.linalg.norm(u))
    nd = float(np.linalg.norm(d))
    if nu == 0.0 or nd == 0.0:
        return float("nan")
    return float(np.dot(u, d) / (nu * nd))


def pearson_correlation(
    user_vec: np.ndarray, door_vec: np.ndarray
) -> float:
    """Pearson correlation robust to NaN entries.

    This is cosine of the *mean-centred* vectors, so it measures **pattern
    agreement** across odors rather than raw positive-orthant overlap. For
    a user's ΔF/F max vector vs a DoOR response vector, this is what you
    actually want: +1 = same peaks / troughs, 0 = unrelated, -1 = anti-
    correlated.
    """
    u = np.asarray(user_vec, dtype=np.float64)
    d = np.asarray(door_vec, dtype=np.float64)
    mask = np.isfinite(u) & np.isfinite(d)
    if int(mask.sum()) < 2:
        return float("nan")
    u = u[mask]
    d = d[mask]
    u = u - u.mean()
    d = d - d.mean()
    nu = float(np.linalg.norm(u))
    nd = float(np.linalg.norm(d))
    if nu == 0.0 or nd == 0.0:
        return float("nan")
    return float(np.dot(u, d) / (nu * nd))


# ---------------------------------------------------------------------------
# DoOR pairwise monotonic projection (port of DoOR.functions::project_points
# + back_project). Places a user response vector onto DoOR's per-receptor
# [0, 1] consensus scale by fitting the best of five monotonic functions
# through their overlapping (normalized) odor points.
# ---------------------------------------------------------------------------

#: Minimum overlapping odors required to attempt a monotonic fit.
#: DoOR's own threshold is 5 — that prevents 3-parameter models from
#: overfitting 3-4 point pairs. Our odor panels are small (~7 odors)
#: and receptors routinely have 2-3 NaN entries in that subset, so a
#: strict 5-floor leaves the column blank too often. We lower the floor
#: to 3 and *instead* restrict the candidate-model set by parameter
#: count (``_model_param_counts`` below) so a linear (2-param) fit is
#: the only thing that survives at the minimum.
_MIN_OVERLAP_FOR_MERGE: int = 3


def _door_norm_minmax(x: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """DoOR's ``door_norm``: min-max rescale to [0, 1], returning the
    ``(x_norm, x_min, x_max)`` triple so callers can invert.

    Flat inputs return zeros (DoOR's convention).
    """
    xmin = float(np.nanmin(x))
    xmax = float(np.nanmax(x))
    rng = xmax - xmin
    if rng == 0 or not np.isfinite(rng):
        return np.zeros_like(x, dtype=np.float64), xmin, xmax
    return (x.astype(np.float64) - xmin) / rng, xmin, xmax


def _model_linear(x, a, b):
    return a + b * x


def _model_exp(x, a, b, c):
    return a + b * np.exp(c * x)


def _model_sigmoid(x, asym, xmid, scal):
    # SSlogis: Asym / (1 + exp((xmid - x) / scal))
    return asym / (1.0 + np.exp((xmid - x) / scal))


def _model_asymp(x, asym, r0, lrc):
    # SSasymp: Asym + (R0 - Asym) * exp(-exp(lrc) * x)
    return asym + (r0 - asym) * np.exp(-np.exp(lrc) * x)


def _model_asymp_off(x, asym, lrc, c0):
    # SSasympOff: Asym * (1 - exp(-exp(lrc) * (x - c0)))
    return asym * (1.0 - np.exp(-np.exp(lrc) * (x - c0)))


#: Candidate ``(name, fn, initial_params, param_bounds)`` list.
#: Mirrors DoOR.functions::modelfunction_* (forward only — for our port
#: we don't flip the x/y roles since the caller decides direction).
_CANDIDATE_MODELS: Tuple[Tuple[str, object, Tuple[float, ...], Tuple[Tuple[float, ...], Tuple[float, ...]]], ...] = (
    ("linear",    _model_linear,    (0.0, 1.0),        ((-np.inf, -np.inf), (np.inf, np.inf))),
    ("exp",       _model_exp,       (0.0, 0.1, 1.0),   ((-np.inf, -np.inf, -10.0), (np.inf, np.inf, 10.0))),
    ("sigmoid",   _model_sigmoid,   (1.0, 0.5, 0.2),   ((0.0, -5.0, 1e-3), (5.0, 5.0, 10.0))),
    ("asymp",     _model_asymp,     (1.0, 0.0, 0.0),   ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))),
    ("asymp_off", _model_asymp_off, (1.0, 0.0, 0.0),   ((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0))),
)


def _fit_best_monotonic(
    x_norm: np.ndarray, y_norm: np.ndarray
) -> Optional[Tuple[str, object, Tuple[float, ...], float]]:
    """Fit each candidate model, reject non-monotonic fits, return the
    best one by residual RMS.

    The candidate set is restricted by the number of data points
    available: a model with ``p`` parameters needs at least ``p + 1``
    points to leave any residual degree of freedom. With fewer points
    a 3-param model (exp / sigmoid / asymp / asymp_off) would overfit,
    so we drop them and fall back to linear only.

    Returns ``(name, fn, popt, rms)`` or ``None`` if nothing converged.
    """
    from scipy.optimize import curve_fit

    n_points = int(x_norm.size)
    x_fine = np.linspace(0.0, 1.0, 100)
    best: Optional[Tuple[str, object, Tuple[float, ...], float]] = None

    for name, fn, p0, bounds in _CANDIDATE_MODELS:
        n_params = len(p0)
        # Eligibility rules (per-model version of DoOR's 5-odor floor):
        #   * linear (2 params): need ≥ 3 points. Enough to define a
        #     line with 1 residual DoF — safe for overlap=3 fallbacks.
        #   * 3-param models: need ≥ 5 points (2 residual DoF). Below
        #     that a 3-param curve will near-perfectly interpolate,
        #     out-rank linear on RMS, then extrapolate wildly on the
        #     non-overlap odors. DoOR's original 5-floor was set for
        #     exactly this reason.
        min_points = 3 if n_params == 2 else 5
        if n_points < min_points:
            continue
        try:
            popt, _ = curve_fit(
                fn, x_norm, y_norm, p0=p0, bounds=bounds, maxfev=2000
            )
        except (RuntimeError, ValueError):
            continue
        try:
            y_fine = fn(x_fine, *popt)
        except Exception:  # noqa: BLE001
            continue
        if not np.all(np.isfinite(y_fine)):
            continue
        dy = np.diff(y_fine)
        if not (np.all(dy >= -1e-4) or np.all(dy <= 1e-4)):
            # non-monotonic — DoOR only keeps monotonic fits
            continue
        y_pred = fn(x_norm, *popt)
        rms = float(np.sqrt(np.mean((y_pred - y_norm) ** 2)))
        if best is None or rms < best[3]:
            best = (name, fn, tuple(popt), rms)

    return best


@dataclass
class DoorProjection:
    """Outcome of :func:`project_to_door_scale`."""

    projected: np.ndarray       # user's vector on DoOR's per-receptor [0, 1] scale
    model_name: str             # "linear" / "exp" / "sigmoid" / "asymp" / "asymp_off" / ""
    rms: float                  # fit residual RMS (on normalized [0, 1] space)
    n_overlap: int              # number of (user, DoOR) odor pairs actually used
    ok: bool                    # True iff fit succeeded


def project_to_door_scale(
    user_vec: np.ndarray, door_vec: np.ndarray
) -> DoorProjection:
    """Project a user scalar response vector onto DoOR's per-receptor scale.

    Simplified port of ``DoOR.functions::back_project``:

      1. ``door_norm`` both vectors to [0, 1] on their common, non-NaN
         odors.
      2. Fit the best of five monotonic functions (linear, exponential,
         sigmoid, asymptotic, asymptotic-with-offset) by residual RMS.
      3. Apply the fit to every finite user entry (not just the overlap)
         to get a DoOR-normalized projected value.
      4. Multiply back by the DoOR vector's observed range so the output
         is on the same magnitude scale DoOR's non-normalized consensus
         matrix uses for that receptor.

    Args:
        user_vec: (n_odors,) user scalar response.
        door_vec: (n_odors,) DoOR response for the target receptor.

    Returns:
        :class:`DoorProjection`. ``projected`` is all-NaN if fewer than
        :data:`_MIN_OVERLAP_FOR_MERGE` overlapping odors exist or no
        monotonic model converges.
    """
    u = np.asarray(user_vec, dtype=np.float64)
    d = np.asarray(door_vec, dtype=np.float64)
    if u.shape != d.shape:
        raise ValueError(
            f"user_vec shape {u.shape} != door_vec shape {d.shape}"
        )
    mask = np.isfinite(u) & np.isfinite(d)
    n_overlap = int(mask.sum())
    nan_out = np.full(u.shape, np.nan, dtype=np.float32)

    if n_overlap < _MIN_OVERLAP_FOR_MERGE:
        return DoorProjection(
            projected=nan_out,
            model_name="",
            rms=float("nan"),
            n_overlap=n_overlap,
            ok=False,
        )

    u_common = u[mask]
    d_common = d[mask]
    u_n, u_min, u_max = _door_norm_minmax(u_common)
    d_n, d_min, d_max = _door_norm_minmax(d_common)
    if u_max == u_min or d_max == d_min:
        return DoorProjection(
            projected=nan_out,
            model_name="",
            rms=float("nan"),
            n_overlap=n_overlap,
            ok=False,
        )

    best = _fit_best_monotonic(u_n, d_n)
    if best is None:
        return DoorProjection(
            projected=nan_out,
            model_name="",
            rms=float("nan"),
            n_overlap=n_overlap,
            ok=False,
        )

    name, fn, popt, rms = best

    # Apply fit to ALL finite user values (not just the overlap).
    out = np.full(u.shape, np.nan, dtype=np.float64)
    u_finite_mask = np.isfinite(u)
    u_finite_n = (u[u_finite_mask] - u_min) / (u_max - u_min)
    try:
        projected_n = fn(u_finite_n, *popt)
    except Exception:  # noqa: BLE001
        return DoorProjection(
            projected=nan_out, model_name="",
            rms=float("nan"), n_overlap=n_overlap, ok=False,
        )
    # Clip to [0, 1] before rescaling back — fit can extrapolate slightly
    # when user values sit outside the overlap range.
    projected_n = np.clip(projected_n, 0.0, 1.0)
    # Linear back-transform into DoOR's original magnitude range.
    out[u_finite_mask] = projected_n * (d_max - d_min) + d_min

    return DoorProjection(
        projected=out.astype(np.float32),
        model_name=name,
        rms=rms,
        n_overlap=n_overlap,
        ok=True,
    )


def rank_glomeruli_by_similarity(
    user_max_vec: np.ndarray,
    bundle: DoorResponseBundle,
    top_n: int = 3,
    metric: str = "pearson",
) -> List[Tuple[str, float]]:
    """Return ``[(glomerulus, score), ...]`` ranked best-first.

    Args:
        metric: ``"pearson"`` (default, recommended) or ``"cosine"``.
    """
    fn = pearson_correlation if metric == "pearson" else cosine_similarity
    scores: List[Tuple[str, float]] = []
    for i, glom in enumerate(bundle.glomeruli):
        sim = fn(user_max_vec, bundle.response_matrix[i])
        if not np.isfinite(sim):
            continue
        scores.append((glom, sim))
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores[:top_n]
