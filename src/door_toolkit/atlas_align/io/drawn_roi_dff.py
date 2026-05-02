"""
Drawn-ROI ΔF/F extractor (DoOR-style)
=====================================

Per-polygon ΔF/F + scalar response, methodology-matched to
Munch & Galizia (2016) Sci Rep 6:21841 "DoOR 2.0", Methods
"Data analysis: calcium imaging" (pp. 11-12). Shares its numerics
with the standalone ``scripts/extract_imagej_roi_signals.py`` so values
generated in the GUI are directly comparable to DoOR glomerular vectors.

Key design decisions:

* **Memory-mapped stacks.** The ``al_cropped_stack.tif`` files for each
  trial are 1.7 GB each; memmapping means we only read the pixels
  inside the polygon, not the whole movie.
* **First trial per odor** (matching the user's original script).
* **DoOR bleach correction.** The exponential fit excludes the evoked
  window (``[odor_on − 8, odor_off + 110]`` at 10 FPS) and up-weights
  pre-stim samples 100× via ``curve_fit`` ``sigma``. Falls back to a
  weighted linear trend on the same fit mask.
* **F₀ = pre-odor mean.** Full pre-odor segment (frames
  ``[0, odor_on)``) from each trial's ``trial.json``.
* **Signed DoOR scalar.** ``mean(ΔF/F[on, on+5 s])
  − mean(ΔF/F[on−2.5 s, on])`` at 10 FPS. Negative values preserved so
  inhibitory responses are recoverable.

The returned :class:`DrawnROIDff` exposes both the full ΔF/F traces and
the **signed DoOR scalar per odor**, which is what the Response dock
and DoOR comparison consume via :meth:`DrawnROIDff.scalar_dff_vector`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tifffile

from door_toolkit.atlas_align.config import get_logger
from door_toolkit.atlas_align.io.dff_loader import DEFAULT_ODOR_LETTER_MAP

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DoOR-method parameters (must match extract_imagej_roi_signals.py)
# ---------------------------------------------------------------------------

#: Imaging rate for the fly-4 protocol.
FPS: float = 10.0

#: Frames omitted from the bleach fit so the exponential is not pulled
#: into the evoked response. 0.75 s pre-onset and 11 s post-offset at 10 FPS.
FIT_EXCLUDE_PRE_FRAMES: int = 8
FIT_EXCLUDE_POST_FRAMES: int = 110

#: Per-sample sigma in ``curve_fit``. Pre-odor samples get
#: ``SIGMA_PRE_ODOR`` so their per-point cost weight
#: (∝ 1/σ²) is 100× larger than post-odor.
SIGMA_PRE_ODOR: float = 0.1
SIGMA_POST_ODOR: float = 1.0

#: Scalar response window — post is the full odor window from
#: ``trial.json`` (default frames 150..190 = 4 s), pre is a fixed
#: 21-frame (≈2.1 s) span ending right before odor onset
#: (default frames 129..149).
SCALAR_PRE_FRAMES: int = 21

#: Default per-odor window if ``trial.json`` is missing / malformed.
DEFAULT_ODOR_ON: int = 150
DEFAULT_ODOR_OFF: int = 190

#: Subdirectories under each trial's ``images/`` folder to search for the
#: pre-cropped stack, in descending preference order.
_PREPROC_SUBDIRS: Tuple[str, ...] = (
    "caiman_pnr_preproc_p70",
    "caiman_pnr_preproc_p80",
    "caiman_pnr_preproc_p90",
    "caiman_run",
)
_STACK_FILENAME: str = "al_cropped_stack.tif"
_TRIAL_DIR_RE = re.compile(r"^trial_(?P<num>\d+)_OFM_(?P<letter>[A-Za-z])$")


# ---------------------------------------------------------------------------
# Numerics: DoOR bleach correction + signed scalar response
# ---------------------------------------------------------------------------


def _exp_decay(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(-b * t) + c


@dataclass
class _BleachFit:
    """Per-trace bleach-fit artifacts (internal)."""

    corrected: np.ndarray     # (T,) float64
    fitted: np.ndarray        # (T,) float64 — model over full trace
    exp_ok: bool              # False → weighted linear fallback
    residual_rms: float       # on the fit region


def bleach_correct(
    trace: np.ndarray,
    odor_on: int,
    odor_off: int,
) -> _BleachFit:
    """Fit A·exp(−B·t)+C to the non-evoked frames and divide it out.

    Matches :func:`extract_imagej_roi_signals.bleach_correct` exactly.

    * Fit region = full trace with ``[odor_on − FIT_EXCLUDE_PRE_FRAMES,
      odor_off + FIT_EXCLUDE_POST_FRAMES)`` removed.
    * Pre-odor samples (``t < odor_on``) carry ``SIGMA_PRE_ODOR`` so
      their per-point cost is 100× the post-odor samples.
    * Division is clamped at 1% of ``fitted.max()`` to avoid
      blow-ups on a late tail.
    * On ``curve_fit`` failure, falls back to a weighted linear
      regression on the same fit mask (never a silent bare except).
    """
    from scipy.optimize import curve_fit

    T = int(len(trace))
    trace_f = np.asarray(trace, dtype=np.float64)
    t_all = np.arange(T, dtype=np.float64)

    exclude_start = max(0, int(odor_on) - FIT_EXCLUDE_PRE_FRAMES)
    exclude_end = min(T, int(odor_off) + FIT_EXCLUDE_POST_FRAMES)
    fit_mask = np.ones(T, dtype=bool)
    fit_mask[exclude_start:exclude_end] = False
    if int(fit_mask.sum()) < 10:
        fit_mask = np.ones(T, dtype=bool)

    sigma = np.full(T, SIGMA_POST_ODOR, dtype=np.float64)
    sigma[: max(0, min(T, int(odor_on)))] = SIGMA_PRE_ODOR

    t_fit = t_all[fit_mask]
    y_fit = trace_f[fit_mask]
    sigma_fit = sigma[fit_mask]

    a0 = max(float(trace_f[0] - trace_f[-1]), 1.0)
    c0 = max(float(trace_f[-1]), 1.0)
    b0 = 0.01

    exp_ok = True
    try:
        popt, _ = curve_fit(
            _exp_decay, t_fit, y_fit,
            p0=[a0, b0, c0],
            sigma=sigma_fit, absolute_sigma=False,
            bounds=([0.0, 0.0, 0.0], [np.inf, 1.0, np.inf]),
            maxfev=5000,
        )
        fitted = _exp_decay(t_all, *popt).astype(np.float64)
    except (RuntimeError, ValueError) as exc:
        logger.warning(
            "bleach_correct: exp-fit failed (%s: %s); "
            "falling back to weighted linear trend.",
            type(exc).__name__, exc,
        )
        w = 1.0 / sigma_fit
        coef = np.polyfit(t_fit, y_fit, deg=1, w=w)
        fitted = np.polyval(coef, t_all).astype(np.float64)
        exp_ok = False

    floor = max(float(fitted.max()) * 0.01, 1e-9)
    fitted_safe = np.where(fitted < floor, floor, fitted)
    corrected = trace_f * (fitted_safe[0] / fitted_safe)

    resid = trace_f[fit_mask] - fitted[fit_mask]
    rms = float(np.sqrt((resid ** 2).mean())) if resid.size else float("nan")

    return _BleachFit(
        corrected=corrected,
        fitted=fitted_safe,
        exp_ok=exp_ok,
        residual_rms=rms,
    )


def door_row_scale(vec: np.ndarray) -> np.ndarray:
    """Rescale a 1-D response vector onto DoOR's per-receptor scale.

    Port of ``DoOR.functions::door_norm`` → ``reset_sfr`` for our
    signed-ΔF/F input. DoOR treats SFR (spontaneous firing rate) as
    "just another odor" in the data matrix; for calcium-imaging studies
    it sets that SFR row to 0. So ``min`` and ``max`` in DoOR's
    ``door_norm`` always include 0 as a data point. Implemented here
    with an effective range:

        eff_min = min(vec.min(), 0)
        eff_max = max(vec.max(), 0)
        scaled  = vec / (eff_max − eff_min)

    This keeps the result within ``[-1, +1]``:
    * all-positive row → ``[0, vec.max / vec.max] = [0, 1]``
    * all-negative row → ``[vec.min / |vec.min|, 0] = [-1, 0]``
    * mixed row        → excitation positive, inhibition negative,
      total span = 1.0, matching DoOR's per-receptor [0, 1] scaling
      with SFR reset to 0.

    Flat / all-NaN vectors are returned unchanged.
    """
    v = np.asarray(vec, dtype=np.float64)
    if v.size == 0 or not np.isfinite(v).any():
        return v.copy()
    eff_min = min(float(np.nanmin(v)), 0.0)
    eff_max = max(float(np.nanmax(v)), 0.0)
    span = eff_max - eff_min
    if span == 0.0 or not np.isfinite(span):
        return v.copy()
    return v / span


def response_scalar(
    dff: np.ndarray, odor_on: int, odor_off: int
) -> float:
    """Signed scalar response: odor-window mean minus pre-window mean.

    * Post window = ``dff[odor_on : odor_off]`` (full odor pulse, 40 frames
      / 4 s for the default ``150..190`` range).
    * Pre window  = ``dff[odor_on − SCALAR_PRE_FRAMES : odor_on]``
      (21 frames / ≈ 2.1 s, default ``129..149``).

    Sign preserved; no clipping. Returns ``0.0`` only if either window
    falls entirely off the trace.
    """
    T = int(len(dff))
    on = max(0, min(T, int(odor_on)))
    off = max(on, min(T, int(odor_off)))
    pre_start = max(0, on - SCALAR_PRE_FRAMES)
    if off <= on or on <= pre_start:
        return 0.0
    return float(dff[on:off].mean() - dff[pre_start:on].mean())


# ---------------------------------------------------------------------------
# Per-polygon extraction container
# ---------------------------------------------------------------------------


@dataclass
class DrawnROIDff:
    """Per-odor ΔF/F traces + signed DoOR scalar for one polygon."""

    odor_order: List[str]
    trial_for_odor: Dict[str, str]
    raw_traces: Dict[str, np.ndarray]          # odor -> (T,) raw mean trace
    corrected_traces: Dict[str, np.ndarray]    # odor -> (T,) bleach-corrected
    dff_traces: Dict[str, np.ndarray]          # odor -> (T,) ΔF/F
    scalar_dff: Dict[str, float]               # odor -> signed DoOR scalar
    odor_windows: Dict[str, Tuple[int, int]]   # odor -> (on, off)
    fit_exp_ok: Dict[str, bool] = field(default_factory=dict)
    fit_rms: Dict[str, float] = field(default_factory=dict)

    def scalar_dff_vector(self) -> np.ndarray:
        """Signed DoOR scalar per odor, in :attr:`odor_order`."""
        return np.array(
            [self.scalar_dff.get(o, 0.0) for o in self.odor_order],
            dtype=np.float32,
        )

    def scalar_dff_door_scaled_vector(self) -> np.ndarray:
        """DoOR-calibrated per-odor scalar (``door_norm`` + ``reset_sfr`` @ SFR=0).

        Each ROI's vector is divided by its own peak-to-trough range, so
        excitation and inhibition scale symmetrically around 0 and the
        row's dynamic range is 1.0 — directly comparable to DoOR's
        per-receptor [0, 1] consensus vectors.
        """
        return door_row_scale(self.scalar_dff_vector()).astype(np.float32)

    # Backwards-compat alias — older callers may still reach for peak_dff.
    @property
    def peak_dff(self) -> Dict[str, float]:  # pragma: no cover - alias
        return self.scalar_dff

    def peak_dff_vector(self) -> np.ndarray:  # pragma: no cover - alias
        return self.scalar_dff_vector()


class DrawnROIDffExtractor:
    """Lazy per-polygon ΔF/F computer. Caches memmapped stacks per odor."""

    def __init__(
        self,
        fly_dir: Path,
        odor_order: List[str],
        *,
        odor_letter_map: Optional[Dict[str, str]] = None,
        baseline_frames: Optional[int] = None,  # deprecated — ignored
    ) -> None:
        self.fly_dir = Path(fly_dir).expanduser().resolve()
        self.odor_order = list(odor_order)
        self._odor_letter_map = dict(odor_letter_map or DEFAULT_ODOR_LETTER_MAP)
        self._name_to_letter = {v: k for k, v in self._odor_letter_map.items()}
        if baseline_frames is not None:
            logger.info(
                "DrawnROIDffExtractor: `baseline_frames=%s` is ignored — "
                "F₀ now uses the full pre-odor segment per trial.",
                baseline_frames,
            )

        al_meta = json.loads(
            (self.fly_dir / "analysis" / "al_roi" / "al_meta.json").read_text()
        )
        self.bbox_y0_y1_x0_x1: List[int] = list(al_meta["bbox_y0_y1_x0_x1"])
        logger.info(
            "DrawnROIDffExtractor fly_dir=%s bbox=%s "
            "fit-exclude=[odor_on−%d, odor_off+%d)  "
            "scalar: post=[odor_on, odor_off)  pre=[odor_on−%d, odor_on)  "
            "sigma_pre=%.2f sigma_post=%.2f",
            self.fly_dir, self.bbox_y0_y1_x0_x1,
            FIT_EXCLUDE_PRE_FRAMES, FIT_EXCLUDE_POST_FRAMES,
            SCALAR_PRE_FRAMES,
            SIGMA_PRE_ODOR, SIGMA_POST_ODOR,
        )

        self._trial_for_odor: Dict[str, str] = self._resolve_trials()
        self._odor_windows: Dict[str, Tuple[int, int]] = self._load_odor_windows()
        self._stack_cache: Dict[str, np.ndarray] = {}

    # ----------------------------------------------------- setup

    def _resolve_trials(self) -> Dict[str, str]:
        """Find first trial per odor letter (matches original script)."""
        by_letter: Dict[str, Tuple[int, str]] = {}
        for sub in self.fly_dir.iterdir():
            if not sub.is_dir():
                continue
            m = _TRIAL_DIR_RE.match(sub.name)
            if m is None:
                continue
            letter = m.group("letter").upper()
            num = int(m.group("num"))
            if letter not in by_letter or by_letter[letter][0] > num:
                by_letter[letter] = (num, sub.name)
        mapping: Dict[str, str] = {}
        for odor in self.odor_order:
            letter = self._name_to_letter.get(odor)
            if letter is None:
                continue
            entry = by_letter.get(letter)
            if entry is not None:
                mapping[odor] = entry[1]
                logger.info("  odor %s → %s", odor, entry[1])
            else:
                logger.warning("  odor %s (letter %s) has no trial folder", odor, letter)
        return mapping

    def _load_odor_windows(self) -> Dict[str, Tuple[int, int]]:
        """Read odor on/off frame indices from each trial's ``trial.json``."""
        windows: Dict[str, Tuple[int, int]] = {}
        for odor, trial_name in self._trial_for_odor.items():
            tj = self.fly_dir / trial_name / "trial.json"
            if tj.is_file():
                try:
                    meta = json.loads(tj.read_text())
                    windows[odor] = (
                        int(meta["acquisition"]["odor_frame_start"]),
                        int(meta["acquisition"]["odor_frame_end"]),
                    )
                except (KeyError, ValueError):
                    windows[odor] = (DEFAULT_ODOR_ON, DEFAULT_ODOR_OFF)
            else:
                windows[odor] = (DEFAULT_ODOR_ON, DEFAULT_ODOR_OFF)
        return windows

    def _find_stack_path(self, trial_name: str) -> Optional[Path]:
        images = self.fly_dir / trial_name / "images"
        for sub in _PREPROC_SUBDIRS:
            cand = images / sub / _STACK_FILENAME
            if cand.is_file():
                return cand
        return None

    def _get_stack(self, odor: str) -> Optional[np.ndarray]:
        """Return a (T, H, W) memmapped array for this odor's first trial."""
        if odor in self._stack_cache:
            return self._stack_cache[odor]
        trial_name = self._trial_for_odor.get(odor)
        if trial_name is None:
            return None
        path = self._find_stack_path(trial_name)
        if path is None:
            logger.warning("  no cropped stack under trial %s", trial_name)
            return None
        try:
            mm = tifffile.memmap(str(path))
            logger.info("  memmap %s (%s)", path, mm.shape)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "  memmap failed for %s (%s); falling back to full imread.",
                path, e,
            )
            mm = tifffile.imread(str(path)).astype(np.float32, copy=False)
        self._stack_cache[odor] = mm
        return mm

    # ----------------------------------------------------- public

    @property
    def ready_odors(self) -> List[str]:
        return [o for o in self.odor_order if o in self._trial_for_odor]

    def compute_for_polygon(
        self,
        xs_fullframe: np.ndarray,
        ys_fullframe: np.ndarray,
    ) -> DrawnROIDff:
        """DoOR-style ΔF/F + scalar for one polygon across all odors.

        Polygon vertices are in the same coordinate system as the
        reference image (e.g. fullframe 2048×2048). We subtract the AL
        bbox to reach cropped-stack coords, rasterise, average per frame,
        then apply the DoOR bleach correction and signed scalar response.
        """
        from skimage.draw import polygon as sk_polygon

        Y0, Y1, X0, X1 = self.bbox_y0_y1_x0_x1
        H, W = Y1 - Y0, X1 - X0
        xs = np.asarray(xs_fullframe, dtype=np.float64) - X0
        ys = np.asarray(ys_fullframe, dtype=np.float64) - Y0
        rr, cc = sk_polygon(ys, xs, shape=(H, W))

        raw_traces: Dict[str, np.ndarray] = {}
        corrected_traces: Dict[str, np.ndarray] = {}
        dff_traces: Dict[str, np.ndarray] = {}
        scalar_dff: Dict[str, float] = {}
        fit_exp_ok: Dict[str, bool] = {}
        fit_rms: Dict[str, float] = {}

        for odor in self.odor_order:
            stack = self._get_stack(odor)
            if stack is None:
                continue
            T = int(stack.shape[0])
            if rr.size == 0:
                raw = np.zeros(T, dtype=np.float64)
            else:
                # Read only the polygon pixels per frame — sequential
                # reads per frame are much friendlier to memmap chunking
                # than a single fancy-index over all frames at once.
                polygon_vals = np.asarray(stack[:, rr, cc], dtype=np.float64)
                raw = polygon_vals.mean(axis=1) if polygon_vals.size else np.zeros(T)

            odor_on, odor_off = self._odor_windows.get(
                odor, (DEFAULT_ODOR_ON, DEFAULT_ODOR_OFF)
            )
            odor_on = max(0, min(T, int(odor_on)))
            odor_off = max(odor_on + 1, min(T, int(odor_off)))

            fit = bleach_correct(raw, odor_on=odor_on, odor_off=odor_off)
            corrected = fit.corrected

            # F₀ = mean of the full pre-odor segment (Methods p. 11).
            if odor_on >= 1:
                f0 = float(corrected[:odor_on].mean())
            else:
                f0 = 1.0
            if f0 == 0.0 or not np.isfinite(f0):
                f0 = 1.0
            dff = (corrected - f0) / f0

            scalar = response_scalar(dff, odor_on=odor_on, odor_off=odor_off)

            logger.info(
                "  [%s] trial=%s T=%d odor=[%d,%d) fit=%s rms=%.3g "
                "F₀=%.2f scalar=%+.4f",
                odor, self._trial_for_odor.get(odor, "?"), T,
                odor_on, odor_off,
                "exp" if fit.exp_ok else "linear-fallback",
                fit.residual_rms, f0, scalar,
            )

            raw_traces[odor] = raw.astype(np.float32)
            corrected_traces[odor] = corrected.astype(np.float32)
            dff_traces[odor] = dff.astype(np.float32)
            scalar_dff[odor] = scalar
            fit_exp_ok[odor] = fit.exp_ok
            fit_rms[odor] = fit.residual_rms

        return DrawnROIDff(
            odor_order=list(raw_traces.keys()),
            trial_for_odor={o: self._trial_for_odor[o] for o in raw_traces},
            raw_traces=raw_traces,
            corrected_traces=corrected_traces,
            dff_traces=dff_traces,
            scalar_dff=scalar_dff,
            odor_windows={
                o: self._odor_windows.get(o, (DEFAULT_ODOR_ON, DEFAULT_ODOR_OFF))
                for o in raw_traces
            },
            fit_exp_ok=fit_exp_ok,
            fit_rms=fit_rms,
        )
