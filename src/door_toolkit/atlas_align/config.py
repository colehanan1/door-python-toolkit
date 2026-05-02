"""
Configuration and logging
=========================

Central place for atlas_align-wide constants and logging setup.

Logging policy (per project spec):

* Default level ``INFO`` on stdout, ``DEBUG`` to a rotating file at
  ``~/.atlas_align/logs/run.log``.
* Uses :class:`rich.logging.RichHandler` when ``rich`` is importable,
  falls back to a plain :class:`logging.StreamHandler` otherwise.
* Every public function in core modules is expected to emit an entry
  log at DEBUG level and an exit/elapsed log at DEBUG level.

Call :func:`configure_logging` exactly once at process start. Subsequent
calls are idempotent (they replace handlers rather than stack them).
"""

from __future__ import annotations

import logging
import logging.handlers
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

#: Root of per-user atlas_align state (logs, cached poses, etc.).
USER_STATE_DIR = Path.home() / ".atlas_align"

#: Rotating log file location.
LOG_FILE = USER_STATE_DIR / "logs" / "run.log"

# ---------------------------------------------------------------------------
# Default values surfaced in both CLI and GUI
# ---------------------------------------------------------------------------

#: Default IoU cutoff below which an ROI is flagged as unassigned.
DEFAULT_IOU_THRESHOLD: float = 0.30

#: Maximum iterations of the greedy conflict-resolution loop in the
#: IoU matcher before we bail out with a warning.
MAX_ASSIGNMENT_ITERATIONS: int = 10

#: Debounce interval (ms) between a pose slider change and the projection
#: recompute. Keeps sliders responsive without thrashing the worker.
PROJECTION_DEBOUNCE_MS: int = 50

#: Soft ceiling on how long a single projection recompute may take before
#: the main window shows a "slow projection" warning in the status bar.
PROJECTION_WARN_MS: int = 400

# ---------------------------------------------------------------------------
# Default pose (identity)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DefaultPose:
    """Identity pose used to seed the GUI and tests.

    Attributes:
        tx, ty, tz: Translation in pixels (applied after scale + rotate).
        rx, ry, rz: Euler-ZYX rotation angles in degrees.
        sx, sy, sz: Per-axis scale (1.0 = identity).
        flip_x, flip_y, flip_z: Mirror flags along each axis.
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


IDENTITY_POSE = DefaultPose()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

_LOG_FORMAT_CONSOLE = "%(message)s"
_LOG_FORMAT_FILE = (
    "%(asctime)s %(levelname)-7s %(name)s:%(lineno)d  %(message)s"
)
_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"

_ROOT_LOGGER_NAME = "door_toolkit.atlas_align"

_configured: bool = False


def _build_console_handler(level: int) -> logging.Handler:
    """Return a RichHandler if possible, else a plain stdlib StreamHandler."""
    try:
        from rich.logging import RichHandler

        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            show_time=True,
            log_time_format="[%X]",
        )
    except ImportError:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(_LOG_FORMAT_FILE, datefmt=_LOG_DATEFMT)
        )
    handler.setLevel(level)
    return handler


def _build_file_handler(log_file: Path) -> logging.Handler:
    """Create a rotating file handler at ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter(_LOG_FORMAT_FILE, datefmt=_LOG_DATEFMT)
    )
    handler.setLevel(logging.DEBUG)
    return handler


def configure_logging(
    verbose: bool = False,
    log_file: Optional[Path] = None,
    reset: bool = False,
) -> logging.Logger:
    """Initialise logging for the atlas_align subpackage.

    Idempotent: on re-entry, existing handlers are replaced.

    Args:
        verbose: If True, console handler streams at DEBUG; otherwise INFO.
        log_file: Override for the rotating-file log path. Defaults to
            ``~/.atlas_align/logs/run.log``.
        reset: Force a fresh configuration even if this function has
            already run in the current process.

    Returns:
        The root ``door_toolkit.atlas_align`` logger.
    """
    global _configured

    console_level = logging.DEBUG if verbose else logging.INFO
    target_log_file = Path(log_file) if log_file is not None else LOG_FILE

    logger = logging.getLogger(_ROOT_LOGGER_NAME)

    if _configured and not reset:
        # Still allow level adjustment on re-entry.
        for handler in logger.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                continue
            handler.setLevel(console_level)
        return logger

    # Wipe any pre-existing handlers so we don't stack duplicates.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    logger.addHandler(_build_console_handler(console_level))
    logger.addHandler(_build_file_handler(target_log_file))

    # Quiet the noisy upstream libraries when operating under our root.
    for noisy in ("urllib3", "matplotlib", "PIL", "PyQt6"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True
    logger.debug(
        "Logging configured: console=%s, file=%s",
        logging.getLevelName(console_level),
        target_log_file,
    )
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a logger under the atlas_align root.

    Use this in every module instead of ``logging.getLogger(__name__)`` to
    guarantee the handlers installed by :func:`configure_logging` apply.

    Args:
        name: Optional dotted suffix; if None, returns the root logger.

    Returns:
        A child of ``door_toolkit.atlas_align``.
    """
    if name is None:
        return logging.getLogger(_ROOT_LOGGER_NAME)
    if name.startswith(_ROOT_LOGGER_NAME):
        return logging.getLogger(name)
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
