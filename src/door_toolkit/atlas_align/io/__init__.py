"""I/O helpers for the atlas_align subpackage.

Modules:

* :mod:`~.atlas_loader` — read the builder's output directory into an
  :class:`AtlasBundle`.
* :mod:`~.roi_loader` — parse FIJI RoiManager ``.zip`` into Python
  dataclasses.
* :mod:`~.roi_exporter` — write assignments back out to a FIJI
  ``.zip``.
* :mod:`~.pose_io` — serialise/deserialise 10-DOF GUI poses.
"""

from __future__ import annotations

from door_toolkit.atlas_align.io.atlas_loader import (
    AtlasBundle,
    load_atlas_bundle,
)

__all__ = ["AtlasBundle", "load_atlas_bundle"]
