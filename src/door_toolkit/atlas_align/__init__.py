"""
atlas_align
===========

Interactive Drosophila antennal lobe glomerulus identification.

This subpackage provides:

* A one-shot builder that turns FlyWire projection-neuron dendrite arbors
  into a 3D integer labelmap registered to the JRC2018F template brain
  (:mod:`door_toolkit.atlas_align.atlas_builder`).
* A PyQt6 GUI for interactively posing the 3D atlas to match a 2D imaging
  plane and assigning glomerular identities to user-drawn ROIs via IoU
  matching (:mod:`door_toolkit.atlas_align.gui`).
* Supporting core modules for 10-DOF volume transforms, 3D→2D projection,
  ROI↔glomerulus IoU assignment, and FIJI RoiManager I/O.

See :mod:`door_toolkit.atlas_align.config` for logging setup and default
paths.
"""

from __future__ import annotations

__all__ = ["__version__", "configure_logging"]

__version__ = "0.1.0"

from door_toolkit.atlas_align.config import configure_logging
