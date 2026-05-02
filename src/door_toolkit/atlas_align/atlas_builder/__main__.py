"""Module entry point for ``python -m door_toolkit.atlas_align.atlas_builder``."""

from __future__ import annotations

import sys

from door_toolkit.atlas_align.atlas_builder.build_atlas import main

if __name__ == "__main__":
    sys.exit(main())
