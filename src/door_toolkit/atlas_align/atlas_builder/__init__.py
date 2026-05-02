"""Atlas-builder subpackage.

Exposes :func:`main` (CLI entry point) and :func:`build_labelmap`
(library-callable equivalent that returns paths to the generated
outputs).
"""

from __future__ import annotations

from door_toolkit.atlas_align.atlas_builder.build_atlas import (
    build_labelmap,
    main,
)

__all__ = ["main", "build_labelmap"]
