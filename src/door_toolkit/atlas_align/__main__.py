"""Entry point for ``python -m door_toolkit.atlas_align``.

Delegates to :func:`door_toolkit.atlas_align.gui.main_window.main`.
"""

from __future__ import annotations

import sys


def main() -> int:
    from door_toolkit.atlas_align.gui.main_window import main as gui_main

    return gui_main()


if __name__ == "__main__":
    sys.exit(main())
