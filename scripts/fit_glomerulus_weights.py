#!/usr/bin/env python
"""
Fit a baseline glomerulus weight vector from control/baseline PER labels.

Usage:
    python scripts/fit_glomerulus_weights.py \
        --config configs/glomerulus_weight_baseline.yaml \
        --feature-set union \
        --outdir out/glomerulus_baseline_union

This script is a thin wrapper around door_toolkit.cli_glomerulus.glomerulus_main().
"""

import sys
from pathlib import Path

# Ensure src/ is importable when running as a standalone script
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from door_toolkit.cli_glomerulus import glomerulus_main

if __name__ == "__main__":
    glomerulus_main()
