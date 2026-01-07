#!/usr/bin/env python3
"""
Audit: proxy Shapley importance invariants (variance-based).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from door_toolkit.pathways.analyzer import PathwayAnalyzer


def main() -> None:
    # Tiny synthetic behavioral dataset (loaded for audit parity; not used by the proxy method).
    out_dir = Path("outputs/audit/importance_shapley_proxy")
    out_dir.mkdir(parents=True, exist_ok=True)
    behavior_df = pd.DataFrame(
        [[0.1, 0.5, 0.9]], index=["opto_hex"], columns=["Hexanol", "Benzaldehyde", "Linalool"]
    )
    behavior_csv = out_dir / "synthetic_behavior.csv"
    behavior_df.to_csv(behavior_csv)

    analyzer = PathwayAnalyzer("door_cache")
    odorants = analyzer.encoder.odorant_names[:5]
    importance = analyzer.compute_shapley_importance("feeding", odorants=list(odorants))

    if importance:
        total = float(sum(importance.values()))
        assert abs(total - 1.0) < 1e-6
        assert all(v >= 0.0 for v in importance.values())

    # Permutation invariance for odorant list order.
    reversed_importance = analyzer.compute_shapley_importance(
        "feeding", odorants=list(reversed(odorants))
    )
    keys = set(importance.keys()) | set(reversed_importance.keys())
    for key in keys:
        v1 = importance.get(key, 0.0)
        v2 = reversed_importance.get(key, 0.0)
        assert abs(v1 - v2) < 1e-9

    print("OK: Shapley-proxy importance invariants passed.")


if __name__ == "__main__":
    main()
