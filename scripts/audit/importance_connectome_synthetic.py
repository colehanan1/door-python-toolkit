#!/usr/bin/env python3
"""
Audit: synthetic connectome amplification invariants.
"""

from __future__ import annotations

import numpy as np
import torch

from door_toolkit.pathways.connectome_analysis import compute_connectome_influence


def main() -> None:
    receptor_names = ["R1", "R2", "R3"]
    s_orn = np.array([1.0, 2.0, 3.0], dtype=np.float64)

    # Case 1: Uniform fanout yields amplification factor mean ~= 1 and amplified == base.
    A = torch.ones((2, 3), dtype=torch.float32)  # PN x ORN
    B = torch.ones((2, 2), dtype=torch.float32)  # KC x PN
    result = compute_connectome_influence(
        s_orn,
        A,
        B,
        receptor_names=receptor_names,
        pn_ids=["PN1", "PN2"],
        kc_ids=["KC1", "KC2"],
        top_pn=2,
        top_kc=2,
    )
    amp_mean = float(result.orn_table["amplification_factor_kc_mean1"].mean())
    assert abs(amp_mean - 1.0) < 1e-9
    assert np.allclose(
        result.orn_table["connectome_amplified_importance"].to_numpy(),
        result.orn_table["base_importance"].to_numpy(),
        atol=1e-9,
    )

    # Case 2: Permute receptor order and verify named outputs are stable.
    perm = np.array([2, 0, 1], dtype=int)
    A_perm = A[:, perm]
    s_perm = s_orn[perm]
    names_perm = [receptor_names[i] for i in perm]
    result_perm = compute_connectome_influence(
        s_perm,
        A_perm,
        B,
        receptor_names=names_perm,
        pn_ids=["PN1", "PN2"],
        kc_ids=["KC1", "KC2"],
        top_pn=2,
        top_kc=2,
    )
    base_map = dict(
        zip(result.orn_table["receptor"], result.orn_table["connectome_amplified_importance"])
    )
    perm_map = dict(
        zip(result_perm.orn_table["receptor"], result_perm.orn_table["connectome_amplified_importance"])
    )
    for receptor in receptor_names:
        assert abs(base_map[receptor] - perm_map[receptor]) < 1e-9

    print("OK: Connectome importance invariants passed.")


if __name__ == "__main__":
    main()
