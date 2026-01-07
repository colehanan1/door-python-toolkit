"""Unit tests for connectome-aware post-hoc analysis utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from door_toolkit.pathways.connectome_analysis import (
    align_orn_connectome,
    compute_connectome_influence,
    orient_connectome,
)


def test_orient_connectome_transposes_to_pn_by_orn_and_kc_by_pn():
    n_orn, n_pn, n_kc = 3, 4, 2
    # Stored as ORN×PN and PN×KC (common export convention).
    A_raw = torch.tensor(
        [
            [1.0, 10.0, 100.0, 1000.0],
            [2.0, 20.0, 200.0, 2000.0],
            [3.0, 30.0, 300.0, 3000.0],
        ]
    )  # (3, 4)
    B_raw = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ]
    )  # (4, 2)

    A_pn_by_orn, B_kc_by_pn, report = orient_connectome(A_raw, B_raw, n_orn=n_orn)
    assert tuple(A_pn_by_orn.shape) == (n_pn, n_orn)
    assert tuple(B_kc_by_pn.shape) == (n_kc, n_pn)
    assert report["chosen"]["A_oriented"] == "transposed"
    assert report["chosen"]["B_oriented"] == "transposed"

    # Propagation sanity: with s_orn=[1,0,0], s_pn is the first ORN column.
    s_orn = [1.0, 0.0, 0.0]
    out = compute_connectome_influence(
        s_orn,
        A_pn_by_orn,
        B_kc_by_pn,
        receptor_names=["r0", "r1", "r2"],
        pn_ids=[f"pn{i}" for i in range(n_pn)],
        kc_ids=[f"kc{i}" for i in range(n_kc)],
        top_pn=10,
        top_kc=10,
    )
    assert out.s_pn.tolist() == [1.0, 10.0, 100.0, 1000.0]
    assert out.s_kc.tolist() == [1111.0, 0.0]


def test_orient_connectome_noop_when_already_oriented():
    n_orn, n_pn, n_kc = 3, 4, 2
    A_pn_by_orn = torch.ones(n_pn, n_orn)
    B_kc_by_pn = torch.ones(n_kc, n_pn)
    A2, B2, report = orient_connectome(A_pn_by_orn, B_kc_by_pn, n_orn=n_orn)
    assert tuple(A2.shape) == (n_pn, n_orn)
    assert tuple(B2.shape) == (n_kc, n_pn)
    assert report["chosen"]["A_oriented"] == "as_is"
    assert report["chosen"]["B_oriented"] == "as_is"


def test_align_orn_connectome_reorders_columns():
    # A is (PN, ORN_total)
    A = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])  # (2, 3)
    connectome_receptors = ["rA", "rB", "rC"]
    target_receptors = ["rC", "rA"]
    A_aligned, rep = align_orn_connectome(A, connectome_receptors, target_receptors)
    assert tuple(A_aligned.shape) == (2, 2)
    assert A_aligned.tolist() == [[3.0, 1.0], [30.0, 10.0]]
    assert rep["n_model_receptors"] == 2


def test_align_orn_connectome_fails_fast_on_missing_receptor():
    A = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="missing model receptors"):
        align_orn_connectome(A, ["rA", "rB"], ["rA", "rC"])


def test_align_orn_connectome_matches_training_receptor_set_indices():
    training_set = json.loads(Path("data/mappings/training_receptor_set.json").read_text(encoding="utf-8"))
    connectome_order = training_set["connectivity_receptor_order"]
    expected_indices = training_set["receptor_indices_in_connectivity"]
    target_receptors = training_set["receptors"]

    assert len(target_receptors) == 55
    assert len(expected_indices) == 55

    A_dummy = torch.zeros(1, len(connectome_order))
    _, rep = align_orn_connectome(A_dummy, connectome_order, target_receptors)
    assert rep["selected_indices"] == expected_indices


def test_compute_connectome_influence_deterministic_and_sanity_ranking():
    # ORN0 has lower base importance but huge fanout; amplified should rank ORN0 top.
    A = torch.tensor([[10.0, 1.0, 1.0], [10.0, 1.0, 1.0]])  # (PN=2, ORN=3)
    B = torch.ones(2, 2)  # (KC=2, PN=2)
    base = [1.0, 2.0, 2.0]
    receptors = ["r0", "r1", "r2"]

    out1 = compute_connectome_influence(base, A, B, receptor_names=receptors, top_pn=10, top_kc=10)
    out2 = compute_connectome_influence(base, A, B, receptor_names=receptors, top_pn=10, top_kc=10)

    pd.testing.assert_frame_equal(out1.orn_table, out2.orn_table, check_exact=True)
    pd.testing.assert_frame_equal(out1.pn_table, out2.pn_table, check_exact=True)
    pd.testing.assert_frame_equal(out1.kc_table, out2.kc_table, check_exact=True)
    assert out1.summary == out2.summary

    assert out1.orn_table.iloc[0]["receptor"] == "r0"
    assert out1.summary["downstream_fraction_top_5_orns"] == 1.0


def test_compute_connectome_influence_fails_on_shape_mismatch():
    A = torch.ones(2, 3)  # (PN=2, ORN=3)
    B = torch.ones(2, 3)  # (KC=2, PN=? mismatch: expects 2)
    with pytest.raises(ValueError, match="PN dimensions do not match"):
        compute_connectome_influence([1.0, 2.0, 3.0], A, B, receptor_names=["r0", "r1", "r2"])
