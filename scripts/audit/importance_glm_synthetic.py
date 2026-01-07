#!/usr/bin/env python3
"""
Audit: synthetic GLM importance invariants (weight + ablation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from door_toolkit.pathways.behavior_rate_model import (
    SparseRateGLM,
    TrainingTable,
    ablation_importance,
    weight_importance_dataframe,
)


@dataclass(frozen=True)
class SyntheticSpec:
    receptor_names: List[str]
    n_datasets: int
    n_odors: int
    seed: int = 0


def build_synthetic_table(spec: SyntheticSpec) -> TrainingTable:
    rng = np.random.default_rng(spec.seed)
    n_cells = spec.n_datasets * spec.n_odors
    n_channels = len(spec.receptor_names)

    # Deterministic features (x_train unused but required by table schema).
    x_test = rng.normal(0.0, 1.0, size=(n_cells, n_channels)).astype(np.float32)
    x_train = np.zeros_like(x_test)

    datasets = []
    test_odors = []
    reward_flags = []
    for d in range(spec.n_datasets):
        ds_name = "opto_A" if d == 0 else "control_B"
        for o in range(spec.n_odors):
            datasets.append(ds_name)
            test_odors.append(f"odor_{o+1}")
            reward_flags.append(1.0 if ds_name.startswith("opto_") else 0.0)

    # Build labels from a known linear rule (receptor 0 only).
    model = SparseRateGLM(
        n_channels=n_channels,
        include_test=True,
        include_train=False,
        include_interaction=False,
        include_diff=False,
        include_reward=False,
    )
    with torch.no_grad():
        model.w_test[:] = 0.0
        model.w_test[0] = 3.0
        model.intercept.fill_(0.0)

    x_test_t = torch.tensor(x_test, dtype=torch.float32)
    x_train_t = torch.tensor(x_train, dtype=torch.float32)
    reward_t = torch.tensor(reward_flags, dtype=torch.float32)
    with torch.no_grad():
        logits = model(x_test_t, x_train_t, reward_t)
        y = torch.sigmoid(logits).detach().cpu().numpy().astype(np.float32)

    return TrainingTable(
        dataset=datasets,
        test_odor=test_odors,
        y=torch.tensor(y, dtype=torch.float32),
        x_test=x_test_t,
        x_train=x_train_t,
        reward_flag=reward_t,
        receptor_names=list(spec.receptor_names),
    )


def compute_importance(
    table: TrainingTable, weight_vector: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    model = SparseRateGLM(
        n_channels=len(table.receptor_names),
        include_test=True,
        include_train=False,
        include_interaction=False,
        include_diff=False,
        include_reward=False,
    )
    with torch.no_grad():
        model.w_test[:] = torch.tensor(weight_vector, dtype=torch.float32)
        model.intercept.fill_(0.0)

    weight_df = weight_importance_dataframe(model, table.receptor_names)
    ab_global, _, _ = ablation_importance(model, table)
    weight_scores = weight_df.set_index("receptor")["importance_weight"].to_dict()
    ab_scores = ab_global.set_index("receptor")["delta_bce"].to_dict()
    return (
        np.array([weight_scores[r] for r in table.receptor_names], dtype=np.float64),
        np.array([ab_scores[r] for r in table.receptor_names], dtype=np.float64),
    )


def assert_invariants(receptor_names: List[str], weight_scores: np.ndarray, ab_scores: np.ndarray) -> None:
    # Known driver receptor should dominate.
    assert receptor_names[0] == receptor_names[weight_scores.argmax()]
    assert receptor_names[0] == receptor_names[ab_scores.argmax()]

    # Non-driver receptors should have near-zero ablation impact.
    if len(ab_scores) > 1:
        max_other = np.max(ab_scores[1:])
        assert max_other < 1e-6, f"Expected near-zero ablation for non-drivers, got {max_other}"


def main() -> None:
    spec = SyntheticSpec(
        receptor_names=["R1", "R2", "R3", "R4"],
        n_datasets=2,
        n_odors=3,
        seed=7,
    )

    table = build_synthetic_table(spec)
    base_weights = np.zeros(len(spec.receptor_names), dtype=np.float32)
    base_weights[0] = 3.0

    weight_scores, ab_scores = compute_importance(table, base_weights)
    assert_invariants(spec.receptor_names, weight_scores, ab_scores)

    # Permutation invariance (named results should be stable).
    perm = np.array([2, 0, 3, 1], dtype=int)
    perm_names = [spec.receptor_names[i] for i in perm]
    perm_table = TrainingTable(
        dataset=table.dataset,
        test_odor=table.test_odor,
        y=table.y,
        x_test=table.x_test[:, perm],
        x_train=table.x_train[:, perm],
        reward_flag=table.reward_flag,
        receptor_names=perm_names,
    )
    perm_weights = base_weights[perm]
    weight_scores_perm, ab_scores_perm = compute_importance(perm_table, perm_weights)

    # Compare by receptor name.
    base_by_name = dict(zip(spec.receptor_names, weight_scores))
    perm_by_name = dict(zip(perm_names, weight_scores_perm))
    for receptor in spec.receptor_names:
        assert abs(base_by_name[receptor] - perm_by_name[receptor]) < 1e-9

    base_ab_by_name = dict(zip(spec.receptor_names, ab_scores))
    perm_ab_by_name = dict(zip(perm_names, ab_scores_perm))
    for receptor in spec.receptor_names:
        assert abs(base_ab_by_name[receptor] - perm_ab_by_name[receptor]) < 1e-9

    print("OK: GLM importance invariants passed.")


if __name__ == "__main__":
    main()
