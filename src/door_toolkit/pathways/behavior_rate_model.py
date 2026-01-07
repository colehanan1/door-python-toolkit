"""
Dataset-level PER rate model (Option 1)
======================================

This module implements a sparse generalized linear model (GLM) that predicts
aggregate proboscis extension response (PER) rates for each (dataset, test odor)
cell in a behavioral matrix.

Key definitions
---------------
- A "dataset" is a training condition (e.g., ``opto_hex``, ``hex_control``).
- Each dataset has a trained odor (the odor paired with reward for opto_... datasets).
- Each cell in the behavioral CSV is the PER rate after training when a test odor
  is presented (fractional label in ``[0, 1]``).

Decision → Evidence → Implementation
-----------------------------------
- BCE on fractional labels:
  Decision: use ``binary_cross_entropy_with_logits`` with ``y ∈ [0,1]``.
  Evidence: PER rate is a mean of Bernoulli outcomes; BCE is the negative log-likelihood
  under a Bernoulli model for fractional/soft targets (common in distillation).
  Implementation: train loss = BCE(logits, y), with logits passed through sigmoid only
  for reporting/predictions.

- L1 regularization:
  Decision: apply L1 penalty to ORN weights to encourage sparsity.
  Evidence: sparse solutions improve interpretability and stability for ORN importance.
  Implementation: ``λ * (|w_test|_1 + |w_train|_1 + |w_int|_1 + |w_diff|_1)``.

- Ablation-based importance:
  Decision: prioritize ablation-based importance for experiment planning.
  Evidence: weight magnitude can be misleading with correlated features; ablation measures
  loss sensitivity in the model's operating regime.
  Implementation: for each ORN channel i, set x_test[:,i]=x_train[:,i]=0 and recompute BCE;
  importance is ``ΔBCE = BCE_ablated - BCE_baseline``.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

from door_toolkit.encoder import DoOREncoder


def _clean_token(text: str) -> str:
    """Normalize a label for mapping lookups (alphanumeric only, lowercase)."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def default_behavior_odorant_candidates() -> Dict[str, Optional[List[str]]]:
    """
    Default mapping from behavioral CSV odor labels → DoOR odor name candidates.

    Notes:
    - Keys are normalized by ``_clean_token``.
    - Values are candidate DoOR names (case-insensitive) to try in order.
    - ``None`` indicates a control stimulus that should be encoded as a zero vector.
    """
    return {
        "3octonol": ["3-octanol", "oct-3-ol", "octan-3-ol"],
        "air": None,  # control stimulus, not in DoOR
        "applecidervinegar": ["acetic acid", "ethanoic acid"],
        "benzaldehyde": ["benzaldehyde"],
        "citral": ["citral", "geranial"],
        "ethylbutyrate": ["ethyl butyrate", "ethyl butanoate"],
        # Behavioral naming artifact: same chemical, different training regimen label.
        "ethylbutyrate6training": ["ethyl butyrate", "ethyl butanoate"],
        "hexanol": ["1-hexanol", "hexan-1-ol"],
        "linalool": ["linalool"],
    }


class OdorResolutionLogger:
    """
    Track all odor name resolution decisions for reproducibility and debugging.

    Each resolution is logged with the method used, candidates tried, and final result.
    """

    def __init__(self) -> None:
        self.resolutions: Dict[str, Dict] = {}

    def log_resolution(
        self,
        csv_name: str,
        door_name: str,
        method: str,
        candidates_tried: Optional[List[str]] = None,
        matched_candidate: Optional[str] = None,
    ) -> None:
        """Log a successful odor name resolution."""
        self.resolutions[csv_name] = {
            "csv_name": csv_name,
            "door_name": door_name,
            "status": "RESOLVED",
            "method": method,  # "candidate_mapping", "exact", "variant", "fuzzy"
            "candidates_tried": candidates_tried or [],
            "matched_candidate": matched_candidate,
        }

    def log_not_found(
        self,
        csv_name: str,
        reason: str,
        closest_matches: Optional[List[str]] = None,
    ) -> None:
        """Log a failed odor name resolution."""
        self.resolutions[csv_name] = {
            "csv_name": csv_name,
            "door_name": None,
            "status": "NOT_FOUND",
            "reason": reason,
            "closest_matches": closest_matches or [],
            "encoded_as": "zero_vector",
        }

    def get_door_name(self, csv_name: str) -> Optional[str]:
        """Get the resolved DoOR name for a CSV odor label."""
        if csv_name in self.resolutions:
            return self.resolutions[csv_name].get("door_name")
        return None

    def to_json(self) -> Dict[str, Dict]:
        """Export resolution log as JSON-serializable dict."""
        return self.resolutions


def resolve_door_odor_name(
    csv_odorant_name: str,
    encoder: DoOREncoder,
    odorant_candidates: Optional[Mapping[str, Optional[Sequence[str]]]] = None,
    resolution_logger: Optional[OdorResolutionLogger] = None,
) -> Optional[str]:
    """
    Resolve a behavioral CSV odor label to a DoOR odor name.

    Args:
        csv_odorant_name: Odor name from behavioral CSV
        encoder: DoOR encoder instance
        odorant_candidates: Optional mapping of normalized names → candidate DoOR names
        resolution_logger: Optional logger to track resolution decisions

    Returns:
        DoOR odor name if available, else None for control stimuli (encoded as zeros).
    """
    candidates_map = (
        dict(odorant_candidates) if odorant_candidates is not None else default_behavior_odorant_candidates()
    )

    key = _clean_token(csv_odorant_name)
    if key in candidates_map:
        candidates = candidates_map[key]
        if candidates is None:
            # Control stimulus (e.g., air, paraffin oil)
            if resolution_logger is not None:
                resolution_logger.log_resolution(
                    csv_name=csv_odorant_name,
                    door_name=None,
                    method="control_stimulus",
                    candidates_tried=[],
                )
            return None
        for candidate in candidates:
            try:
                _ = encoder.encode(candidate, fill_missing=0.0)
                if resolution_logger is not None:
                    resolution_logger.log_resolution(
                        csv_name=csv_odorant_name,
                        door_name=candidate,
                        method="candidate_mapping",
                        candidates_tried=list(candidates),
                        matched_candidate=candidate,
                    )
                return candidate
            except KeyError:
                continue

    # Try light normalization variants before fuzzy matching.
    variants = [
        str(csv_odorant_name),
        str(csv_odorant_name).replace("_", " "),
        str(csv_odorant_name).replace("-", " "),
        str(csv_odorant_name).replace("_", " ").replace("-", " "),
    ]
    for v in variants:
        try:
            _ = encoder.encode(v, fill_missing=0.0)
            if resolution_logger is not None:
                resolution_logger.log_resolution(
                    csv_name=csv_odorant_name,
                    door_name=v,
                    method="variant_normalization",
                    candidates_tried=variants,
                    matched_candidate=v,
                )
            return v
        except KeyError:
            continue

    # Fallback: fuzzy match against DoOR odorant names
    import difflib

    query = str(csv_odorant_name).lower().replace("_", " ").replace("-", " ").strip()
    matches = difflib.get_close_matches(query, encoder.odorant_names, n=1, cutoff=0.6)
    if matches:
        if resolution_logger is not None:
            resolution_logger.log_resolution(
                csv_name=csv_odorant_name,
                door_name=matches[0],
                method="fuzzy_match",
                candidates_tried=difflib.get_close_matches(query, encoder.odorant_names, n=5, cutoff=0.4),
                matched_candidate=matches[0],
            )
        return matches[0]

    # Not found
    if resolution_logger is not None:
        closest = difflib.get_close_matches(query, encoder.odorant_names, n=3, cutoff=0.3)
        resolution_logger.log_not_found(
            csv_name=csv_odorant_name,
            reason="no_candidates_matched",
            closest_matches=closest,
        )
    return None


def default_dataset_trained_odor_mapping() -> Dict[str, str]:
    """
    Default condition → trained odor mapping for common optogenetic datasets.

    Values are behavioral CSV odor labels (not necessarily DoOR-canonical names).
    """
    return {
        # Optogenetic conditions
        "opto_hex": "Hexanol",
        "opto_EB": "Ethyl_Butyrate",
        "opto_EB_6_training": "Ethyl_Butyrate",
        "opto_benz_1": "Benzaldehyde",
        "opto_ACV": "Apple_Cider_Vinegar",
        "opto_3-oct": "3-Octonol",
        "opto_3oct": "3-Octonol",
        "opto_AIR": "AIR",
        # Matched controls (if present)
        "hex_control": "Hexanol",
        "EB_control": "Ethyl_Butyrate",
        "Benz_control": "Benzaldehyde",
    }


def resolve_trained_odor_for_dataset(
    dataset_name: str, mapping: Optional[Mapping[str, str]] = None
) -> str:
    """
    Resolve trained odor label for a dataset/condition.

    Decision → Evidence → Implementation:
    - Decision: prefer explicit mapping, fallback to deterministic parsing.
    - Evidence: condition strings are not standardized (e.g., ``opto_benz_1``).
    - Implementation: lookup in mapping first; else parse common tokens.
    """
    mapping_dict = dict(mapping) if mapping is not None else default_dataset_trained_odor_mapping()
    if dataset_name in mapping_dict:
        return mapping_dict[dataset_name]

    name = str(dataset_name)
    if name.startswith("opto_"):
        suffix = name[len("opto_") :].lower()
        suffix_clean = _clean_token(suffix)
        if suffix_clean in {"hex", "hexanol"}:
            return "Hexanol"
        if suffix_clean in {"eb", "ethylbutyrate"}:
            return "Ethyl_Butyrate"
        if suffix_clean.startswith("benz"):
            return "Benzaldehyde"
        if suffix_clean in {"acv", "applecidervinegar"}:
            return "Apple_Cider_Vinegar"
        if "3oct" in suffix_clean or "3octonol" in suffix_clean:
            return "3-Octonol"
        if suffix_clean == "air":
            return "AIR"

    if name.endswith("_control"):
        base = name[: -len("_control")].lower()
        if base in {"hex", "hexanol"}:
            return "Hexanol"
        if base in {"eb", "ethylbutyrate"}:
            return "Ethyl_Butyrate"
        if base.startswith("benz"):
            return "Benzaldehyde"

    raise ValueError(
        f"Could not resolve trained odor for dataset '{dataset_name}'. "
        "Provide an explicit mapping via the CLI/config."
    )


@dataclass(frozen=True)
class TrainingTable:
    dataset: List[str]
    test_odor: List[str]
    y: torch.Tensor  # shape (N,)
    x_test: torch.Tensor  # shape (N, C)
    x_train: torch.Tensor  # shape (N, C)
    reward_flag: torch.Tensor  # shape (N,)
    receptor_names: List[str]


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 0
    epochs: int = 500
    lr: float = 1e-2
    l1_lambda: float = 1e-3
    include_test: bool = True
    include_train: bool = True
    include_interaction: bool = True
    include_diff: bool = False
    include_reward: bool = True


class SparseRateGLM(nn.Module):
    """
    Sparse GLM for PER rates with interpretable ORN weights.

    logit(p) = b + b_reward * reward_flag
               + w_test·x_test + w_train·x_train
               + w_int·(x_test ⊙ x_train) + w_diff·(x_test - x_train)
    """

    def __init__(
        self,
        n_channels: int,
        include_test: bool = True,
        include_train: bool = True,
        include_interaction: bool = True,
        include_diff: bool = False,
        include_reward: bool = True,
    ) -> None:
        super().__init__()
        self.n_channels = int(n_channels)
        self.include_test = bool(include_test)
        self.include_train = bool(include_train)
        self.include_interaction = bool(include_interaction)
        self.include_diff = bool(include_diff)
        self.include_reward = bool(include_reward)

        self.intercept = nn.Parameter(torch.zeros(()))
        self.reward_weight = nn.Parameter(torch.zeros(())) if self.include_reward else None

        self.w_test = nn.Parameter(torch.zeros(self.n_channels)) if self.include_test else None
        self.w_train = nn.Parameter(torch.zeros(self.n_channels)) if self.include_train else None
        self.w_int = (
            nn.Parameter(torch.zeros(self.n_channels)) if self.include_interaction else None
        )
        self.w_diff = nn.Parameter(torch.zeros(self.n_channels)) if self.include_diff else None

    def forward(self, x_test: torch.Tensor, x_train: torch.Tensor, reward_flag: torch.Tensor) -> torch.Tensor:
        logits = self.intercept
        if self.include_reward and self.reward_weight is not None:
            logits = logits + self.reward_weight * reward_flag
        if self.include_test and self.w_test is not None:
            logits = logits + (x_test * self.w_test).sum(dim=1)
        if self.include_train and self.w_train is not None:
            logits = logits + (x_train * self.w_train).sum(dim=1)
        if self.include_interaction and self.w_int is not None:
            logits = logits + ((x_test * x_train) * self.w_int).sum(dim=1)
        if self.include_diff and self.w_diff is not None:
            logits = logits + ((x_test - x_train) * self.w_diff).sum(dim=1)
        return logits

    def l1_penalty(self) -> torch.Tensor:
        penalty = torch.zeros(())
        for w in (self.w_test, self.w_train, self.w_int, self.w_diff):
            if w is not None:
                penalty = penalty + w.abs().sum()
        return penalty


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_training_table(
    behavior_df: pd.DataFrame,
    encoder: DoOREncoder,
    *,
    include_datasets: Optional[Sequence[str]] = None,
    exclude_datasets: Optional[Sequence[str]] = None,
    odorant_candidates: Optional[Mapping[str, Optional[Sequence[str]]]] = None,
    dataset_trained_odor_map: Optional[Mapping[str, str]] = None,
    resolution_logger: Optional[OdorResolutionLogger] = None,
    receptor_mask: Optional[np.ndarray] = None,
) -> TrainingTable:
    """
    Convert a behavioral matrix into a row-wise training table of observed cells.

    Args:
        behavior_df: Behavioral matrix (datasets × odors)
        encoder: DoOR encoder instance
        include_datasets: If provided, only include these datasets
        exclude_datasets: Datasets to exclude
        odorant_candidates: Mapping of odor names to DoOR candidates
        dataset_trained_odor_map: Mapping of dataset names to trained odor labels
        resolution_logger: Optional logger to track odor name resolutions
        receptor_mask: Optional boolean mask for adult-only receptors

    Returns:
        TrainingTable with observed cells and features
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for behavior rate model training (install extras: torch).")

    df = behavior_df.copy()
    df.index = df.index.astype(str)
    df.columns = [str(c) for c in df.columns]

    if include_datasets is not None:
        include_set = set(include_datasets)
        df = df.loc[[d for d in df.index if d in include_set]]
    if exclude_datasets is not None:
        exclude_set = set(exclude_datasets)
        df = df.loc[[d for d in df.index if d not in exclude_set]]

    # Determine effective number of channels after masking
    n_effective_channels = np.sum(receptor_mask) if receptor_mask is not None else encoder.n_channels

    # Cache odor encodings (DoOR name or None → vector)
    odor_to_vec: Dict[Optional[str], np.ndarray] = {}

    def odor_vector(csv_odor: str) -> np.ndarray:
        door_name = resolve_door_odor_name(
            csv_odor,
            encoder,
            odorant_candidates=odorant_candidates,
            resolution_logger=resolution_logger,
        )
        if door_name not in odor_to_vec:
            if door_name is None:
                odor_to_vec[door_name] = np.zeros(n_effective_channels, dtype=np.float32)
            else:
                vec = encoder.encode(door_name, fill_missing=0.0)
                if not isinstance(vec, np.ndarray):
                    vec = np.array(vec)
                vec = vec.astype(np.float32)
                # Apply receptor mask if provided
                if receptor_mask is not None:
                    vec = vec[receptor_mask]
                odor_to_vec[door_name] = vec
        return odor_to_vec[door_name]

    dataset_list: List[str] = []
    test_odor_list: List[str] = []
    y_list: List[float] = []
    x_test_list: List[np.ndarray] = []
    x_train_list: List[np.ndarray] = []
    reward_list: List[float] = []

    for dataset_name in df.index:
        trained_csv = resolve_trained_odor_for_dataset(dataset_name, mapping=dataset_trained_odor_map)
        x_train_vec = odor_vector(trained_csv)
        reward_flag = 1.0 if str(dataset_name).startswith("opto_") else 0.0

        row = df.loc[dataset_name]
        for test_odor in df.columns:
            y = row[test_odor]
            if pd.isna(y):
                continue
            dataset_list.append(dataset_name)
            test_odor_list.append(test_odor)
            y_list.append(float(y))
            x_test_list.append(odor_vector(test_odor))
            x_train_list.append(x_train_vec)
            reward_list.append(reward_flag)

    if len(y_list) == 0:
        raise ValueError("No observed (non-NaN) behavioral cells to train on.")

    x_test_arr = np.stack(x_test_list, axis=0)
    x_train_arr = np.stack(x_train_list, axis=0)

    y_tensor = torch.tensor(y_list, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test_arr, dtype=torch.float32)
    x_train_tensor = torch.tensor(x_train_arr, dtype=torch.float32)
    reward_tensor = torch.tensor(reward_list, dtype=torch.float32)

    # Determine receptor names (masked if applicable)
    if receptor_mask is not None:
        receptor_names_masked = [r for r, keep in zip(encoder.receptor_names, receptor_mask) if keep]
    else:
        receptor_names_masked = list(encoder.receptor_names)

    return TrainingTable(
        dataset=dataset_list,
        test_odor=test_odor_list,
        y=y_tensor,
        x_test=x_test_tensor,
        x_train=x_train_tensor,
        reward_flag=reward_tensor,
        receptor_names=receptor_names_masked,
    )


def build_adult_receptor_mask(
    encoder: DoOREncoder,
    training_receptor_set_path: str | Path,
) -> Tuple[np.ndarray, Dict]:
    """
    Build boolean mask for adult-only ORNs and generate manifest.

    Args:
        encoder: DoOR encoder instance
        training_receptor_set_path: Path to training_receptor_set.json

    Returns:
        Tuple of (mask array, manifest dict) where:
        - mask: boolean array of shape (n_channels,) indicating which receptors to keep
        - manifest: dict with kept/excluded receptors, counts, and provenance
    """
    # Load training receptor set
    with open(training_receptor_set_path, 'r', encoding='utf-8') as f:
        receptor_set = json.load(f)

    adult_receptors = set(receptor_set.get("receptors", []))
    n_adult = len(adult_receptors)

    # Build mask
    mask = np.array([r in adult_receptors for r in encoder.receptor_names], dtype=bool)

    included = [r for r, keep in zip(encoder.receptor_names, mask) if keep]
    excluded = [r for r, keep in zip(encoder.receptor_names, mask) if not keep]

    manifest = {
        "n_total_receptors": len(encoder.receptor_names),
        "n_adult_only": n_adult,
        "n_excluded": len(excluded),
        "included_receptors": included,
        "excluded_receptors": excluded,
        "excluded_reason": "larval_or_unmapped",
        "source_file": str(training_receptor_set_path),
        "encoder_receptor_order": list(encoder.receptor_names),
    }

    return mask, manifest


@dataclass(frozen=True)
class FitMetrics:
    bce: float
    total_loss: float
    r2: Optional[float]
    pearson_r: Optional[float]


def _safe_pearsonr(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    if np.allclose(np.std(y_true), 0.0) or np.allclose(np.std(y_pred), 0.0):
        return None
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> Optional[float]:
    if len(y_true) < 2:
        return None
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if math.isclose(denom, 0.0):
        return None
    num = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - (num / denom)


def train_sparse_rate_glm(
    table: TrainingTable,
    config: TrainConfig,
) -> Tuple[SparseRateGLM, List[float]]:
    """
    Train the sparse rate GLM with full-batch Adam.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for training (install extras: torch).")

    set_reproducible_seed(config.seed)

    include_reward = bool(config.include_reward)
    # If reward_flag is constant, disable the reward term for identifiability.
    if include_reward:
        unique = torch.unique(table.reward_flag)
        if unique.numel() == 1:
            include_reward = False

    model = SparseRateGLM(
        n_channels=table.x_test.shape[1],
        include_test=config.include_test,
        include_train=config.include_train,
        include_interaction=config.include_interaction,
        include_diff=config.include_diff,
        include_reward=include_reward,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.lr))

    losses: List[float] = []
    for _ in range(int(config.epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(table.x_test, table.x_train, table.reward_flag)
        bce = F.binary_cross_entropy_with_logits(logits, table.y)
        l1 = model.l1_penalty()
        loss = bce + float(config.l1_lambda) * l1
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return model, losses


if TORCH_AVAILABLE:

    @torch.no_grad()
    def predict_rates(
        model: SparseRateGLM,
        x_test: torch.Tensor,
        x_train: torch.Tensor,
        reward_flag: torch.Tensor,
    ) -> torch.Tensor:
        logits = model(x_test, x_train, reward_flag)
        return torch.sigmoid(logits)

    @torch.no_grad()
    def evaluate_fit(
        model: SparseRateGLM,
        table: TrainingTable,
        l1_lambda: float = 0.0,
    ) -> FitMetrics:
        logits = model(table.x_test, table.x_train, table.reward_flag)
        bce = float(F.binary_cross_entropy_with_logits(logits, table.y).cpu().item())
        total = bce + float(l1_lambda) * float(model.l1_penalty().cpu().item())
        y_true = table.y.detach().cpu().numpy()
        y_pred = torch.sigmoid(logits).detach().cpu().numpy()
        return FitMetrics(
            bce=bce,
            total_loss=total,
            r2=_safe_r2(y_true, y_pred),
            pearson_r=_safe_pearsonr(y_true, y_pred),
        )

else:  # pragma: no cover

    def predict_rates(*args, **kwargs):
        raise ImportError("PyTorch is required (install extras: torch).")

    def evaluate_fit(*args, **kwargs):
        raise ImportError("PyTorch is required (install extras: torch).")


def weight_importance_dataframe(
    model: SparseRateGLM, receptor_names: Sequence[str]
) -> pd.DataFrame:
    """
    Weight-based ORN importance.

    Definition: importance_i = |w_test[i]| + |w_train[i]| + |w_int[i]| + |w_diff[i]|
    where absent blocks contribute 0.
    """
    def _to_np(param: Optional[torch.Tensor]) -> np.ndarray:
        if param is None:
            return np.zeros(len(receptor_names), dtype=np.float64)
        return param.detach().cpu().numpy().astype(np.float64)

    w_test = _to_np(model.w_test)
    w_train = _to_np(model.w_train)
    w_int = _to_np(model.w_int)
    w_diff = _to_np(model.w_diff)
    importance = np.abs(w_test) + np.abs(w_train) + np.abs(w_int) + np.abs(w_diff)

    df = pd.DataFrame(
        {
            "receptor": list(receptor_names),
            "importance_weight": importance,
            "w_test": w_test,
            "w_train": w_train,
            "w_int": w_int,
            "w_diff": w_diff,
        }
    )
    df = df.sort_values("importance_weight", ascending=False).reset_index(drop=True)
    return df


def ablation_importance(
    model: SparseRateGLM, table: TrainingTable
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Ablation-based ORN importance on the observed training table.

    Returns:
        (global_df, by_dataset_df, by_test_odor_df)
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required (install extras: torch).")

    with torch.no_grad():
        logits0 = model(table.x_test, table.x_train, table.reward_flag)
        loss0 = F.binary_cross_entropy_with_logits(logits0, table.y, reduction="none")

    n_channels = table.x_test.shape[1]
    receptors = table.receptor_names

    global_rows: List[Dict[str, float | str]] = []
    by_dataset_rows: List[Dict[str, float | str]] = []
    by_odor_rows: List[Dict[str, float | str]] = []

    # Precompute grouping keys
    ds = np.array(table.dataset, dtype=object)
    od = np.array(table.test_odor, dtype=object)

    for i in range(n_channels):
        x_test_ab = table.x_test.clone()
        x_train_ab = table.x_train.clone()
        x_test_ab[:, i] = 0.0
        x_train_ab[:, i] = 0.0

        with torch.no_grad():
            logits_ab = model(x_test_ab, x_train_ab, table.reward_flag)
            loss_ab = F.binary_cross_entropy_with_logits(logits_ab, table.y, reduction="none")
            delta = (loss_ab - loss0).detach().cpu().numpy()

        global_rows.append(
            {
                "receptor": receptors[i],
                "delta_bce": float(np.mean(delta)),
            }
        )

        # By dataset
        for dataset_name in sorted(set(ds.tolist())):
            mask = ds == dataset_name
            if not np.any(mask):
                continue
            by_dataset_rows.append(
                {
                    "dataset": dataset_name,
                    "receptor": receptors[i],
                    "delta_bce": float(np.mean(delta[mask])),
                }
            )

        # By test odor
        for odor_name in sorted(set(od.tolist())):
            mask = od == odor_name
            if not np.any(mask):
                continue
            by_odor_rows.append(
                {
                    "test_odor": odor_name,
                    "receptor": receptors[i],
                    "delta_bce": float(np.mean(delta[mask])),
                }
            )

    global_df = pd.DataFrame(global_rows).sort_values("delta_bce", ascending=False).reset_index(drop=True)
    by_dataset_df = pd.DataFrame(by_dataset_rows)
    by_odor_df = pd.DataFrame(by_odor_rows)
    return global_df, by_dataset_df, by_odor_df


def build_prediction_long_format(
    behavior_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    dataset_trained_map: Mapping[str, str],
    resolution_logger: OdorResolutionLogger,
) -> pd.DataFrame:
    """
    Convert wide-format predictions to long format with metadata.

    Args:
        behavior_df: Original behavioral matrix (datasets × odors)
        pred_df: Predicted rates matrix (datasets × odors)
        dataset_trained_map: Mapping of dataset names → trained odor labels
        resolution_logger: Odor resolution logger with mappings

    Returns:
        Long-format DataFrame with columns:
        [dataset, test_odor, y_true, y_pred, split, trained_odor, trained_odor_door, test_odor_door]
    """
    rows = []
    for dataset in behavior_df.index:
        for test_odor in behavior_df.columns:
            y_true = behavior_df.loc[dataset, test_odor]
            y_pred = pred_df.loc[dataset, test_odor]

            # Determine split: observed if y_true is non-NaN
            split = "observed" if pd.notna(y_true) else "extrapolated"

            # Get trained odor info
            trained_odor_csv = dataset_trained_map.get(str(dataset), "UNKNOWN")
            trained_odor_door = resolution_logger.get_door_name(trained_odor_csv)

            # Get test odor DoOR name
            test_odor_door = resolution_logger.get_door_name(str(test_odor))

            rows.append({
                "dataset": str(dataset),
                "test_odor": str(test_odor),
                "y_true": y_true if pd.notna(y_true) else np.nan,
                "y_pred": float(y_pred),
                "split": split,
                "trained_odor": trained_odor_csv,
                "trained_odor_door": trained_odor_door if trained_odor_door is not None else "UNKNOWN",
                "test_odor_door": test_odor_door if test_odor_door is not None else "UNKNOWN",
            })

    return pd.DataFrame(rows)


def split_observed_extrapolated(predictions_long: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split long-format predictions into observed and extrapolated DataFrames.

    Args:
        predictions_long: Long-format predictions with 'split' column

    Returns:
        Tuple of (observed_df, extrapolated_df)
    """
    observed = predictions_long[predictions_long['split'] == 'observed'].copy().reset_index(drop=True)
    extrapolated = predictions_long[predictions_long['split'] == 'extrapolated'].copy().reset_index(drop=True)
    return observed, extrapolated


def compute_per_dataset_metrics(
    predictions_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fit metrics broken down by dataset.

    Args:
        predictions_long: Long-format predictions (observed cells only)

    Returns:
        DataFrame with columns: [dataset, n_cells, bce, r2, pearson_r, mae, trained_odor]
    """
    rows = []
    for dataset, group in predictions_long.groupby('dataset'):
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values

        # Remove NaN (should not be present in observed cells, but defensive)
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            continue

        # BCE
        eps = 1e-7
        bce = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

        # R²
        r2 = _safe_r2(y_true, y_pred)

        # Pearson r
        pearson_r = _safe_pearsonr(y_true, y_pred)

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # Trained odor
        trained_odor = group['trained_odor'].iloc[0] if 'trained_odor' in group.columns else "UNKNOWN"

        rows.append({
            "dataset": str(dataset),
            "n_cells": len(y_true),
            "bce": float(bce),
            "r2": float(r2) if r2 is not None else None,
            "pearson_r": float(pearson_r) if pearson_r is not None else None,
            "mae": float(mae),
            "trained_odor": trained_odor,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("bce", ascending=False).reset_index(drop=True)
    return df


def compute_per_test_odor_metrics(
    predictions_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute fit metrics broken down by test odor.

    Args:
        predictions_long: Long-format predictions (observed cells only)

    Returns:
        DataFrame with columns: [test_odor, n_cells, bce, r2, pearson_r, mae, door_name]
    """
    rows = []
    for test_odor, group in predictions_long.groupby('test_odor'):
        y_true = group['y_true'].values
        y_pred = group['y_pred'].values

        # Remove NaN
        mask = ~np.isnan(y_true)
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        if len(y_true) == 0:
            continue

        # BCE
        eps = 1e-7
        bce = -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

        # R²
        r2 = _safe_r2(y_true, y_pred)

        # Pearson r
        pearson_r = _safe_pearsonr(y_true, y_pred)

        # MAE
        mae = np.mean(np.abs(y_true - y_pred))

        # Door name
        door_name = group['test_odor_door'].iloc[0] if 'test_odor_door' in group.columns else "UNKNOWN"

        rows.append({
            "test_odor": str(test_odor),
            "n_cells": len(y_true),
            "bce": float(bce),
            "r2": float(r2) if r2 is not None else None,
            "pearson_r": float(pearson_r) if pearson_r is not None else None,
            "mae": float(mae),
            "door_name": door_name,
        })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.sort_values("bce", ascending=False).reset_index(drop=True)
    return df


def compute_learning_effect_metrics(
    predictions_long: pd.DataFrame,
    opto_control_pairs: Mapping[str, str],
) -> pd.DataFrame:
    """
    Compute learning effect (ΔPER = opto - control) for matched pairs.

    Args:
        predictions_long: Long-format predictions
        opto_control_pairs: Dict mapping opto_dataset → control_dataset

    Returns:
        DataFrame with columns: [test_odor, delta_per_true, delta_per_pred, error, opto_dataset, control_dataset]
    """
    rows = []
    for opto_dataset, control_dataset in opto_control_pairs.items():
        # Filter for these two datasets
        opto_data = predictions_long[predictions_long['dataset'] == opto_dataset].copy()
        control_data = predictions_long[predictions_long['dataset'] == control_dataset].copy()

        # Merge on test_odor
        merged = pd.merge(
            opto_data[['test_odor', 'y_true', 'y_pred']],
            control_data[['test_odor', 'y_true', 'y_pred']],
            on='test_odor',
            suffixes=('_opto', '_control'),
        )

        for _, row in merged.iterrows():
            delta_true = row['y_true_opto'] - row['y_true_control']
            delta_pred = row['y_pred_opto'] - row['y_pred_control']
            error = abs(delta_pred - delta_true)

            rows.append({
                "test_odor": row['test_odor'],
                "delta_per_true": float(delta_true) if pd.notna(delta_true) else np.nan,
                "delta_per_pred": float(delta_pred),
                "error": float(error) if pd.notna(error) else np.nan,
                "opto_dataset": opto_dataset,
                "control_dataset": control_dataset,
            })

    return pd.DataFrame(rows)


def load_behavior_matrix_csv(
    path: str | Path,
    validate_rates: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Load behavioral matrix CSV with optional format detection and validation.

    Args:
        path: Path to CSV file
        validate_rates: If True, validate and normalize rate values

    Returns:
        Tuple of (df, format_metadata) where format_metadata contains:
        - format_type: "wide" or "long"
        - rescaled: bool (True if rates were rescaled from [0,100] to [0,1])
        - original_range: (min, max) of original values
        - n_cells: number of non-NaN numeric cells
    """
    df = pd.read_csv(Path(path), index_col=0)

    # Detect format
    format_meta = detect_csv_format(df)

    # If long format, pivot to wide (datasets × odors)
    if format_meta["format_type"] == "long":
        # Assume columns are [dataset/condition, odor, rate/value]
        dataset_col = df.columns[0] if 'dataset' not in df.columns else 'dataset'
        odor_col = df.columns[1] if 'odor' not in df.columns else 'odor'
        rate_col = df.columns[2] if 'rate' not in df.columns else 'rate'
        df = df.pivot(index=dataset_col, columns=odor_col, values=rate_col)

    # Validate and normalize rates
    if validate_rates:
        df, rescale_meta = validate_and_normalize_rates(df, allow_rescale=True)
        format_meta.update(rescale_meta)
    else:
        format_meta.update({"rescaled": False, "original_range": None, "n_cells": 0})

    # Standardize for downstream code
    df.index = df.index.astype(str)
    df.columns = [str(c) for c in df.columns]

    return df, format_meta


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def detect_csv_format(df: pd.DataFrame) -> Dict:
    """
    Auto-detect CSV format (wide vs long) and return metadata.

    Args:
        df: Loaded DataFrame

    Returns:
        Metadata dict with keys: format_type ("wide" or "long"), n_datasets, n_odors
    """
    # Check if it looks like long format (has columns like 'dataset', 'odor', 'rate')
    cols_lower = [str(c).lower() for c in df.columns]
    has_dataset_col = any(c in cols_lower for c in ['dataset', 'condition', 'experiment'])
    has_odor_col = any(c in cols_lower for c in ['odor', 'odorant', 'test_odor'])
    has_rate_col = any(c in cols_lower for c in ['rate', 'per', 'response', 'value'])

    if has_dataset_col and has_odor_col and has_rate_col:
        # Long format detected
        return {
            "format_type": "long",
            "n_datasets": df['dataset'].nunique() if 'dataset' in df.columns else df[df.columns[0]].nunique(),
            "n_odors": df['odor'].nunique() if 'odor' in df.columns else df[df.columns[1]].nunique(),
        }
    else:
        # Assume wide format (datasets × odors)
        return {
            "format_type": "wide",
            "n_datasets": len(df.index),
            "n_odors": len(df.columns),
        }


def validate_and_normalize_rates(df: pd.DataFrame, allow_rescale: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Validate rate values and auto-rescale from [0,100] to [0,1] if needed.

    Args:
        df: DataFrame with numeric rate values
        allow_rescale: If True, auto-rescale [0,100] → [0,1]

    Returns:
        Tuple of (normalized_df, metadata dict)
    """
    # Extract all numeric values (skip NaN)
    numeric_values = df.select_dtypes(include=[np.number]).values.flatten()
    numeric_values = numeric_values[~np.isnan(numeric_values)]

    if len(numeric_values) == 0:
        return df, {"rescaled": False, "original_range": None, "n_cells": 0}

    min_val = float(np.min(numeric_values))
    max_val = float(np.max(numeric_values))

    # Check if in [0, 1]
    if min_val >= 0.0 and max_val <= 1.0:
        return df, {
            "rescaled": False,
            "original_range": (min_val, max_val),
            "n_cells": len(numeric_values),
        }

    # Check if in [0, 100]
    if min_val >= 0.0 and max_val <= 100.0 and allow_rescale:
        df_rescaled = df.copy()
        # Rescale numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df_rescaled[col] = df[col] / 100.0
        return df_rescaled, {
            "rescaled": True,
            "original_range": (min_val, max_val),
            "n_cells": len(numeric_values),
        }

    # Values outside both valid ranges
    raise ValueError(
        f"Rate values outside valid ranges [0,1] and [0,100]. "
        f"Found range: [{min_val:.2f}, {max_val:.2f}]. "
        f"Please check your data."
    )
