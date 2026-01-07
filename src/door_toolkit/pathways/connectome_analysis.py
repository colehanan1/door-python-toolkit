"""
Connectome-aware, post-training interpretation for the behavior-rate model.

This module implements a minimal first-order "wiring propagation" analysis that
maps ORN-level model importance to downstream PN/KC influence using fixed
connectivity matrices (ORN→PN, PN→KC).

Decision → Evidence → Implementation
-----------------------------------
Decision: Use linear propagation (no dynamics, no learning in connectome).
Evidence: The behavior-rate model is a static GLM; a transparent post-hoc linear
          readout is the least-assumptive way to ask whether important ORNs sit
          in privileged wiring positions.
Implementation: s_pn = A @ s_orn and s_kc = B @ s_pn after inferring matrix
                orientation and aligning ORN ordering via metadata.

Decision: Require explicit ORN ordering metadata for alignment.
Evidence: Silent receptor-order mismatches are a common, catastrophic failure
          mode in connectomics pipelines; "shape matches" is not sufficient.
Implementation: If `connectivity_metadata.json` lacks receptor names, or if any
                model receptor is missing from that list, raise ValueError and
                skip/abort depending on caller strictness.

Decision: Define "connectome-amplified importance" via two-hop KC fanout.
Evidence: PNs are an intermediate layer; KC reach is a closer proxy for
          downstream capacity in the canonical ORN→PN→KC pathway.
Implementation: fanout_kc = sum_kc ( (B @ A)[:, orn] ), normalize by mean, then
                amplified = base_importance * fanout_kc_norm.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


PathLike = Union[str, Path]


def autodetect_connectome_dir(repo_root: PathLike) -> Optional[Path]:
    """
    Try common repository-relative locations for connectivity artifacts.

    Search order:
      1) data/pgcn_features/connectivity
      2) data/connectivity

    Returns:
        Path to a directory that contains both `orn_pn_connectivity.pt` and
        `pn_kc_connectivity.pt`, else None.
    """
    root = Path(repo_root)
    candidates = [
        root / "data/pgcn_features/connectivity",
        root / "data/connectivity",
    ]
    for d in candidates:
        if (d / "orn_pn_connectivity.pt").exists() and (d / "pn_kc_connectivity.pt").exists():
            return d
    return None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _torch_load_tensor(path: Path) -> "torch.Tensor":
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required to load .pt connectivity matrices.")
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # pragma: no cover
        obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj
    return torch.tensor(obj)


def load_connectome_matrices(connectome_dir: PathLike) -> Tuple["torch.Tensor", "torch.Tensor", Dict]:
    """
    Load ORN→PN and PN→KC connectivity matrices (+ optional metadata).

    Expected files:
      - orn_pn_connectivity.pt
      - pn_kc_connectivity.pt
      - connectivity_metadata.json (optional, but strongly recommended)
    """
    d = Path(connectome_dir)
    orn_pn_path = d / "orn_pn_connectivity.pt"
    pn_kc_path = d / "pn_kc_connectivity.pt"
    meta_path = d / "connectivity_metadata.json"

    if not orn_pn_path.exists():
        raise FileNotFoundError(f"Missing ORN→PN connectivity: {orn_pn_path}")
    if not pn_kc_path.exists():
        raise FileNotFoundError(f"Missing PN→KC connectivity: {pn_kc_path}")

    A = _torch_load_tensor(orn_pn_path).to(dtype=torch.float32, device="cpu")
    B = _torch_load_tensor(pn_kc_path).to(dtype=torch.float32, device="cpu")

    meta: Dict = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    meta["_paths"] = {
        "connectome_dir": str(d.resolve()),
        "orn_pn_connectivity": str(orn_pn_path.resolve()),
        "pn_kc_connectivity": str(pn_kc_path.resolve()),
        "connectivity_metadata": str(meta_path.resolve()) if meta_path.exists() else None,
    }
    meta["_hashes"] = {
        "orn_pn_connectivity_sha256": _sha256_file(orn_pn_path),
        "pn_kc_connectivity_sha256": _sha256_file(pn_kc_path),
        "connectivity_metadata_sha256": _sha256_file(meta_path) if meta_path.exists() else None,
    }
    meta["_loaded_shapes"] = {
        "orn_pn": list(A.shape),
        "pn_kc": list(B.shape),
    }
    return A, B, meta


def orient_connectome(A: "torch.Tensor", B: "torch.Tensor", n_orn: int) -> Tuple["torch.Tensor", "torch.Tensor", Dict]:
    """
    Orient matrices for propagation:
      - A_oriented: (n_pn, n_orn_total)  [rows = PNs, cols = ORNs]
      - B_oriented: (n_kc, n_pn)         [rows = KCs, cols = PNs]

    Orientation is inferred from shapes. The only hard requirement is that the
    PN dimension is shared between A and B.

    Args:
        A: ORN↔PN connectivity (either ORN×PN or PN×ORN)
        B: PN↔KC connectivity (either PN×KC or KC×PN)
        n_orn: Expected ORN count for the *model* (used only to break ties when
               shapes are ambiguous).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for connectome analysis.")

    a0, a1 = int(A.shape[0]), int(A.shape[1])
    b0, b1 = int(B.shape[0]), int(B.shape[1])

    # Identify PN dimension as the shared dimension between A and B.
    shared = sorted({a0, a1}.intersection({b0, b1}))
    if len(shared) != 1:
        raise ValueError(
            "Cannot infer PN dimension from shapes: "
            f"A={tuple(A.shape)}, B={tuple(B.shape)} (shared={shared})."
        )
    n_pn = shared[0]

    # Decide A orientation: rows must be PNs.
    if a0 == n_pn and a1 != n_pn:
        A_pn_by_orn = A
        a_oriented = "as_is"
    elif a1 == n_pn and a0 != n_pn:
        A_pn_by_orn = A.t()
        a_oriented = "transposed"
    else:
        # Ambiguous (e.g., a0==a1==n_pn). Fall back to expected ORN count.
        if a0 == n_pn and a1 == n_pn:
            raise ValueError(
                "Ambiguous ORN↔PN matrix (square). Provide non-square connectivity or metadata. "
                f"A={tuple(A.shape)}"
            )
        raise ValueError(f"Unexpected ORN↔PN shape: A={tuple(A.shape)} with inferred n_pn={n_pn}.")

    # Decide B orientation: columns must be PNs.
    if b1 == n_pn and b0 != n_pn:
        B_kc_by_pn = B
        b_oriented = "as_is"
    elif b0 == n_pn and b1 != n_pn:
        B_kc_by_pn = B.t()
        b_oriented = "transposed"
    else:
        if b0 == n_pn and b1 == n_pn:
            raise ValueError(
                "Ambiguous PN↔KC matrix (square). Provide non-square connectivity or metadata. "
                f"B={tuple(B.shape)}"
            )
        raise ValueError(f"Unexpected PN↔KC shape: B={tuple(B.shape)} with inferred n_pn={n_pn}.")

    report = {
        "inferred": {"n_pn": int(n_pn), "n_orn_model_expected": int(n_orn)},
        "input_shapes": {"A": [a0, a1], "B": [b0, b1]},
        "chosen": {
            "A_oriented": a_oriented,
            "B_oriented": b_oriented,
            "A_oriented_shape": list(A_pn_by_orn.shape),
            "B_oriented_shape": list(B_kc_by_pn.shape),
        },
    }

    return A_pn_by_orn, B_kc_by_pn, report


def align_orn_connectome(
    A_pn_by_orn: "torch.Tensor",
    connectome_receptors: Sequence[str],
    target_receptors: Sequence[str],
) -> Tuple["torch.Tensor", Dict]:
    """
    Subset/reorder A columns (ORNs) to match the model's ORN order.

    Args:
        A_pn_by_orn: ORN→PN matrix oriented as (n_pn, n_orn_total)
        connectome_receptors: receptor names in A's ORN axis order
        target_receptors: receptor names in the *model* order (length = 55 for adult-only runs)
    """
    if A_pn_by_orn.shape[1] != len(connectome_receptors):
        raise ValueError(
            "connectome receptor_names length does not match A's ORN dimension: "
            f"len(receptor_names)={len(connectome_receptors)} vs A.shape[1]={int(A_pn_by_orn.shape[1])}."
        )

    idx_by_name = {str(r): i for i, r in enumerate(connectome_receptors)}
    missing = [r for r in target_receptors if str(r) not in idx_by_name]
    if missing:
        raise ValueError(
            "Connectome receptor_names missing model receptors (cannot align ORN order). "
            f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''}"
        )

    indices = torch.tensor([idx_by_name[str(r)] for r in target_receptors], dtype=torch.long)
    A_aligned = A_pn_by_orn.index_select(dim=1, index=indices)

    report = {
        "n_connectome_receptors_total": int(len(connectome_receptors)),
        "n_model_receptors": int(len(target_receptors)),
        "missing_model_receptors": missing,
        "selected_indices": indices.detach().cpu().numpy().astype(int).tolist(),
    }
    return A_aligned, report


def _to_1d_float64(x: Union[np.ndarray, "torch.Tensor", Sequence[float]]) -> np.ndarray:
    if isinstance(x, np.ndarray):
        arr = x.astype(np.float64, copy=False)
    elif TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        arr = x.detach().cpu().numpy().astype(np.float64, copy=False)
    else:
        arr = np.asarray(list(x), dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D vector, got shape {arr.shape}.")
    return arr


def _gini_nonnegative(x: np.ndarray) -> float:
    """Gini coefficient for a non-negative 1D vector (0 = uniform, 1 = maximally concentrated)."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("gini expects 1D vector")
    if np.any(x < 0):
        raise ValueError("gini expects non-negative values")
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    xs = np.sort(x)
    n = xs.size
    i = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * np.sum(i * xs) / (n * total)) - (n + 1) / n)


def _stable_rank_desc(values: np.ndarray, labels: Sequence[str]) -> List[int]:
    """
    Deterministic rank for descending values with label tiebreaker.

    Returns:
        ranks: list of 1-based ranks aligned to input order.
    """
    values = np.asarray(values, dtype=np.float64)
    labels_arr = np.asarray([str(x) for x in labels], dtype=object)
    order = np.lexsort((labels_arr, -values))  # primary: -values, secondary: label
    ranks = np.empty_like(order, dtype=int)
    ranks[order] = np.arange(1, len(values) + 1, dtype=int)
    return ranks.tolist()


@dataclass(frozen=True)
class ConnectomeInfluenceResult:
    """
    Container for computed connectome-aware summaries.

    Attributes:
        s_pn: PN influence vector (n_pn,)
        s_kc: KC influence vector (n_kc,)
        orn_table: ORN-level table with base and amplified importance
        pn_table: PN influence table (sorted desc)
        kc_table: KC influence table (sorted desc)
        summary: JSON-serializable summary metrics
    """

    s_pn: np.ndarray
    s_kc: np.ndarray
    orn_table: pd.DataFrame
    pn_table: pd.DataFrame
    kc_table: pd.DataFrame
    summary: Dict


def compute_connectome_influence(
    s_orn: Union[np.ndarray, "torch.Tensor", Sequence[float]],
    A_pn_by_orn: "torch.Tensor",
    B_kc_by_pn: "torch.Tensor",
    *,
    receptor_names: Sequence[str],
    pn_ids: Optional[Sequence[Union[int, str]]] = None,
    kc_ids: Optional[Sequence[Union[int, str]]] = None,
    top_pn: int = 100,
    top_kc: int = 200,
) -> ConnectomeInfluenceResult:
    """
    Compute PN/KC influence from an ORN importance vector via linear propagation.

    Args:
        s_orn: Non-negative ORN importance vector (length = n_orn)
        A_pn_by_orn: ORN→PN matrix, oriented as (n_pn, n_orn)
        B_kc_by_pn: PN→KC matrix, oriented as (n_kc, n_pn)
        receptor_names: ORN names aligned to s_orn (and A columns)
        pn_ids: Optional PN identifiers aligned to A rows
        kc_ids: Optional KC identifiers aligned to B rows
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise ImportError("PyTorch is required for connectome analysis.")

    s_orn_np = _to_1d_float64(s_orn)
    if len(s_orn_np) != int(A_pn_by_orn.shape[1]):
        raise ValueError(
            "s_orn length does not match A ORN dimension: "
            f"len(s_orn)={len(s_orn_np)} vs A.shape[1]={int(A_pn_by_orn.shape[1])}."
        )
    if int(B_kc_by_pn.shape[1]) != int(A_pn_by_orn.shape[0]):
        raise ValueError(
            "B and A PN dimensions do not match: "
            f"B.shape[1]={int(B_kc_by_pn.shape[1])} vs A.shape[0]={int(A_pn_by_orn.shape[0])}."
        )
    if len(receptor_names) != len(s_orn_np):
        raise ValueError("receptor_names must align to s_orn (same length).")
    if np.any(s_orn_np < 0):
        raise ValueError("s_orn must be non-negative for this interpretation (use abs(weights) or similar).")

    s_orn_t = torch.tensor(s_orn_np, dtype=torch.float64).unsqueeze(1)  # (n_orn, 1)

    # s_pn = A @ s_orn
    if A_pn_by_orn.is_sparse:
        s_pn_t = torch.sparse.mm(A_pn_by_orn.to(dtype=torch.float64), s_orn_t)
    else:
        s_pn_t = A_pn_by_orn.to(dtype=torch.float64).mm(s_orn_t)

    # s_kc = B @ s_pn
    if B_kc_by_pn.is_sparse:
        s_kc_t = torch.sparse.mm(B_kc_by_pn.to(dtype=torch.float64), s_pn_t)
    else:
        s_kc_t = B_kc_by_pn.to(dtype=torch.float64).mm(s_pn_t)

    s_pn = s_pn_t.squeeze(1).detach().cpu().numpy().astype(np.float64)
    s_kc = s_kc_t.squeeze(1).detach().cpu().numpy().astype(np.float64)

    # Fanout metrics per ORN
    A_dense = A_pn_by_orn.to_dense() if A_pn_by_orn.is_sparse else A_pn_by_orn
    B_dense = B_kc_by_pn.to_dense() if B_kc_by_pn.is_sparse else B_kc_by_pn
    A_dense = A_dense.to(dtype=torch.float64)
    B_dense = B_dense.to(dtype=torch.float64)

    fanout_pn = A_dense.sum(dim=0).detach().cpu().numpy().astype(np.float64)  # (n_orn,)
    eff_kc_matrix = B_dense.mm(A_dense)  # (n_kc, n_orn)
    fanout_kc = eff_kc_matrix.sum(dim=0).detach().cpu().numpy().astype(np.float64)  # (n_orn,)

    # Normalized amplification factor (mean=1). Handle all-zero defensively.
    mean_fanout_kc = float(np.mean(fanout_kc)) if float(np.mean(fanout_kc)) > 0 else 1.0
    amp_factor = fanout_kc / mean_fanout_kc
    amplified = s_orn_np * amp_factor

    orn_table = pd.DataFrame(
        {
            "receptor": [str(r) for r in receptor_names],
            "base_importance": s_orn_np,
            "fanout_pn": fanout_pn,
            "fanout_kc": fanout_kc,
            "amplification_factor_kc_mean1": amp_factor,
            "connectome_amplified_importance": amplified,
        }
    )
    orn_table["rank_base_importance"] = _stable_rank_desc(orn_table["base_importance"].to_numpy(), orn_table["receptor"])
    orn_table["rank_connectome_amplified_importance"] = _stable_rank_desc(
        orn_table["connectome_amplified_importance"].to_numpy(), orn_table["receptor"]
    )
    orn_table["rank_fanout_kc"] = _stable_rank_desc(orn_table["fanout_kc"].to_numpy(), orn_table["receptor"])

    orn_table = orn_table.sort_values(
        ["connectome_amplified_importance", "receptor"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # PN/KC influence tables
    pn_ids_list = [str(i) for i in range(len(s_pn))] if pn_ids is None else [str(x) for x in pn_ids]
    kc_ids_list = [str(i) for i in range(len(s_kc))] if kc_ids is None else [str(x) for x in kc_ids]
    if len(pn_ids_list) != len(s_pn):
        raise ValueError("pn_ids length must match A rows.")
    if len(kc_ids_list) != len(s_kc):
        raise ValueError("kc_ids length must match B rows.")

    pn_table = pd.DataFrame({"pn_id": pn_ids_list, "pn_index": np.arange(len(s_pn), dtype=int), "influence": s_pn})
    pn_table = pn_table.sort_values(["influence", "pn_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    pn_table = pn_table.head(int(top_pn)).copy()

    kc_table = pd.DataFrame({"kc_id": kc_ids_list, "kc_index": np.arange(len(s_kc), dtype=int), "influence": s_kc})
    kc_table = kc_table.sort_values(["influence", "kc_id"], ascending=[False, True], kind="mergesort").reset_index(drop=True)
    kc_table = kc_table.head(int(top_kc)).copy()

    # Downstream concentration summary based on ORN contributions to total KC mass.
    contributions = s_orn_np * fanout_kc
    total_downstream = float(np.sum(contributions))
    contrib_sorted = np.sort(contributions)[::-1]

    def frac_top_k(k: int) -> float:
        if total_downstream <= 0:
            return 0.0
        k = max(1, min(int(k), len(contrib_sorted)))
        return float(np.sum(contrib_sorted[:k]) / total_downstream)

    summary = {
        "n_orns": int(len(s_orn_np)),
        "n_pns": int(len(s_pn)),
        "n_kcs": int(len(s_kc)),
        "total_pn_influence": float(np.sum(s_pn)),
        "total_kc_influence": float(np.sum(s_kc)),
        "total_downstream_contribution_mass": total_downstream,
        "downstream_fraction_top_5_orns": frac_top_k(5),
        "downstream_fraction_top_10_orns": frac_top_k(10),
        "downstream_fraction_top_15_orns": frac_top_k(15),
        "downstream_gini_contribution": _gini_nonnegative(contributions),
        "top_orns_by_amplified_importance": orn_table.head(10)[
            ["receptor", "connectome_amplified_importance", "base_importance", "fanout_kc"]
        ].to_dict(orient="records"),
        "top_pns_by_influence": pn_table.head(10)[["pn_id", "influence"]].to_dict(orient="records"),
        "top_kcs_by_influence": kc_table.head(10)[["kc_id", "influence"]].to_dict(orient="records"),
    }

    return ConnectomeInfluenceResult(
        s_pn=s_pn,
        s_kc=s_kc,
        orn_table=orn_table,
        pn_table=pn_table,
        kc_table=kc_table,
        summary=summary,
    )


def write_connectome_outputs(output_dir: PathLike, results: Mapping) -> None:
    """
    Write connectome analysis artifacts under `output_dir/connectome_analysis/`.

    Expected keys in `results`:
      - connectome_inputs: dict (written to connectome_inputs.json)
      - orn_table: pd.DataFrame
      - pn_table: pd.DataFrame
      - kc_table: pd.DataFrame
      - summary: dict (written to connectome_summary.json)
      - orn_to_pn_edges: Optional[pd.DataFrame]
    """
    out_root = Path(output_dir) / "connectome_analysis"
    out_root.mkdir(parents=True, exist_ok=True)

    (out_root / "connectome_inputs.json").write_text(
        json.dumps(results["connectome_inputs"], indent=2, sort_keys=True),
        encoding="utf-8",
    )
    results["orn_table"].to_csv(out_root / "orn_connectome_amplified_importance.csv", index=False)
    results["pn_table"].to_csv(out_root / "pn_influence.csv", index=False)
    results["kc_table"].to_csv(out_root / "kc_influence.csv", index=False)
    (out_root / "connectome_summary.json").write_text(
        json.dumps(results["summary"], indent=2, sort_keys=True),
        encoding="utf-8",
    )

    edges = results.get("orn_to_pn_edges", None)
    if isinstance(edges, pd.DataFrame):
        edges.to_csv(out_root / "orn_to_pn_top_edges.csv", index=False)


def build_orn_to_pn_top_edges(
    A_pn_by_orn: "torch.Tensor",
    receptor_names: Sequence[str],
    *,
    pn_ids: Optional[Sequence[Union[int, str]]] = None,
    top_orns: Sequence[str],
    top_edges_per_orn: int = 30,
) -> pd.DataFrame:
    """
    Optional helper: build ORN→PN edge list for a subset of ORNs.
    """
    A_dense = A_pn_by_orn.to_dense() if A_pn_by_orn.is_sparse else A_pn_by_orn
    A_dense = A_dense.detach().cpu().numpy().astype(np.float64)
    idx_by_receptor = {str(r): i for i, r in enumerate(receptor_names)}
    pn_ids_list = [str(i) for i in range(A_dense.shape[0])] if pn_ids is None else [str(x) for x in pn_ids]

    rows: List[Dict[str, object]] = []
    for receptor in top_orns:
        if str(receptor) not in idx_by_receptor:
            continue
        j = idx_by_receptor[str(receptor)]
        col = A_dense[:, j]
        nz = np.flatnonzero(col > 0)
        if nz.size == 0:
            continue
        # Sort edges by weight desc, deterministic by pn_id tie-breaker.
        weights = col[nz]
        pn_labels = np.asarray([pn_ids_list[i] for i in nz], dtype=object)
        order = np.lexsort((pn_labels, -weights))
        keep = order[: max(1, int(top_edges_per_orn))]
        for k in keep:
            i = int(nz[k])
            rows.append(
                {
                    "receptor": str(receptor),
                    "pn_id": str(pn_ids_list[i]),
                    "pn_index": i,
                    "weight": float(col[i]),
                }
            )

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df
    return df.sort_values(["weight", "receptor", "pn_id"], ascending=[False, True, True], kind="mergesort").reset_index(
        drop=True
    )

