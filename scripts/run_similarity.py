#!/usr/bin/env python
"""CLI for odor-odor similarity with missing-data handling.

Examples
--------
From CSV (auto-detects odor columns and maps to DoOR names):
    python scripts/run_similarity.py \
        --csv /tmp/reaction_rates_no_citral.csv \
        --method cosine --mask-mode pairwise --min-overlap 10 \
        --cache-path door_cache --outdir out/sim_pairwise --plot

Global intersection mode:
    python scripts/run_similarity.py \
        --csv /tmp/reaction_rates_no_citral.csv \
        --method cosine --mask-mode global --min-overlap 10 \
        --cache-path door_cache --outdir out/sim_global --plot

Find similar to a query odor:
    python scripts/run_similarity.py \
        --csv /tmp/reaction_rates_no_citral.csv \
        --query Benzaldehyde --method pearson --mask-mode pairwise \
        --min-overlap 10 --cache-path door_cache --outdir out/sim_query --plot

Manual odor list (already DoOR names):
    python scripts/run_similarity.py \
        --odors "benzaldehyde,ethyl butyrate,1-hexanol,3-octanol" \
        --method cosine --mask-mode pairwise --min-overlap 10 \
        --cache-path door_cache --outdir out/sim_pairwise --plot
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Ensure src/ is importable
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root / "src") not in sys.path:
    sys.path.insert(0, str(_repo_root / "src"))

from door_toolkit.encoder import DoOREncoder
from door_toolkit.similarity import find_similar, similarity_matrix

# CSV odor name -> DoOR name mapping (same as compare_models_predictions.py)
CSV_ODOR_TO_DOOR = {
    "3-Octonol": "3-octanol",
    "Apple_Cider_Vinegar": "acetic acid",
    "Benzaldehyde": "benzaldehyde",
    "Citral": "citral",
    "Ethyl_Butyrate": "ethyl butyrate",
    "Hexanol": "1-hexanol",
    "Linalool": "linalool",
}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_similarity_heatmap(
    S: pd.DataFrame,
    method: str,
    mask_mode: str,
    outdir: Path,
) -> str:
    """Annotated similarity matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    vmin = -1.0 if method in ("pearson", "spearman") else 0.0
    sns.heatmap(
        S,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=1.0,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": f"{method} similarity"},
        ax=ax,
    )
    ax.set_title(
        f"Odor Similarity ({method}, {mask_mode})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    p = outdir / "similarity_heatmap.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_overlap_heatmap(
    overlap: pd.DataFrame,
    mask_mode: str,
    outdir: Path,
) -> str:
    """Annotated overlap count heatmap."""
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        overlap,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "shared measured features"},
        ax=ax,
    )
    ax.set_title(
        f"Feature Overlap ({mask_mode})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    p = outdir / "overlap_heatmap.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_coverage_bar(
    coverage: pd.DataFrame,
    n_total: int,
    outdir: Path,
) -> str:
    """Horizontal bar chart of per-odor receptor coverage."""
    fig, ax = plt.subplots(figsize=(8, max(4, len(coverage) * 0.6)))
    odors = coverage.index.tolist()
    measured = coverage["n_measured"].values
    colors = plt.cm.viridis(measured / n_total)

    bars = ax.barh(odors, measured, color=colors, edgecolor="black", linewidth=0.5)
    ax.axvline(n_total, color="red", linestyle="--", linewidth=1.5, label=f"total ({n_total})")
    ax.set_xlabel("Measured receptors", fontsize=12)
    ax.set_title("Per-Odor Receptor Coverage", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)

    # Value labels
    for bar, n, pct in zip(bars, measured, coverage["coverage_pct"].values):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{n} ({pct:.0f}%)",
            va="center", fontsize=10,
        )
    ax.set_xlim(0, n_total * 1.2)
    ax.invert_yaxis()
    plt.tight_layout()
    p = outdir / "coverage_bar.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(p)


def plot_query_results(
    df: pd.DataFrame,
    query: str,
    method: str,
    outdir: Path,
) -> str:
    """Horizontal bar chart of similarity scores for query mode."""
    fig, ax = plt.subplots(figsize=(9, max(4, len(df) * 0.7)))

    sims = df["similarity"].values
    targets = df["target_odor"].values
    overlaps = df["overlap_n"].values

    # Color by similarity value
    norm = plt.Normalize(vmin=min(sims.min(), 0), vmax=max(sims.max(), 1))
    cmap = plt.cm.RdYlGn
    colors = cmap(norm(sims))

    bars = ax.barh(targets, sims, color=colors, edgecolor="black", linewidth=0.5)

    # Value + overlap labels
    for bar, sim, ovlp in zip(bars, sims, overlaps):
        x = bar.get_width()
        label = f"{sim:.3f}  (n={ovlp})"
        ax.text(
            x + 0.01 if x >= 0 else x - 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            ha="left" if x >= 0 else "right",
            fontsize=10,
        )

    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(f"{method} similarity", fontsize=12)
    ax.set_title(
        f"Most Similar to '{query}'",
        fontsize=14,
        fontweight="bold",
    )
    ax.invert_yaxis()

    # Set sensible x limits
    pad = 0.15
    xmin = min(sims.min(), 0) - pad
    xmax = max(sims.max(), 0) + pad
    ax.set_xlim(xmin, xmax)

    plt.tight_layout()
    safe_name = query.replace(" ", "_").replace("/", "_")
    p = outdir / f"similar_to_{safe_name}.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------------------
# Odor detection / mapping
# ---------------------------------------------------------------------------


def _detect_odor_columns(csv_path: str) -> list[str]:
    """Auto-detect odor columns from CSV (skip dataset/air and >50% NaN cols)."""
    df = pd.read_csv(csv_path)
    non_odor = {"dataset", "air"}
    odor_cols = []
    for col in df.columns:
        if col.lower() in non_odor:
            continue
        n_missing = pd.to_numeric(df[col], errors="coerce").isna().sum()
        if len(df) > 0 and n_missing / len(df) > 0.5:
            continue
        odor_cols.append(col)
    return odor_cols


def _map_csv_odor_to_door(csv_name: str) -> str:
    """Map CSV odor name to DoOR name using the canonical mapping."""
    if csv_name in CSV_ODOR_TO_DOOR:
        return CSV_ODOR_TO_DOOR[csv_name]
    # Try case-insensitive match
    for k, v in CSV_ODOR_TO_DOOR.items():
        if k.lower() == csv_name.lower():
            return v
    sys.exit(
        f"Error: CSV odor '{csv_name}' not in mapping. "
        f"Known mappings: {CSV_ODOR_TO_DOOR}"
    )


def _get_odors(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    """Return (door_names, csv_to_door_map) from --csv, --odors, or --odor-file.

    When --csv is used, auto-detects odor columns and maps to DoOR names.
    Returns the mapping so query names can also be resolved.
    """
    csv_to_door: dict[str, str] = {}

    if args.csv:
        csv_odors = _detect_odor_columns(args.csv)
        print(f"Detected {len(csv_odors)} odor columns from CSV: {csv_odors}")
        door_odors = []
        for col in csv_odors:
            door_name = _map_csv_odor_to_door(col)
            csv_to_door[col.lower()] = door_name
            if col != door_name:
                print(f"  {col!r} → {door_name!r}")
            door_odors.append(door_name)
        return door_odors, csv_to_door

    if args.odor_file:
        path = Path(args.odor_file)
        if not path.exists():
            sys.exit(f"Error: odor file not found: {path}")
        odors = [
            line.strip()
            for line in path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
    elif args.odors:
        odors = [o.strip() for o in args.odors.split(",") if o.strip()]
    else:
        sys.exit("Error: provide --csv, --odors, or --odor-file")

    if len(odors) < 2:
        sys.exit("Error: need at least 2 odors")
    return odors, csv_to_door


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Odor-odor similarity with missing-data handling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to PER CSV (auto-detects odor columns and maps to DoOR names)",
    )
    parser.add_argument(
        "--odors",
        type=str,
        default=None,
        help='Comma-separated DoOR odor names (e.g. "benzaldehyde,ethanol")',
    )
    parser.add_argument(
        "--odor-file",
        type=str,
        default=None,
        help="Path to text file with one odor name per line",
    )
    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Query odor for find-similar mode (omit for full matrix)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cosine",
        choices=["cosine", "pearson", "spearman"],
        help="Similarity method (default: cosine)",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="pairwise",
        choices=["pairwise", "global"],
        help="pairwise: per-pair feature intersection (default); "
        "global: shared features across all odors",
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=10,
        help="Minimum shared measured features (default: 10)",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default="door_cache",
        help="Path to DoOR cache directory (default: door_cache)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for CSV artifacts",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of results for query mode (default: 10)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate plots (heatmaps, bar charts)",
    )
    args = parser.parse_args()

    encoder = DoOREncoder(args.cache_path, use_torch=False)
    odors, csv_to_door = _get_odors(args)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.query:
        # --- Query mode ---
        # Resolve query name (may be CSV name or DoOR name)
        query = args.query
        query_lower = query.lower()
        if csv_to_door and query_lower in csv_to_door:
            query = csv_to_door[query_lower]
            print(f"  Query {args.query!r} → {query!r}")
        elif query_lower not in encoder.name_to_inchikey:
            # Try mapping
            mapped = CSV_ODOR_TO_DOOR.get(args.query)
            if mapped:
                query = mapped
                print(f"  Query {args.query!r} → {query!r}")

        df = find_similar(
            odor_names=odors,
            query_odor=query,
            encoder=encoder,
            top_k=args.top_k,
            method=args.method,
            mask_mode=args.mask_mode,
            min_overlap=args.min_overlap,
        )
        safe_name = query.replace(" ", "_").replace("/", "_")
        out_path = outdir / f"similar_to_{safe_name}.csv"
        df.to_csv(out_path, index=False)

        print(f"=== Similarity: find-similar mode ===")
        print(f"Query odor   : {query}")
        print(f"Candidates   : {len(odors) - 1}")
        print(f"Method       : {args.method}")
        print(f"Mask mode    : {args.mask_mode}")
        print(f"Min overlap  : {args.min_overlap}")
        print(f"Results      : {len(df)}")
        n_skipped = df["similarity"].isna().sum()
        if n_skipped:
            print(f"Skipped      : {n_skipped} (insufficient overlap)")
        print(f"Output       : {out_path}")

        if args.plot:
            p = plot_query_results(df, query, args.method, outdir)
            print(f"Plot         : {p}")
    else:
        # --- Full matrix mode ---
        S, meta = similarity_matrix(
            odor_names=odors,
            encoder=encoder,
            method=args.method,
            mask_mode=args.mask_mode,
            min_overlap=args.min_overlap,
        )
        S.to_csv(outdir / "similarity_matrix.csv")
        meta["overlap_matrix"].to_csv(outdir / "overlap_matrix.csv")
        meta["coverage"].to_csv(outdir / "coverage.csv")

        n = len(odors)
        n_pairs = n * (n - 1) // 2
        print(f"=== Similarity: full matrix mode ===")
        print(f"Odors        : {n}")
        print(f"Features     : {encoder.n_channels}")
        print(f"Method       : {args.method}")
        print(f"Mask mode    : {args.mask_mode}")
        print(f"Min overlap  : {args.min_overlap}")
        if meta["global_overlap_n"] is not None:
            print(f"Global overlap: {meta['global_overlap_n']}")
        print(f"Total pairs  : {n_pairs}")
        print(f"Skipped      : {meta['n_skipped']}")
        print(f"Output dir   : {outdir}")
        print("Files:")
        print(f"  similarity_matrix.csv")
        print(f"  overlap_matrix.csv")
        print(f"  coverage.csv")

        if args.plot:
            plots = []
            plots.append(plot_similarity_heatmap(S, args.method, args.mask_mode, outdir))
            plots.append(plot_overlap_heatmap(meta["overlap_matrix"], args.mask_mode, outdir))
            plots.append(plot_coverage_bar(meta["coverage"], encoder.n_channels, outdir))
            print("Plots:")
            for p in plots:
                print(f"  {p}")


if __name__ == "__main__":
    main()
