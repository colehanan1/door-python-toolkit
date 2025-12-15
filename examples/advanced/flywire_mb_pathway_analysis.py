"""
FlyWire Mushroom Body Pathway Analysis
=======================================

Validate LASSO-identified receptors using FlyWire connectome data.

This script:
1. Loads LASSO behavioral prediction results
2. Maps receptors to FlyWire ORN neurons
3. Traces pathways: ORN → PN → KC → MBON
4. Calculates connectivity metrics
5. Integrates anatomy with behavioral importance
6. Exports priority matrix for experimental validation

Usage:
    python examples/advanced/flywire_mb_pathway_analysis.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from door_toolkit.flywire import FlyWireMapper
from door_toolkit.flywire.mushroom_body_tracer import MushroomBodyTracer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_lasso_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all LASSO behavioral prediction results."""
    results = {}

    for json_file in results_dir.glob("*_model.json"):
        condition_name = json_file.stem.replace("_model", "")

        with open(json_file, "r") as f:
            data = json.load(f)

        # Handle NaN in cv_r2_score
        if isinstance(data.get("cv_r2_score"), str) and data["cv_r2_score"] == "NaN":
            data["cv_r2_score"] = np.nan

        results[condition_name] = data
        logger.info(
            f"Loaded {condition_name}: {data['n_receptors_selected']} receptors, "
            f"R² = {data.get('cv_r2_score', 'N/A')}"
        )

    return results


def extract_top_receptors(lasso_results: Dict, n_top: int = 10) -> Dict[str, float]:
    """
    Extract top receptors across all conditions.

    Args:
        lasso_results: Dict of {condition: model_data}
        n_top: Number of top receptors to extract per condition

    Returns:
        Dict of {receptor: max_abs_weight} across all conditions
    """
    # Sensillum to receptor mapping
    SENSILLUM_TO_RECEPTOR = {
        "ab2B": "Or85a",  # ab2B sensillum expresses Or85a (and Or85b)
        "ab2A": "Or59b",
        "ab3A": "Or22a",
        "ab1A": "Or42b",
    }

    receptor_weights = {}

    for condition, data in lasso_results.items():
        weights = data.get("lasso_weights", {})

        for receptor, weight in weights.items():
            abs_weight = abs(weight)

            # Translate sensillum labels to receptor names
            mapped_receptor = SENSILLUM_TO_RECEPTOR.get(receptor, receptor)

            if (
                mapped_receptor not in receptor_weights
                or abs_weight > receptor_weights[mapped_receptor]
            ):
                receptor_weights[mapped_receptor] = abs_weight

    # Sort and take top N
    sorted_receptors = sorted(receptor_weights.items(), key=lambda x: x[1], reverse=True)[:n_top]

    return dict(sorted_receptors)


def main():
    """Run complete FlyWire mushroom body pathway analysis."""
    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "flywire"
    results_dir = base_dir / "behavioral_prediction_results"
    output_dir = base_dir / "flywire_mb_analysis"
    output_dir.mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("FLYWIRE MUSHROOM BODY PATHWAY ANALYSIS")
    logger.info("=" * 80)

    # ============================================================================
    # STEP 1: Load LASSO Results
    # ============================================================================
    logger.info("\n[STEP 1] Loading LASSO behavioral prediction results...")

    lasso_results = load_lasso_results(results_dir)

    if not lasso_results:
        logger.error("No LASSO results found! Please run behavioral prediction first.")
        return

    # Extract top receptors
    top_receptors = extract_top_receptors(lasso_results, n_top=10)
    logger.info(f"\nTop {len(top_receptors)} receptors across all conditions:")
    for i, (receptor, weight) in enumerate(top_receptors.items(), 1):
        logger.info(f"  {i:2d}. {receptor:12s}  max_weight = {weight:.4f}")

    # ============================================================================
    # STEP 2: Initialize FlyWire Mapper
    # ============================================================================
    logger.info("\n[STEP 2] Initializing FlyWire mapper...")

    labels_path = data_dir / "processed_labels.csv.gz"
    if not labels_path.exists():
        logger.error(f"FlyWire labels not found: {labels_path}")
        return

    mapper = FlyWireMapper(str(labels_path), auto_parse=True)

    # ============================================================================
    # STEP 3: Map Receptors to ORN Neurons
    # ============================================================================
    logger.info("\n[STEP 3] Mapping receptors to FlyWire ORN neurons...")

    receptor_to_orns = {}

    for receptor in top_receptors.keys():
        logger.info(f"\nMapping {receptor}...")
        cells = mapper.find_receptor_cells(receptor)

        if cells:
            orn_ids = [cell["root_id"] for cell in cells]
            receptor_to_orns[receptor] = orn_ids
            logger.info(f"  ✓ Found {len(orn_ids)} ORN neurons")
        else:
            logger.warning(f"  ✗ No neurons found for {receptor}")

    logger.info(f"\nMapped {len(receptor_to_orns)}/{len(top_receptors)} receptors to FlyWire")

    # ============================================================================
    # STEP 4: Initialize Mushroom Body Tracer
    # ============================================================================
    logger.info("\n[STEP 4] Initializing mushroom body pathway tracer...")

    synapse_path = data_dir / "connections_princeton.csv.gz"
    cell_types_path = data_dir / "consolidated_cell_types.csv.gz"

    if not synapse_path.exists():
        logger.error(f"Synapse data not found: {synapse_path}")
        return
    if not cell_types_path.exists():
        logger.error(f"Cell types not found: {cell_types_path}")
        return

    tracer = MushroomBodyTracer(
        synapse_path=str(synapse_path),
        cell_types_path=str(cell_types_path),
        min_synapse_threshold=1,
    )

    # ============================================================================
    # STEP 5: Trace Pathways (ORN → PN → KC → MBON)
    # ============================================================================
    logger.info("\n[STEP 5] Tracing mushroom body pathways...")

    pathways = []
    metrics_list = []

    for receptor, orn_ids in receptor_to_orns.items():
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Tracing pathway for {receptor}")
        logger.info(f"{'=' * 60}")

        try:
            pathway = tracer.trace_receptor_pathway(receptor, orn_ids)
            pathways.append(pathway)

            # Calculate connectivity metrics
            metrics = tracer.calculate_connectivity_metrics(pathway)
            metrics_list.append(metrics)

        except Exception as e:
            logger.error(f"Failed to trace {receptor}: {e}")
            continue

    # ============================================================================
    # STEP 6: Export Pathway Summaries
    # ============================================================================
    logger.info("\n[STEP 6] Exporting pathway summaries...")

    if pathways:
        tracer.export_pathway_csv(pathways, str(output_dir / "flywire_pathway_summaries.csv"))

    if metrics_list:
        tracer.export_metrics_csv(
            metrics_list, str(output_dir / "flywire_connectivity_metrics.csv")
        )

    # ============================================================================
    # STEP 7: Integrate with LASSO Sensitivity
    # ============================================================================
    logger.info("\n[STEP 7] Creating final priority matrix...")

    priority_rows = []

    for receptor, lasso_weight in top_receptors.items():
        # Find matching metrics
        metric = next((m for m in metrics_list if m.receptor_name == receptor), None)

        if metric:
            # Normalize LASSO weight (0-1 scale)
            max_weight = max(top_receptors.values())
            normalized_lasso = lasso_weight / max_weight if max_weight > 0 else 0.0

            # Composite score: 60% behavioral importance + 40% circuit connectivity
            final_score = (normalized_lasso * 0.6) + (metric.circuit_score * 0.4)

            # Determine priority
            if final_score > 0.7:
                priority = "TEST FIRST ⭐⭐⭐"
            elif final_score > 0.5:
                priority = "TEST SECOND ⭐⭐"
            elif final_score > 0.3:
                priority = "OPTIONAL ⭐"
            else:
                priority = "SKIP"

            priority_rows.append(
                {
                    "receptor": receptor,
                    "lasso_weight": lasso_weight,
                    "normalized_lasso": normalized_lasso,
                    "circuit_score": metric.circuit_score,
                    "final_score": final_score,
                    "orn_to_pn_strength": metric.orn_to_pn_strength,
                    "kc_coverage": metric.kc_coverage,
                    "alpha_beta_fraction": metric.alpha_beta_fraction,
                    "gamma_fraction": metric.gamma_fraction,
                    "circuit_type": metric.to_dict()["circuit_type"],
                    "experiment_priority": priority,
                }
            )

    priority_df = pd.DataFrame(priority_rows)
    priority_df = priority_df.sort_values("final_score", ascending=False)
    priority_df.to_csv(output_dir / "final_priority_matrix.csv", index=False)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL PRIORITY MATRIX")
    logger.info("=" * 80)
    print(
        priority_df[
            [
                "receptor",
                "normalized_lasso",
                "circuit_score",
                "final_score",
                "circuit_type",
                "experiment_priority",
            ]
        ].to_string(index=False)
    )

    # ============================================================================
    # STEP 8: Generate Visualizations
    # ============================================================================
    logger.info("\n[STEP 8] Generating visualizations...")

    # Plot 1: Scatter plot of LASSO weight vs circuit score
    fig, ax = plt.subplots(figsize=(10, 8))

    for _, row in priority_df.iterrows():
        color = "blue" if row["circuit_type"] == "appetitive" else "red"
        size = row["final_score"] * 1000
        ax.scatter(
            row["normalized_lasso"],
            row["circuit_score"],
            s=size,
            alpha=0.6,
            color=color,
            edgecolors="k",
            linewidths=1.5,
        )
        ax.annotate(
            row["receptor"],
            (row["normalized_lasso"], row["circuit_score"]),
            fontsize=10,
            xytext=(5, 5),
            textcoords="offset points",
        )

    ax.set_xlabel("LASSO Importance (Normalized)", fontsize=14)
    ax.set_ylabel("Circuit Connectivity Score", fontsize=14)
    ax.set_title(
        "Behavioral Importance vs Anatomical Validation\n"
        "Size = Final Priority Score | Color = Circuit Type",
        fontsize=16,
    )
    ax.grid(alpha=0.3)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="blue", label="Appetitive (α/β lobe)"),
        Patch(facecolor="red", label="Aversive (γ lobe)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_dir / "priority_scatter.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved priority scatter plot to {output_dir / 'priority_scatter.png'}")

    # Plot 2: Bar chart of final scores
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["blue" if ct == "appetitive" else "red" for ct in priority_df["circuit_type"]]
    ax.barh(
        range(len(priority_df)), priority_df["final_score"], color=colors, alpha=0.7, edgecolor="k"
    )
    ax.set_yticks(range(len(priority_df)))
    ax.set_yticklabels(priority_df["receptor"])
    ax.set_xlabel("Final Priority Score", fontsize=12)
    ax.set_title(
        "Experimental Priority Ranking\n(LASSO Importance + Circuit Connectivity)", fontsize=14
    )
    ax.axvline(0.7, color="green", linestyle="--", alpha=0.5, label="TEST FIRST threshold")
    ax.axvline(0.5, color="orange", linestyle="--", alpha=0.5, label="TEST SECOND threshold")
    ax.legend()
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_dir / "priority_bar.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved priority bar chart to {output_dir / 'priority_bar.png'}")

    # ============================================================================
    # STEP 9: Generate Report
    # ============================================================================
    logger.info("\n[STEP 9] Generating analysis report...")

    report_lines = [
        "=" * 80,
        "FLYWIRE MUSHROOM BODY PATHWAY ANALYSIS REPORT",
        "=" * 80,
        "",
        "OVERVIEW",
        "-" * 80,
        f"LASSO conditions analyzed: {len(lasso_results)}",
        f"Top receptors identified: {len(top_receptors)}",
        f"Receptors mapped to FlyWire: {len(receptor_to_orns)}",
        f"Pathways successfully traced: {len(pathways)}",
        "",
        "TOP RECEPTORS BY LASSO WEIGHT",
        "-" * 80,
    ]

    for i, (receptor, weight) in enumerate(list(top_receptors.items())[:10], 1):
        report_lines.append(f"{i:2d}. {receptor:12s}  weight = {weight:.4f}")

    report_lines.extend(
        [
            "",
            "CONNECTIVITY METRICS",
            "-" * 80,
            priority_df[
                [
                    "receptor",
                    "orn_to_pn_strength",
                    "kc_coverage",
                    "alpha_beta_fraction",
                    "circuit_score",
                ]
            ].to_string(index=False),
            "",
            "FINAL PRIORITY MATRIX",
            "-" * 80,
            priority_df[
                [
                    "receptor",
                    "normalized_lasso",
                    "circuit_score",
                    "final_score",
                    "circuit_type",
                    "experiment_priority",
                ]
            ].to_string(index=False),
            "",
            "KEY FINDINGS",
            "-" * 80,
        ]
    )

    # Add key findings
    top_receptor = priority_df.iloc[0]
    report_lines.append(
        f"1. TOP CANDIDATE: {top_receptor['receptor']} "
        f"(Final Score: {top_receptor['final_score']:.3f})"
    )
    report_lines.append(
        f"   - LASSO importance: {top_receptor['normalized_lasso']:.3f} "
        f"(rank 1/{len(priority_df)})"
    )
    report_lines.append(
        f"   - Circuit connectivity: {top_receptor['circuit_score']:.3f}"
    )
    report_lines.append(
        f"   - Circuit type: {top_receptor['circuit_type'].upper()}"
    )
    report_lines.append(
        f"   - Recommendation: {top_receptor['experiment_priority']}"
    )

    appetitive_count = sum(priority_df["circuit_type"] == "appetitive")
    aversive_count = sum(priority_df["circuit_type"] == "aversive")
    report_lines.append("")
    report_lines.append(
        f"2. CIRCUIT TYPE DISTRIBUTION: {appetitive_count} appetitive, {aversive_count} aversive"
    )

    high_priority = sum(priority_df["final_score"] > 0.7)
    report_lines.append("")
    report_lines.append(
        f"3. EXPERIMENTAL VALIDATION: {high_priority} high-priority receptors for optogenetic testing"
    )

    report_lines.extend(
        [
            "",
            "OUTPUT FILES",
            "-" * 80,
            f"1. Pathway summaries: {output_dir / 'flywire_pathway_summaries.csv'}",
            f"2. Connectivity metrics: {output_dir / 'flywire_connectivity_metrics.csv'}",
            f"3. Priority matrix: {output_dir / 'final_priority_matrix.csv'}",
            f"4. Priority scatter plot: {output_dir / 'priority_scatter.png'}",
            f"5. Priority bar chart: {output_dir / 'priority_bar.png'}",
            f"6. This report: {output_dir / 'analysis_report.txt'}",
            "",
            "=" * 80,
        ]
    )

    report_text = "\n".join(report_lines)

    with open(output_dir / "analysis_report.txt", "w") as f:
        f.write(report_text)

    print("\n" + report_text)

    logger.info(f"\n✓ Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
