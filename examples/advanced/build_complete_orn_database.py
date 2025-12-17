"""
Build Complete ORN-FlyWire Connectivity Database
=================================================

Create a comprehensive database mapping all 78 DoOR receptors to FlyWire
mushroom body connectivity.

This script:
1. Loads all 78 DoOR receptors
2. Maps each to FlyWire ORN neurons
3. Traces complete pathways (ORN → PN → KC → MBON)
4. Calculates connectivity metrics for each
5. Exports complete database (CSV + JSON)
6. Creates lookup functions for instant queries

Run once, use forever!

Usage:
    python examples/advanced/build_complete_orn_database.py

Output:
    flywire_orn_database/
      ├── flywire_orn_complete_v1.0.csv     # Excel-friendly
      ├── flywire_orn_complete_v1.0.json    # Python-friendly
      ├── database_statistics.txt           # Coverage report
      └── database_visualization.png        # Overview plot
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

matplotlib.use("Agg")

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from door_toolkit import DoOREncoder
from door_toolkit.flywire import FlyWireMapper
from door_toolkit.flywire.mushroom_body_tracer import MushroomBodyTracer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("orn_database_build.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def get_all_door_receptors(door_cache: str) -> List[str]:
    """Get all 78 DoOR receptors."""
    encoder = DoOREncoder(door_cache)

    # Load response matrix to get all receptors
    response_df = pd.read_parquet(Path(door_cache) / "response_matrix_norm.parquet")

    # Get all Or, Ir, Gr receptors
    all_receptors = response_df.columns.tolist()

    # Filter to olfactory receptors
    olfactory_receptors = [
        r for r in all_receptors
        if (r.startswith('Or') or r.startswith('Ir') or r.startswith('Gr'))
        and not any(skip in r for skip in ['ORco', 'Orco'])  # Skip co-receptors
    ]

    logger.info(f"Found {len(olfactory_receptors)} olfactory receptors in DoOR")
    return sorted(olfactory_receptors)


def map_receptor_to_flywire(
    receptor: str,
    mapper: FlyWireMapper,
    tracer: MushroomBodyTracer
) -> Dict:
    """
    Map a single receptor to FlyWire and trace pathway.

    Returns dict with all connectivity data or None if mapping fails.
    """
    try:
        # Find ORN neurons
        cells = mapper.find_receptor_cells(receptor)

        if not cells:
            logger.warning(f"No FlyWire neurons found for {receptor}")
            return {
                "receptor": receptor,
                "status": "not_found",
                "n_orns": 0,
                "error": "No FlyWire mapping"
            }

        orn_ids = [cell["root_id"] for cell in cells]

        # Trace pathway
        pathway = tracer.trace_receptor_pathway(receptor, orn_ids)

        # Calculate metrics
        metrics = tracer.calculate_connectivity_metrics(pathway)

        # Compile complete data
        result = {
            "receptor": receptor,
            "status": "success",
            # ORN data
            "n_orns": len(orn_ids),
            "orn_ids_sample": orn_ids[:5],  # Just first 5 for reference
            # PN data
            "n_pns": len(pathway.unique_pns),
            "pn_ids_sample": list(pathway.unique_pns)[:5],
            "orn_to_pn_synapses": pathway.total_orn_to_pn_synapses,
            # KC data
            "n_kcs": len(pathway.unique_kcs),
            "pn_to_kc_synapses": pathway.total_pn_to_kc_synapses,
            "kc_compartments": pathway.kc_compartments,
            "kc_alpha_beta": pathway.kc_compartments.get("alpha_beta", 0),
            "kc_gamma": pathway.kc_compartments.get("gamma", 0),
            "kc_alpha_prime_beta_prime": pathway.kc_compartments.get("alpha_prime_beta_prime", 0),
            # MBON data
            "n_mbons": len({step.target_id for step in pathway.mbon_connections}),
            # Connectivity metrics
            "orn_to_pn_strength": round(metrics.orn_to_pn_strength, 4),
            "kc_coverage": round(metrics.kc_coverage, 4),
            "alpha_beta_fraction": round(metrics.alpha_beta_fraction, 4),
            "gamma_fraction": round(metrics.gamma_fraction, 4),
            "mbon_diversity": metrics.mbon_diversity,
            "circuit_score": round(metrics.circuit_score, 4),
            "circuit_type": metrics.to_dict()["circuit_type"],
        }

        logger.info(
            f"✓ {receptor}: {len(orn_ids)} ORNs, {len(pathway.unique_pns)} PNs, "
            f"{len(pathway.unique_kcs)} KCs, score={metrics.circuit_score:.3f}"
        )

        return result

    except Exception as e:
        logger.error(f"Error processing {receptor}: {e}")
        return {
            "receptor": receptor,
            "status": "error",
            "n_orns": 0,
            "error": str(e)
        }


def build_complete_database(
    door_cache: str,
    flywire_labels: str,
    flywire_synapses: str,
    flywire_cell_types: str,
    output_dir: str
):
    """Build complete ORN-FlyWire connectivity database."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    logger.info("=" * 80)
    logger.info("BUILDING COMPLETE ORN-FLYWIRE CONNECTIVITY DATABASE")
    logger.info("=" * 80)

    # Get all DoOR receptors
    logger.info("\n[STEP 1] Loading all DoOR receptors...")
    all_receptors = get_all_door_receptors(door_cache)
    logger.info(f"Total receptors to process: {len(all_receptors)}")

    # Initialize FlyWire mapper
    logger.info("\n[STEP 2] Initializing FlyWire mapper...")
    mapper = FlyWireMapper(flywire_labels, auto_parse=True)

    # Initialize mushroom body tracer
    logger.info("\n[STEP 3] Initializing mushroom body tracer...")
    tracer = MushroomBodyTracer(
        synapse_path=flywire_synapses,
        cell_types_path=flywire_cell_types,
        min_synapse_threshold=1
    )

    # Process all receptors
    logger.info(f"\n[STEP 4] Processing {len(all_receptors)} receptors...")
    logger.info("This will take 3-5 hours depending on system performance.")
    logger.info("Progress will be saved incrementally.\n")

    results = []
    checkpoint_interval = 10  # Save every 10 receptors

    for i, receptor in enumerate(tqdm(all_receptors, desc="Processing receptors")):
        result = map_receptor_to_flywire(receptor, mapper, tracer)
        results.append(result)

        # Checkpoint save
        if (i + 1) % checkpoint_interval == 0:
            checkpoint_df = pd.DataFrame(results)
            checkpoint_df.to_csv(output_path / f"checkpoint_{i+1}.csv", index=False)
            logger.info(f"Checkpoint saved: {i+1}/{len(all_receptors)} receptors")

    # Create final database
    logger.info("\n[STEP 5] Creating final database...")
    db_df = pd.DataFrame(results)

    # Statistics
    successful = sum(1 for r in results if r["status"] == "success")
    not_found = sum(1 for r in results if r["status"] == "not_found")
    errors = sum(1 for r in results if r["status"] == "error")

    logger.info(f"\nDatabase Statistics:")
    logger.info(f"  Total receptors: {len(results)}")
    logger.info(f"  Successfully mapped: {successful} ({successful/len(results)*100:.1f}%)")
    logger.info(f"  Not found in FlyWire: {not_found}")
    logger.info(f"  Errors: {errors}")

    # Export CSV
    csv_path = output_path / "flywire_orn_complete_v1.0.csv"
    db_df.to_csv(csv_path, index=False)
    logger.info(f"\n✓ Saved CSV database: {csv_path}")

    # Export JSON (more structured)
    json_path = output_path / "flywire_orn_complete_v1.0.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"✓ Saved JSON database: {json_path}")

    # Generate statistics report
    stats_path = output_path / "database_statistics.txt"
    with open(stats_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("FLYWIRE ORN CONNECTIVITY DATABASE v1.0 - STATISTICS\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Total receptors processed: {len(results)}\n")
        f.write(f"Successfully mapped: {successful} ({successful/len(results)*100:.1f}%)\n")
        f.write(f"Not found in FlyWire: {not_found}\n")
        f.write(f"Errors: {errors}\n\n")

        # Successfully mapped receptors only
        success_df = db_df[db_df["status"] == "success"]

        if len(success_df) > 0:
            f.write("CONNECTIVITY SUMMARY (successfully mapped receptors)\n")
            f.write("-" * 80 + "\n")
            f.write(f"Average ORNs per receptor: {success_df['n_orns'].mean():.1f}\n")
            f.write(f"Average PNs contacted: {success_df['n_pns'].mean():.1f}\n")
            f.write(f"Average KCs contacted: {success_df['n_kcs'].mean():.1f}\n")
            f.write(f"Average circuit score: {success_df['circuit_score'].mean():.3f}\n\n")

            f.write("CIRCUIT TYPE DISTRIBUTION\n")
            f.write("-" * 80 + "\n")
            circuit_counts = success_df["circuit_type"].value_counts()
            for circuit, count in circuit_counts.items():
                f.write(f"{circuit}: {count} ({count/len(success_df)*100:.1f}%)\n")

            f.write("\nTOP 10 BY CIRCUIT SCORE\n")
            f.write("-" * 80 + "\n")
            top10 = success_df.nlargest(10, "circuit_score")[
                ["receptor", "circuit_score", "circuit_type", "n_kcs"]
            ]
            f.write(top10.to_string(index=False))

            f.write("\n\nTOP 10 BY KC COVERAGE\n")
            f.write("-" * 80 + "\n")
            top10_kc = success_df.nlargest(10, "kc_coverage")[
                ["receptor", "kc_coverage", "n_kcs", "circuit_type"]
            ]
            f.write(top10_kc.to_string(index=False))

    logger.info(f"✓ Saved statistics report: {stats_path}")

    # Create visualization
    logger.info("\n[STEP 6] Creating database visualization...")
    create_database_visualization(success_df, output_path / "database_visualization.png")

    logger.info("\n" + "=" * 80)
    logger.info("✓ DATABASE BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    logger.info(f"  1. {csv_path}")
    logger.info(f"  2. {json_path}")
    logger.info(f"  3. {stats_path}")
    logger.info(f"  4. {output_path / 'database_visualization.png'}")

    return db_df


def create_database_visualization(df: pd.DataFrame, output_path: Path):
    """Create overview visualization of database."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Circuit score distribution
    ax = axes[0, 0]
    ax.hist(df["circuit_score"], bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax.axvline(df["circuit_score"].mean(), color="red", linestyle="--",
               label=f"Mean: {df['circuit_score'].mean():.3f}")
    ax.set_xlabel("Circuit Score", fontsize=12)
    ax.set_ylabel("Number of Receptors", fontsize=12)
    ax.set_title("Circuit Score Distribution (All 78 Receptors)", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 2: Circuit type pie chart
    ax = axes[0, 1]
    circuit_counts = df["circuit_type"].value_counts()
    colors = ["#4CAF50", "#FF5722", "#9E9E9E"]
    ax.pie(circuit_counts, labels=circuit_counts.index, autopct="%1.1f%%",
           colors=colors, startangle=90)
    ax.set_title("Circuit Type Distribution", fontsize=14)

    # Plot 3: KC coverage vs ORN→PN strength
    ax = axes[1, 0]
    appetitive = df[df["circuit_type"] == "appetitive"]
    aversive = df[df["circuit_type"] == "aversive"]
    ax.scatter(appetitive["orn_to_pn_strength"], appetitive["kc_coverage"],
               alpha=0.6, s=100, c="green", label="Appetitive", edgecolors="k")
    ax.scatter(aversive["orn_to_pn_strength"], aversive["kc_coverage"],
               alpha=0.6, s=100, c="red", label="Aversive", edgecolors="k")
    ax.set_xlabel("ORN→PN Strength", fontsize=12)
    ax.set_ylabel("KC Coverage", fontsize=12)
    ax.set_title("Connectivity Metrics Scatter", fontsize=14)
    ax.legend()
    ax.grid(alpha=0.3)

    # Plot 4: Top 15 by circuit score
    ax = axes[1, 1]
    top15 = df.nlargest(15, "circuit_score")
    colors_bar = ["green" if ct == "appetitive" else "red"
                  for ct in top15["circuit_type"]]
    ax.barh(range(len(top15)), top15["circuit_score"], color=colors_bar,
            alpha=0.7, edgecolor="black")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(top15["receptor"], fontsize=10)
    ax.set_xlabel("Circuit Score", fontsize=12)
    ax.set_title("Top 15 Receptors by Circuit Score", fontsize=14)
    ax.grid(alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"✓ Saved visualization: {output_path}")


def main():
    """Main execution."""

    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    door_cache = base_dir / "door_cache"
    data_dir = base_dir / "data" / "flywire"
    output_dir = base_dir / "flywire_orn_database"

    # Check inputs exist
    required_files = {
        "DoOR cache": door_cache,
        "FlyWire labels": data_dir / "processed_labels.csv.gz",
        "FlyWire synapses": data_dir / "connections_princeton.csv.gz",
        "FlyWire cell types": data_dir / "consolidated_cell_types.csv.gz",
    }

    for name, path in required_files.items():
        if not path.exists():
            logger.error(f"Required file not found: {name} at {path}")
            return

    # Build database
    db_df = build_complete_database(
        door_cache=str(door_cache),
        flywire_labels=str(data_dir / "processed_labels.csv.gz"),
        flywire_synapses=str(data_dir / "connections_princeton.csv.gz"),
        flywire_cell_types=str(data_dir / "consolidated_cell_types.csv.gz"),
        output_dir=str(output_dir),
    )

    logger.info("\n🎉 Complete ORN-FlyWire database ready to use!")
    logger.info(f"\nNext steps:")
    logger.info(f"  1. Review: {output_dir / 'database_statistics.txt'}")
    logger.info(f"  2. Use database: {output_dir / 'flywire_orn_complete_v1.0.csv'}")
    logger.info(f"  3. Create lookup tools (see flywire_orn_tools.py)")
    logger.info(f"  4. Publish to GitHub!")


if __name__ == "__main__":
    main()
