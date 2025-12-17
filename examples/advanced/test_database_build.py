"""
Test Database Build - Quick verification with 5 receptors
==========================================================

Test the database build system on a small subset to verify everything works.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from door_toolkit.flywire import FlyWireMapper
from door_toolkit.flywire.mushroom_body_tracer import MushroomBodyTracer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def map_receptor_to_flywire(receptor: str, mapper: FlyWireMapper, tracer: MushroomBodyTracer):
    """Map a single receptor to FlyWire connectivity (same as main script)."""
    try:
        # Step 1: Find ORN cells
        orn_cells = mapper.find_receptor_cells(receptor)

        if not orn_cells:
            return {
                "receptor": receptor,
                "status": "not_found",
                "n_orns": 0,
            }

        # Step 2: Trace pathway
        orn_ids = [cell["root_id"] for cell in orn_cells]
        pathway = tracer.trace_receptor_pathway(receptor, orn_ids)

        # Step 3: Calculate metrics
        metrics = tracer.calculate_connectivity_metrics(pathway)
        metrics_dict = metrics.to_dict()

        # Step 4: Combine results
        result = {
            "receptor": receptor,
            "status": "success",
            "n_orns": len(pathway.orn_ids),
            "n_pns": len(pathway.unique_pns),
            "n_kcs": len(pathway.unique_kcs),
            "n_mbons": 0,  # MBONs not tracked in pathway
            "orn_to_pn_synapses": pathway.total_orn_to_pn_synapses,
            "pn_to_kc_synapses": pathway.total_pn_to_kc_synapses,
            "kc_alpha_beta": pathway.kc_compartments.get("alpha_beta", 0),
            "kc_gamma": pathway.kc_compartments.get("gamma", 0),
            "kc_alpha_prime_beta_prime": pathway.kc_compartments.get("alpha_prime_beta_prime", 0),
            **metrics_dict,
        }

        return result

    except Exception as e:
        logger.error(f"Error processing {receptor}: {str(e)}")
        return {
            "receptor": receptor,
            "status": "error",
            "error": str(e),
        }


def main():
    """Test database build on 5 receptors."""

    # Paths
    base_dir = Path(__file__).resolve().parents[2]
    data_dir = base_dir / "data" / "flywire"
    output_dir = base_dir / "flywire_orn_database" / "test"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Test receptors (we know these work from previous analysis)
    test_receptors = ["Or22b", "Or85a", "Or42a", "Or49a", "Or46a"]

    logger.info("=" * 80)
    logger.info("DATABASE BUILD TEST - 5 RECEPTORS")
    logger.info("=" * 80)
    logger.info(f"Test receptors: {', '.join(test_receptors)}")

    # Initialize
    logger.info("\n[STEP 1] Initializing FlyWire mapper...")
    mapper = FlyWireMapper(str(data_dir / "processed_labels.csv.gz"), auto_parse=True)

    logger.info("[STEP 2] Initializing mushroom body tracer...")
    tracer = MushroomBodyTracer(
        synapse_path=str(data_dir / "connections_princeton.csv.gz"),
        cell_types_path=str(data_dir / "consolidated_cell_types.csv.gz"),
    )

    # Process receptors
    logger.info("\n[STEP 3] Processing test receptors...")
    results = []

    for receptor in test_receptors:
        logger.info(f"\nProcessing {receptor}...")
        result = map_receptor_to_flywire(receptor, mapper, tracer)
        results.append(result)

        if result["status"] == "success":
            logger.info(f"  ✓ {result['n_orns']} ORNs → {result['n_pns']} PNs → {result['n_kcs']} KCs")
            logger.info(f"  ✓ Circuit score: {result['circuit_score']:.3f} ({result['circuit_type']})")
        else:
            logger.info(f"  ✗ Status: {result['status']}")

    # Save results
    logger.info("\n[STEP 4] Saving test results...")

    # CSV
    df = pd.DataFrame(results)
    csv_path = output_dir / "test_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"  ✓ Saved CSV: {csv_path}")

    # JSON
    json_path = output_dir / "test_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  ✓ Saved JSON: {json_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEST COMPLETE!")
    logger.info("=" * 80)

    successful = sum(1 for r in results if r["status"] == "success")
    logger.info(f"\nResults: {successful}/{len(results)} receptors successfully mapped")

    if successful == len(results):
        logger.info("\n✓✓✓ ALL TESTS PASSED! ✓✓✓")
        logger.info("\nReady to run full database build:")
        logger.info("  python examples/advanced/build_complete_orn_database.py")
    else:
        logger.warning("\n⚠ Some tests failed. Check errors above.")

    return df


if __name__ == "__main__":
    main()
