"""
LASSO Behavioral Prediction Demo
=================================

Complete example demonstrating LASSO regression-based behavioral prediction
from optogenetic manipulation data.

This script:
1. Loads behavioral PER data from optogenetic experiments
2. Matches odorant names to DoOR receptor profiles
3. Fits LASSO models to identify sparse receptor circuits
4. Generates visualizations and exports results

Example usage:
    python examples/lasso_behavioral_prediction_demo.py
"""

import logging
from pathlib import Path

from door_toolkit.pathways import LassoBehavioralPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run complete LASSO behavioral prediction demo."""

    # ========================================================================
    # Configuration
    # ========================================================================

    # Paths (adjust these to your environment)
    DOORCACHE_PATH = "door_cache"  # Path to DoOR cache directory
    BEHAVIOR_CSV_PATH = (
        "/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions(Strictest)/"
        "reaction_rates_summary_unordered.csv"
    )
    OUTPUT_DIR = "behavioral_prediction_results"

    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # Initialize LASSO Behavioral Predictor
    # ========================================================================

    logger.info("=" * 80)
    logger.info("LASSO Behavioral Prediction Demo")
    logger.info("=" * 80)

    predictor = LassoBehavioralPredictor(
        doorcache_path=DOORCACHE_PATH,
        behavior_csv_path=BEHAVIOR_CSV_PATH,
        scale_features=True,  # Standardize receptor features
        scale_targets=False,  # Keep PER in 0-1 range
    )

    logger.info(f"Loaded behavioral data: {predictor.behavioral_data.shape}")
    logger.info(f"DoOR receptors: {predictor.encoder.n_channels}")
    logger.info("")

    # ========================================================================
    # Example 1: Fit model for opto_hex condition
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Example 1: opto_hex - Trained on Hexanol")
    logger.info("=" * 80)

    results_hex = predictor.fit_behavior(
        condition_name="opto_hex",
        lambda_range=[0.0001, 0.001, 0.01, 0.1, 1.0],  # Lambda values to try
        cv_folds=5,  # 5-fold cross-validation
        prediction_mode="test_odorant",  # Use test odorant receptor profiles
    )

    # Print summary
    print(results_hex.summary())

    # Generate plots
    results_hex.plot_predictions(save_to=str(output_path / "opto_hex_predictions.png"))
    results_hex.plot_receptors(save_to=str(output_path / "opto_hex_receptors.png"), n_top=15)

    # Export results
    results_hex.export_csv(str(output_path / "opto_hex_results.csv"))
    results_hex.export_json(str(output_path / "opto_hex_model.json"))

    logger.info(f"✓ opto_hex complete: R² = {results_hex.cv_r2_score:.3f}")
    logger.info("")

    # ========================================================================
    # Example 2: Fit model for opto_EB condition
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Example 2: opto_EB - Trained on Ethyl Butyrate")
    logger.info("=" * 80)

    results_eb = predictor.fit_behavior(
        condition_name="opto_EB",
        lambda_range=[0.0001, 0.001, 0.01, 0.1, 1.0],
        cv_folds=5,
        prediction_mode="test_odorant",
    )

    print(results_eb.summary())
    results_eb.plot_predictions(save_to=str(output_path / "opto_EB_predictions.png"))
    results_eb.plot_receptors(save_to=str(output_path / "opto_EB_receptors.png"), n_top=15)
    results_eb.export_csv(str(output_path / "opto_EB_results.csv"))
    results_eb.export_json(str(output_path / "opto_EB_model.json"))

    logger.info(f"✓ opto_EB complete: R² = {results_eb.cv_r2_score:.3f}")
    logger.info("")

    # ========================================================================
    # Example 3: Fit model for opto_benz_1 condition
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Example 3: opto_benz_1 - Trained on Benzaldehyde")
    logger.info("=" * 80)

    results_benz = predictor.fit_behavior(
        condition_name="opto_benz_1",
        lambda_range=[0.0001, 0.001, 0.01, 0.1, 1.0],
        cv_folds=5,
        prediction_mode="test_odorant",
    )

    print(results_benz.summary())
    results_benz.plot_predictions(save_to=str(output_path / "opto_benz_1_predictions.png"))
    results_benz.plot_receptors(save_to=str(output_path / "opto_benz_1_receptors.png"), n_top=15)
    results_benz.export_csv(str(output_path / "opto_benz_1_results.csv"))
    results_benz.export_json(str(output_path / "opto_benz_1_model.json"))

    logger.info(f"✓ opto_benz_1 complete: R² = {results_benz.cv_r2_score:.3f}")
    logger.info("")

    # ========================================================================
    # Example 4: Compare multiple conditions
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Example 4: Compare Multiple Conditions")
    logger.info("=" * 80)

    comparison_results = predictor.compare_conditions(
        conditions=["opto_hex", "opto_EB", "opto_benz_1", "opto_ACV", "opto_3-oct"],
        lambda_range=[0.0001, 0.001, 0.01, 0.1, 1.0],
        plot=True,
        save_dir=str(output_path / "comparison"),
    )

    logger.info("Comparison Summary:")
    for condition, result in comparison_results.items():
        logger.info(
            f"  {condition:20s}  R² = {result.cv_r2_score:6.3f}  "
            f"Receptors = {result.n_receptors_selected:3d}  "
            f"λ = {result.lambda_value:.4f}"
        )

    logger.info("")

    # ========================================================================
    # Example 5: Analyze receptor overlap across conditions
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Example 5: Receptor Overlap Analysis")
    logger.info("=" * 80)

    # Find common receptors across all conditions
    all_receptors = set()
    for result in comparison_results.values():
        all_receptors.update(result.lasso_weights.keys())

    logger.info(f"Total unique receptors selected: {len(all_receptors)}")

    # Find receptors appearing in multiple conditions
    receptor_counts = {}
    for result in comparison_results.values():
        for receptor in result.lasso_weights.keys():
            receptor_counts[receptor] = receptor_counts.get(receptor, 0) + 1

    # Receptors appearing in 2+ conditions
    shared_receptors = {r: c for r, c in receptor_counts.items() if c >= 2}
    logger.info(f"Receptors appearing in 2+ conditions: {len(shared_receptors)}")

    if shared_receptors:
        logger.info("\nShared receptors:")
        for receptor, count in sorted(shared_receptors.items(), key=lambda x: -x[1]):
            logger.info(f"  {receptor:12s}  appears in {count} conditions")

    logger.info("")

    # ========================================================================
    # Example 6: Explore different prediction modes
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Example 6: Compare Prediction Modes for opto_hex")
    logger.info("=" * 80)

    modes = ["test_odorant", "trained_odorant", "interaction"]
    mode_results = {}

    for mode in modes:
        try:
            result = predictor.fit_behavior(
                condition_name="opto_hex",
                lambda_range=[0.0001, 0.001, 0.01, 0.1, 1.0],
                cv_folds=5,
                prediction_mode=mode,
            )
            mode_results[mode] = result
            logger.info(
                f"  {mode:20s}  R² = {result.cv_r2_score:6.3f}  "
                f"Receptors = {result.n_receptors_selected:3d}"
            )
        except Exception as e:
            logger.warning(f"  {mode:20s}  FAILED: {e}")

    # Save best mode results
    if mode_results:
        best_mode = max(mode_results.items(), key=lambda x: x[1].cv_r2_score)
        logger.info(f"\nBest prediction mode: {best_mode[0]} (R² = {best_mode[1].cv_r2_score:.3f})")

    logger.info("")

    # ========================================================================
    # Summary
    # ========================================================================

    logger.info("=" * 80)
    logger.info("Demo Complete!")
    logger.info("=" * 80)
    logger.info(f"Results saved to: {output_path.absolute()}")
    logger.info("")
    logger.info("Generated files:")
    logger.info("  - Individual condition results (CSV/JSON)")
    logger.info("  - Prediction plots (actual vs predicted PER)")
    logger.info("  - Receptor importance plots (LASSO weights)")
    logger.info("  - Comparison plots (R² across conditions, receptor overlap)")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Review top receptors for each condition")
    logger.info("  2. Validate predictions with independent experiments")
    logger.info("  3. Design optogenetic experiments targeting identified receptors")
    logger.info("  4. Compare with FlyWire connectivity data")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
