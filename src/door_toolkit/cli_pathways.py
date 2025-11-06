"""
Pathways CLI Commands
======================

Command-line interface for pathway analysis features.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from door_toolkit.pathways import (
    PathwayAnalyzer,
    BlockingExperimentGenerator,
    BehavioralPredictor,
)


def pathways_main():
    """Main entry point for door-pathways command."""
    parser = argparse.ArgumentParser(
        description="Pathway analysis tools for DoOR toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Trace Or47b feeding pathway
  door-pathways --cache door_cache --trace or47b-feeding

  # Trace Or42b pathway
  door-pathways --cache door_cache --trace or42b

  # Custom pathway tracing
  door-pathways --cache door_cache --custom-pathway \\
    --receptors Or92a --odorants geosmin --behavior avoidance

  # Compute Shapley importance scores
  door-pathways --cache door_cache --shapley feeding --output importance.json

  # Generate experiment protocol
  door-pathways --cache door_cache --generate-experiment 1 \\
    --output exp1_protocol.json

  # Predict behavior
  door-pathways --cache door_cache --predict-behavior "ethyl butyrate"
        """,
    )

    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to DoOR cache directory",
    )

    parser.add_argument(
        "--flywire-data",
        type=Path,
        help="Optional path to FlyWire community labels",
    )

    parser.add_argument(
        "--trace",
        choices=["or47b-feeding", "or42b", "or92a-avoidance"],
        help="Trace known pathway",
    )

    parser.add_argument(
        "--custom-pathway",
        action="store_true",
        help="Trace custom pathway (requires --receptors, --odorants, --behavior)",
    )

    parser.add_argument(
        "--receptors",
        nargs="+",
        help="Receptor names for custom pathway",
    )

    parser.add_argument(
        "--odorants",
        nargs="+",
        help="Odorant names for custom pathway",
    )

    parser.add_argument(
        "--behavior",
        type=str,
        help="Target behavior for custom pathway",
    )

    parser.add_argument(
        "--shapley",
        type=str,
        help="Compute Shapley importance for behavior",
    )

    parser.add_argument(
        "--find-blocking-targets",
        type=str,
        help="Find critical blocking targets for pathway",
    )

    parser.add_argument(
        "--generate-experiment",
        type=int,
        choices=[1, 2, 3, 6],
        help="Generate experiment protocol (1=veto, 2=microsurgery, 3=tagging, 6=blocker map)",
    )

    parser.add_argument(
        "--predict-behavior",
        type=str,
        help="Predict behavioral response to odorant",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path",
    )

    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format for experiments (default: json)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("door_toolkit.cli_pathways")

    try:
        # Initialize analyzer
        logger.info("Initializing PathwayAnalyzer")
        analyzer = PathwayAnalyzer(
            str(args.cache),
            str(args.flywire_data) if args.flywire_data else None,
        )

        # Trace known pathway
        if args.trace:
            logger.info(f"Tracing pathway: {args.trace}")

            if args.trace == "or47b-feeding":
                pathway = analyzer.trace_or47b_feeding_pathway()
            elif args.trace == "or42b":
                pathway = analyzer.trace_or42b_pathway()
            elif args.trace == "or92a-avoidance":
                pathway = analyzer.trace_custom_pathway(
                    receptors=["Or92a"],
                    odorants=["geosmin"],
                    behavior="avoidance",
                )

            # Display results
            print(f"\nPathway: {pathway.pathway_name}")
            print(f"Target Behavior: {pathway.target_behavior}")
            print(f"Strength: {pathway.strength:.3f}")
            print(f"\nReceptor Contributions:")
            for receptor, contrib in pathway.get_top_receptors(10):
                print(f"  {receptor}: {contrib:.3f}")

            # Export if output specified
            if args.output:
                with open(args.output, "w") as f:
                    json.dump(pathway.to_dict(), f, indent=2)
                logger.info(f"Saved pathway to {args.output}")

            sys.exit(0)

        # Trace custom pathway
        if args.custom_pathway:
            if not all([args.receptors, args.odorants, args.behavior]):
                parser.error("--custom-pathway requires --receptors, --odorants, and --behavior")

            logger.info("Tracing custom pathway")
            pathway = analyzer.trace_custom_pathway(
                receptors=args.receptors,
                odorants=args.odorants,
                behavior=args.behavior,
            )

            print(f"\nCustom Pathway: {pathway.pathway_name}")
            print(f"Target Behavior: {pathway.target_behavior}")
            print(f"Strength: {pathway.strength:.3f}")
            print(f"\nReceptor Contributions:")
            for receptor, contrib in pathway.get_top_receptors():
                print(f"  {receptor}: {contrib:.3f}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(pathway.to_dict(), f, indent=2)
                logger.info(f"Saved pathway to {args.output}")

            sys.exit(0)

        # Compute Shapley importance
        if args.shapley:
            logger.info(f"Computing Shapley importance for: {args.shapley}")
            importance = analyzer.compute_shapley_importance(args.shapley)

            print(f"\nShapley Importance Scores ({args.shapley}):")
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            for receptor, score in sorted_importance[:20]:
                print(f"  {receptor}: {score:.4f}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(importance, f, indent=2)
                logger.info(f"Saved importance scores to {args.output}")

            sys.exit(0)

        # Find blocking targets
        if args.find_blocking_targets:
            logger.info("Finding critical blocking targets")

            # Trace pathway first
            if args.find_blocking_targets == "or47b":
                pathway = analyzer.trace_or47b_feeding_pathway()
            elif args.find_blocking_targets == "or42b":
                pathway = analyzer.trace_or42b_pathway()
            else:
                pathway = None

            targets = analyzer.find_critical_blocking_targets(pathway)

            print(f"\nCritical Blocking Targets:")
            for target in targets:
                print(f"  - {target}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump({"targets": targets}, f, indent=2)

            sys.exit(0)

        # Generate experiment protocol
        if args.generate_experiment:
            logger.info(f"Generating experiment protocol {args.generate_experiment}")

            generator = BlockingExperimentGenerator(
                str(args.cache),
                str(args.flywire_data) if args.flywire_data else None,
            )

            if args.generate_experiment == 1:
                protocol = generator.generate_experiment_1_protocol()
            elif args.generate_experiment == 2:
                protocol = generator.generate_experiment_2_protocol()
            elif args.generate_experiment == 3:
                protocol = generator.generate_experiment_3_protocol()
            elif args.generate_experiment == 6:
                protocol = generator.generate_experiment_6_protocol()

            print(f"\nExperiment Protocol: {protocol.experiment_name}")
            print(f"ID: {protocol.experiment_id}")
            print(f"Hypothesis: {protocol.hypothesis}")
            print(f"\nSteps: {len(protocol.steps)}")
            for step in protocol.steps:
                print(f"  {step.step_number}. {step.action}")

            print(f"\nControls: {len(protocol.controls)}")
            for control in protocol.controls:
                print(f"  - {control}")

            # Export
            if args.output:
                if args.format == "json":
                    protocol.export_json(str(args.output))
                elif args.format == "markdown":
                    protocol.export_markdown(str(args.output))
                logger.info(f"Saved protocol to {args.output}")
            else:
                # Default output path
                output_path = Path(f"experiment_{args.generate_experiment}_protocol.json")
                protocol.export_json(str(output_path))
                logger.info(f"Saved protocol to {output_path}")

            sys.exit(0)

        # Predict behavior
        if args.predict_behavior:
            logger.info(f"Predicting behavior for: {args.predict_behavior}")

            predictor = BehavioralPredictor(str(args.cache))
            prediction = predictor.predict_behavior(args.predict_behavior)

            print(f"\nBehavioral Prediction: {args.predict_behavior}")
            print(f"Predicted Valence: {prediction.predicted_valence}")
            print(f"Confidence: {prediction.confidence:.2%}")
            print(f"\nKey Contributing Receptors:")
            for receptor, contrib in prediction.key_contributors:
                print(f"  {receptor}: {contrib:.3f}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(prediction.to_dict(), f, indent=2)
                logger.info(f"Saved prediction to {args.output}")

            sys.exit(0)

        # If no action specified, show help
        parser.print_help()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    pathways_main()
