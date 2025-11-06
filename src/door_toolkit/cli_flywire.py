"""
FlyWire CLI Commands
====================

Command-line interface for FlyWire integration features.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from door_toolkit.flywire import FlyWireMapper


def flywire_main():
    """Main entry point for door-flywire command."""
    parser = argparse.ArgumentParser(
        description="FlyWire integration tools for DoOR toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Map DoOR receptors to FlyWire cells
  door-flywire --labels data/flywire/processed_labels.csv.gz --cache door_cache --map-receptors

  # Find specific receptor cells
  door-flywire --labels data/flywire/processed_labels.csv.gz --find-receptor Or7a

  # Create spatial activation map
  door-flywire --labels data/flywire/processed_labels.csv.gz --cache door_cache \\
    --spatial-map "ethyl butyrate" --output spatial_map.json

  # Export receptor mappings
  door-flywire --labels data/flywire/processed_labels.csv.gz --cache door_cache \\
    --map-receptors --output flywire_mapping.json
        """,
    )

    parser.add_argument(
        "--labels",
        type=Path,
        required=True,
        help="Path to FlyWire community labels (CSV or CSV.GZ)",
    )

    parser.add_argument(
        "--cache",
        type=Path,
        help="Path to DoOR cache directory (required for some operations)",
    )

    parser.add_argument(
        "--map-receptors",
        action="store_true",
        help="Map all DoOR receptors to FlyWire cells",
    )

    parser.add_argument(
        "--find-receptor",
        type=str,
        help="Find FlyWire cells for specific receptor (e.g., Or42b)",
    )

    parser.add_argument(
        "--spatial-map",
        type=str,
        help="Create spatial activation map for odorant",
    )

    parser.add_argument(
        "--list-receptors",
        action="store_true",
        help="List all unique receptors found in community labels",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (JSON or CSV)",
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Output format (default: json)",
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
    logger = logging.getLogger("door_toolkit.cli_flywire")

    try:
        # Initialize mapper
        logger.info(f"Loading FlyWire community labels from {args.labels}")
        mapper = FlyWireMapper(
            str(args.labels), door_cache_path=str(args.cache) if args.cache else None
        )
        mapper.parse_labels(show_progress=True)

        # Find specific receptor
        if args.find_receptor:
            logger.info(f"Finding cells for receptor: {args.find_receptor}")
            cells = mapper.find_receptor_cells(args.find_receptor)

            print(f"\nFound {len(cells)} cells for {args.find_receptor}:")
            for i, cell in enumerate(cells, 1):
                print(f"\n{i}. Root ID: {cell['root_id']}")
                print(f"   Label: {cell['label']}")
                if "position" in cell:
                    pos = cell["position"]
                    print(f"   Position: ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f})")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump({"receptor": args.find_receptor, "cells": cells}, f, indent=2)
                logger.info(f"Saved results to {args.output}")

            sys.exit(0)

        # Map all receptors
        if args.map_receptors:
            if not args.cache:
                parser.error("--cache is required for --map-receptors")

            logger.info("Mapping all DoOR receptors to FlyWire cells")
            mappings = mapper.map_door_to_flywire(str(args.cache))

            print(f"\nMapping Results:")
            print(f"Total receptors: {len(mappings)}")
            print(f"Total cells: {sum(m.cell_count for m in mappings.values())}")

            # Show top 10 receptors by cell count
            print(f"\nTop 10 receptors by cell count:")
            sorted_mappings = sorted(mappings.items(), key=lambda x: x[1].cell_count, reverse=True)
            for receptor, mapping in sorted_mappings[:10]:
                print(f"  {receptor}: {mapping.cell_count} cells")

            # Export if output specified
            if args.output:
                mapper.export_mapping(str(args.output), format=args.format)
                logger.info(f"Exported mappings to {args.output}")

            # Show statistics
            stats = mapper.get_mapping_statistics()
            print(f"\nMapping Statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

            sys.exit(0)

        # Create spatial activation map
        if args.spatial_map:
            if not args.cache:
                parser.error("--cache is required for --spatial-map")

            logger.info(f"Creating spatial activation map for: {args.spatial_map}")

            # Ensure receptors are mapped first
            if not mapper.receptor_mappings:
                logger.info("Mapping receptors first...")
                mapper.map_door_to_flywire(str(args.cache))

            spatial_map = mapper.create_spatial_activation_map(args.spatial_map, str(args.cache))

            print(f"\nSpatial Activation Map: {args.spatial_map}")
            print(f"Active receptors: {len(spatial_map.receptor_activations)}")
            print(f"Spatial points: {spatial_map.total_cells}")

            # Show top receptors
            print(f"\nTop activated receptors:")
            sorted_receptors = sorted(
                spatial_map.receptor_activations.items(),
                key=lambda x: x[1],
                reverse=True,
            )
            for receptor, activation in sorted_receptors[:10]:
                print(f"  {receptor}: {activation:.3f}")

            # Export if output specified
            if args.output:
                output_data = spatial_map.to_dict()
                with open(args.output, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info(f"Saved spatial map to {args.output}")

            sys.exit(0)

        # List all unique receptors
        if args.list_receptors:
            logger.info("Extracting unique receptors from community labels")
            receptors = mapper.labels_parser.get_unique_receptors()

            print(f"\nFound {len(receptors)} unique receptors:")
            for receptor, count in sorted(receptors.items(), key=lambda x: -x[1])[:50]:
                print(f"  {receptor}: {count} cells")

            if len(receptors) > 50:
                print(f"  ... and {len(receptors) - 50} more")

            sys.exit(0)

        # If no action specified, show help
        parser.print_help()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    flywire_main()
