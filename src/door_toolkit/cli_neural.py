"""
Neural Preprocessing CLI Commands
==================================

Command-line interface for neural network preprocessing features.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from door_toolkit.neural import DoORNeuralPreprocessor


def neural_main():
    """Main entry point for door-neural command."""
    parser = argparse.ArgumentParser(
        description="Neural network preprocessing tools for DoOR toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create sparse KC encoding
  door-neural --cache door_cache --sparse-encode --sparsity 0.05 \\
    --output sparse_data.npy

  # Generate noise-augmented dataset
  door-neural --cache door_cache --augment --n-augmentations 5 \\
    --output-dir augmented_data/

  # Export PGCN dataset
  door-neural --cache door_cache --export-pgcn --output-dir pgcn_dataset/ \\
    --format pytorch

  # Get dataset statistics
  door-neural --cache door_cache --stats

  # Create train/val split
  door-neural --cache door_cache --split --train-fraction 0.8 \\
    --output train_val_split.json
        """,
    )

    parser.add_argument(
        "--cache",
        type=Path,
        required=True,
        help="Path to DoOR cache directory",
    )

    parser.add_argument(
        "--sparse-encode",
        action="store_true",
        help="Create sparse KC-like encoding",
    )

    parser.add_argument(
        "--sparsity",
        type=float,
        default=0.05,
        help="Target sparsity level (default: 0.05)",
    )

    parser.add_argument(
        "--n-kc",
        type=int,
        default=2000,
        help="Number of KC neurons (default: 2000)",
    )

    parser.add_argument(
        "--augment",
        action="store_true",
        help="Generate noise-augmented dataset",
    )

    parser.add_argument(
        "--n-augmentations",
        type=int,
        default=5,
        help="Number of augmentations per sample (default: 5)",
    )

    parser.add_argument(
        "--noise-level",
        type=float,
        default=0.1,
        help="Noise level for augmentation (default: 0.1)",
    )

    parser.add_argument(
        "--export-pgcn",
        action="store_true",
        help="Export complete PGCN training dataset",
    )

    parser.add_argument(
        "--format",
        choices=["pytorch", "numpy", "h5"],
        default="pytorch",
        help="Export format (default: pytorch)",
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Display dataset statistics",
    )

    parser.add_argument(
        "--split",
        action="store_true",
        help="Create train/validation split",
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.8,
        help="Training fraction for split (default: 0.8)",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory path",
    )

    parser.add_argument(
        "--random-seed",
        type=int,
        help="Random seed for reproducibility",
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
    logger = logging.getLogger("door_toolkit.cli_neural")

    try:
        # Initialize preprocessor
        logger.info("Initializing DoORNeuralPreprocessor")
        preprocessor = DoORNeuralPreprocessor(
            str(args.cache),
            n_kc_neurons=args.n_kc,
            random_seed=args.random_seed,
        )

        # Create sparse encoding
        if args.sparse_encode:
            logger.info(f"Creating sparse encoding (sparsity={args.sparsity})")
            sparse_data = preprocessor.create_sparse_encoding(sparsity_level=args.sparsity)

            print(f"\nSparse Encoding Results:")
            print(f"Shape: {sparse_data.shape}")
            print(f"Sparsity: {(sparse_data > 0).mean():.2%}")
            print(f"Mean activation: {sparse_data[sparse_data > 0].mean():.3f}")
            print(f"Max activation: {sparse_data.max():.3f}")

            # Export if output specified
            if args.output:
                import numpy as np

                np.save(args.output, sparse_data)
                logger.info(f"Saved sparse encoding to {args.output}")
            elif args.output_dir:
                import numpy as np

                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / "sparse_encoding.npy", sparse_data)
                logger.info(f"Saved sparse encoding to {output_dir}")

            sys.exit(0)

        # Generate augmented dataset
        if args.augment:
            logger.info(f"Generating augmented dataset ({args.n_augmentations}x augmentation)")
            aug_orn, aug_kc, labels = preprocessor.generate_noise_augmented_responses(
                n_augmentations=args.n_augmentations,
                noise_level=args.noise_level,
            )

            print(f"\nAugmented Dataset:")
            print(f"Original samples: {len(set(labels))}")
            print(f"Augmented samples: {len(labels)}")
            print(f"ORN shape: {aug_orn.shape}")
            print(f"KC shape: {aug_kc.shape}")
            print(f"KC sparsity: {(aug_kc > 0).mean():.2%}")

            # Export
            if args.output_dir:
                import numpy as np

                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)

                np.save(output_dir / "augmented_orn.npy", aug_orn)
                np.save(output_dir / "augmented_kc.npy", aug_kc)

                with open(output_dir / "labels.txt", "w") as f:
                    f.write("\n".join(labels))

                logger.info(f"Saved augmented dataset to {output_dir}")
            else:
                logger.warning("No output directory specified, data not saved")

            sys.exit(0)

        # Export PGCN dataset
        if args.export_pgcn:
            if not args.output_dir:
                parser.error("--output-dir is required for --export-pgcn")

            logger.info(f"Exporting PGCN dataset to {args.output_dir}")
            preprocessor.export_pgcn_dataset(
                str(args.output_dir),
                format=args.format,
                include_sparse=True,
                include_metadata=True,
            )

            print(f"\nPGCN Dataset Exported:")
            print(f"Format: {args.format}")
            print(f"Output directory: {args.output_dir}")
            print(f"Files created:")

            output_dir = Path(args.output_dir)
            for file in sorted(output_dir.iterdir()):
                size_mb = file.stat().st_size / (1024 * 1024)
                print(f"  - {file.name} ({size_mb:.2f} MB)")

            sys.exit(0)

        # Display statistics
        if args.stats:
            logger.info("Computing dataset statistics")
            stats = preprocessor.get_dataset_statistics()

            print(f"\nDataset Statistics:")
            print(f"Number of odorants: {stats['n_odorants']}")
            print(f"Number of receptors: {stats['n_receptors']}")
            print(f"Mean response: {stats['mean_response']:.3f}")
            print(f"Std response: {stats['std_response']:.3f}")
            print(f"Mean receptor coverage: {stats['mean_receptor_coverage']:.1%}")
            print(f"Sparsity (>0.3): {stats['sparsity_at_threshold_0.3']:.2%}")
            print(f"Max response: {stats['max_response']:.3f}")
            print(f"Min response: {stats['min_response']:.3f}")

            if args.output:
                with open(args.output, "w") as f:
                    json.dump(stats, f, indent=2)
                logger.info(f"Saved statistics to {args.output}")

            sys.exit(0)

        # Create train/val split
        if args.split:
            logger.info(f"Creating train/validation split ({args.train_fraction:.1%} train)")
            train_odorants, val_odorants = preprocessor.create_training_validation_split(
                train_fraction=args.train_fraction,
                random_seed=args.random_seed,
            )

            print(f"\nTrain/Validation Split:")
            print(f"Training samples: {len(train_odorants)}")
            print(f"Validation samples: {len(val_odorants)}")
            print(
                f"Split ratio: {len(train_odorants) / (len(train_odorants) + len(val_odorants)):.2%} train"
            )

            # Show sample odorants
            print(f"\nSample training odorants:")
            for odor in train_odorants[:5]:
                print(f"  - {odor}")

            print(f"\nSample validation odorants:")
            for odor in val_odorants[:5]:
                print(f"  - {odor}")

            # Export
            if args.output:
                split_data = {
                    "train": train_odorants,
                    "validation": val_odorants,
                    "train_fraction": args.train_fraction,
                    "random_seed": args.random_seed,
                }
                with open(args.output, "w") as f:
                    json.dump(split_data, f, indent=2)
                logger.info(f"Saved split to {args.output}")

            sys.exit(0)

        # If no action specified, show help
        parser.print_help()

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.debug)
        sys.exit(1)


if __name__ == "__main__":
    neural_main()
