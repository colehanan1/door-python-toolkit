"""
Command-line interface for DoOR toolkit.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

from door_toolkit.data.receptor_groups import RECEPTOR_GROUPS
from door_toolkit.encoder import DoOREncoder
from door_toolkit.extractor import DoORExtractor
from door_toolkit.utils import (
    load_response_matrix,
    load_odor_metadata,
    list_odorants,
    validate_cache,
)


def build_csv_path(base: Path, descriptor: str) -> Path:
    descriptor = descriptor.replace(" ", "-")
    base = Path(base)
    if base.suffix:
        return base.with_name(f"{base.stem}-{descriptor}{base.suffix}")
    return base.with_name(f"{base.name}-{descriptor}.csv")


def write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8") as fh:
        fh.write(",".join(fieldnames) + "\n")
        for row in rows:
            fh.write(",".join(row.get(field, "") for field in fieldnames) + "\n")


def extract_main():
    """Main entry point for door-extract command."""
    parser = argparse.ArgumentParser(
        description="Extract DoOR R data to Python formats",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic extraction
  door-extract --input DoOR.data/data --output door_cache

  # With debug logging
  door-extract -i DoOR.data/data -o cache --debug

  # Validate existing cache
  door-extract --validate door_cache

  # Encode a single odorant
  door-extract --cache door_cache --odor \"ethyl butyrate\"
        """,
    )

    parser.add_argument(
        "-i", "--input", type=Path, help="Path to DoOR.data/data directory containing .RData files"
    )

    parser.add_argument("-o", "--output", type=Path, help="Output directory for cached data")

    parser.add_argument("--validate", type=Path, help="Validate existing cache directory")

    parser.add_argument("--cache", type=Path, help="Existing cache directory for queries")

    parser.add_argument(
        "--odor",
        type=str,
        help="Encode an odorant from the cache and print the receptor response vector",
    )

    parser.add_argument(
        "--odors",
        type=str,
        nargs="+",
        help="Encode multiple odorants and compare their receptor responses",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        help="Optional substring filter when used with --list-odorants",
    )

    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Limit output rows when using --receptor (default: 20; use 0 to show all)",
    )

    parser.add_argument(
        "--receptor",
        type=str,
        help="Show the response profile for a receptor or neuron type (e.g., Or42b)",
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="When used with --odor, display receptor coverage statistics",
    )

    parser.add_argument("--list-odorants", type=Path, help="List odorants in cache directory")
    parser.add_argument(
        "--save",
        type=Path,
        help="Optional path prefix for CSV exports (files get receptor/odor suffixes)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level, format="[%(asctime)s] %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    logger = logging.getLogger("door_toolkit.cli")

    def abort(message: str, exc: Optional[Exception] = None, code: int = 1) -> None:
        """Print an error and exit, emitting tracebacks when --debug is provided."""
        if exc and args.debug:
            logger.exception(message)
        else:
            print(message)
            if exc and not args.debug:
                print("Hint: re-run with --debug for a full traceback.")
        sys.exit(code)

    # Validate cache
    if args.validate:
        print(f"Validating cache: {args.validate}")
        is_valid = validate_cache(str(args.validate))
        if is_valid:
            print("✓ Cache is valid")
            sys.exit(0)
        else:
            print("✗ Cache validation failed")
            sys.exit(1)

    # List odorants
    if args.list_odorants:
        try:
            logger.debug(
                "Listing odorants from cache=%s pattern=%s",
                args.list_odorants,
                args.pattern,
            )
            odors = list_odorants(str(args.list_odorants), pattern=args.pattern)
            print(f"Found {len(odors)} odorants:")
            for odor in odors:
                print(f"  - {odor}")
            sys.exit(0)
        except Exception as e:
            abort("✗ Failed to list odorants", e)

    if args.receptor:
        cache_path = args.cache or args.list_odorants or args.validate or args.output
        if not cache_path:
            parser.error("--cache is required when using --receptor")

        logger.debug(
            "Computing receptor profile for '%s' using cache %s", args.receptor, cache_path
        )

        try:
            response_df = load_response_matrix(str(cache_path))
        except Exception as exc:
            abort("✗ Failed to load cache", exc)

        receptor_key = args.receptor

        if receptor_key.lower() in RECEPTOR_GROUPS:
            group = RECEPTOR_GROUPS[receptor_key.lower()]
            logger.debug("Using receptor group '%s' with %d members", receptor_key, len(group))
            receptors = [r for r in response_df.columns if r in group]
            if not receptors:
                abort(f"✗ No receptors from group '{receptor_key}' found in cache")

            meta = load_odor_metadata(str(cache_path))
            limit = args.top if args.top and args.top > 0 else None
            for receptor_name in receptors:
                csv_rows = []
                print(f"\nResponse profile for {receptor_name}")
                print(f"{'Odorant':40} Response")
                print("-" * 55)
                series = response_df[receptor_name].dropna().sort_values(ascending=False)
                logger.debug(
                    "Loaded %d responses for receptor %s (limit=%s)",
                    len(series),
                    receptor_name,
                    limit,
                )
                limited_series = series.head(limit) if limit else series
                for inchikey, value in limited_series.items():
                    odor_name = meta.loc[inchikey]["Name"] if inchikey in meta.index else inchikey
                    print(f"{str(odor_name)[:40]:40} {value:0.4f}")
                    csv_rows.append({"odorant": str(odor_name), "response": f"{value:0.4f}"})
                if limit:
                    if len(series) > limit:
                        print(
                            f"... ({len(series) - limit} more odorants; re-run with --top 0 to show all)"
                        )
                    print("Lowest responses:")
                    tail_series = series.tail(limit)
                    for inchikey, value in tail_series.items():
                        odor_name = (
                            meta.loc[inchikey]["Name"] if inchikey in meta.index else inchikey
                        )
                        print(f"  {str(odor_name)[:37]:37} {value:0.4f}")
                        csv_rows.append({"odorant": f"bottom-{str(odor_name)}", "response": f"{value:0.4f}"})
                if args.save:
                    output_path = build_csv_path(args.save, f"{receptor_name}")
                    write_csv(output_path, csv_rows)
            sys.exit(0)

        receptor_name = receptor_key
        matching_cols = [col for col in response_df.columns if col.lower() == receptor_name.lower()]
        if not matching_cols:
            # Allow partial match against known receptor labels
            partial = [col for col in response_df.columns if receptor_name.lower() in col.lower()]
            if len(partial) == 1:
                matching_cols = partial
                receptor_name = partial[0]
            else:
                message = f"✗ Receptor '{args.receptor}' not found in cache"
                if partial:
                    print(message)
                    print("Did you mean:")
                    for candidate in partial:
                        print(f"  - {candidate}")
                    sys.exit(1)
                abort(message)

        receptor_name = matching_cols[0]
        logger.debug("Displaying receptor profile for %s", receptor_name)
        series = response_df[receptor_name].dropna().sort_values(ascending=False)

        if series.empty:
            print(f"Receptor {receptor_name} has no recorded responses.")
            sys.exit(0)

        meta = load_odor_metadata(str(cache_path))
        print(f"Response profile for {receptor_name}")
        print(f"{'Odorant':40} Response")
        print("-" * 55)
        limit = args.top if args.top and args.top > 0 else None
        limited_series = series.head(limit) if limit else series
        csv_rows = []
        for inchikey, value in limited_series.items():
            odor_name = meta.loc[inchikey]["Name"] if inchikey in meta.index else inchikey
            print(f"{str(odor_name)[:40]:40} {value:0.4f}")
            csv_rows.append({"odorant": str(odor_name), "response": f"{value:0.4f}"})
        if limit:
            if len(series) > limit:
                print(f"... ({len(series) - limit} more odorants; re-run with --top 0 to show all)")
            print("Lowest responses:")
            tail_series = series.tail(limit)
            for inchikey, value in tail_series.items():
                odor_name = meta.loc[inchikey]["Name"] if inchikey in meta.index else inchikey
                print(f"  {str(odor_name)[:37]:37} {value:0.4f}")
                csv_rows.append({"odorant": f"bottom-{str(odor_name)}", "response": f"{value:0.4f}"})
        if args.save:
            output_path = build_csv_path(args.save, receptor_name)
            write_csv(output_path, csv_rows)

        sys.exit(0)

    if args.odor or args.odors:
        cache_path = args.cache or args.list_odorants or args.validate or args.output
        if not cache_path:
            parser.error("--cache is required when using --odor/--odors")

        try:
            target_odors = args.odors if args.odors else [args.odor]
            logger.debug("Encoding odors %s using cache %s", target_odors, cache_path)
            encoder = DoOREncoder(str(cache_path), use_torch=False)
            matrix = encoder.encode(target_odors)
        except Exception as exc:
            abort("✗ Failed to encode odorant(s)", exc)

        import numpy as np

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        matrix = np.asarray(matrix, dtype=float)
        limit = args.top if args.top and args.top > 0 else None

        stats_rows = []
        for idx, receptor in enumerate(encoder.receptor_names):
            values = matrix[:, idx]
            stats_rows.append(
                {
                    "receptor": receptor,
                    "values": values,
                    "max": values.max(),
                    "min": values.min(),
                    "spread": values.max() - values.min(),
                }
            )

        tables = []
        if matrix.shape[0] > 1:
            spread_sorted = sorted(stats_rows, key=lambda r: r["spread"], reverse=True)
            min_sorted = sorted(stats_rows, key=lambda r: r["min"])
            tables.append(("Max Spread Receptors", spread_sorted))
            tables.append(("Min Response Receptors", min_sorted))
        else:
            max_sorted = sorted(stats_rows, key=lambda r: r["max"], reverse=True)
            min_sorted = sorted(stats_rows, key=lambda r: r["min"])
            tables.append(("Highest Responses", max_sorted))
            tables.append(("Lowest Responses", min_sorted))

        print(f"Cache: {Path(cache_path).resolve()}")
        print(f"Receptors compared: {matrix.shape[1]}")

        for title, rows in tables:
            truncated_rows = rows[:limit] if limit else rows
            header = ["Receptor"] + [f"{odor}" for odor in target_odors]
            if matrix.shape[0] > 1:
                header.append("Δ(max-min)")

            print(f"\n{title}")
            print(" | ".join(f"{h:<15}" for h in header))
            print("-" * (len(header) * 18))

            csv_rows = []
            for row in truncated_rows:
                cells = [f"{row['receptor']:<15}"] + [f"{val:0.4f}" for val in row["values"]]
                if matrix.shape[0] > 1:
                    cells.append(f"{row['spread']:0.4f}")
                print(" | ".join(cells))
                csv_entry = {"receptor": row["receptor"].replace(" ", "-")}
                for odor, val in zip(target_odors, row["values"]):
                    csv_entry[odor.replace(" ", "-")] = f"{val:0.4f}"
                if matrix.shape[0] > 1:
                    csv_entry["spread"] = f"{row['spread']:0.4f}"
                csv_rows.append(csv_entry)

            if limit and len(rows) > len(truncated_rows):
                print(
                    f"... ({len(rows) - len(truncated_rows)} more receptors; re-run with --top 0 to show all)"
                )

            if args.save:
                suffix = title.lower().replace(" ", "-")
                output_path = build_csv_path(args.save, suffix)
                write_csv(output_path, csv_rows)

        if args.coverage:
            print("\nCoverage statistics:")
            for odor in target_odors:
                try:
                    coverage = encoder.get_receptor_coverage(odor)
                except KeyError as exc:
                    print(f"  {odor}: not found ({exc})")
                    continue

                print(f"  {odor}:")
                for key, value in coverage.items():
                    label = key.replace("_", " ").title()
                    if isinstance(value, dict):
                        print(f"    {label}:")
                        for receptor, resp in value.items():
                            print(f"      {receptor:<12} {resp:0.4f}")
                    else:
                        print(f"    {label}: {value}")

        sys.exit(0)

    # Extract
    if not args.input or not args.output:
        parser.error("--input and --output are required for extraction")

    try:
        extractor = DoORExtractor(args.input, args.output)
        extractor.run()
        print(f"\n✓ Extraction complete! Cache: {args.output.absolute()}")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    extract_main()
