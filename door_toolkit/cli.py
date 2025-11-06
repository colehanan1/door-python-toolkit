"""
Command-line interface for DoOR toolkit.
"""

import argparse
import sys
import logging
from pathlib import Path

from door_toolkit.extractor import DoORExtractor
from door_toolkit.utils import validate_cache, list_odorants


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
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=Path,
        help='Path to DoOR.data/data directory containing .RData files'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Output directory for cached data'
    )
    
    parser.add_argument(
        '--validate',
        type=Path,
        help='Validate existing cache directory'
    )
    
    parser.add_argument(
        '--list-odorants',
        type=Path,
        help='List odorants in cache directory'
    )
    
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='[%(asctime)s] %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
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
            odors = list_odorants(str(args.list_odorants))
            print(f"Found {len(odors)} odorants:")
            for odor in odors:
                print(f"  - {odor}")
            sys.exit(0)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
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


if __name__ == '__main__':
    extract_main()
