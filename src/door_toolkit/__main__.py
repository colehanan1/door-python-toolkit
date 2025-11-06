"""
Entry point for running door_toolkit as a module.

Usage:
    python -m door_toolkit --help
    python -m door_toolkit extract -i DoOR.data/data -o door_cache
"""

from door_toolkit.cli import extract_main

if __name__ == "__main__":
    extract_main()
