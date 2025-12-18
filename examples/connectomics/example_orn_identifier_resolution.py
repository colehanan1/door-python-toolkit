"""
Demonstration of ORN/Glomerulus Identifier Resolution
======================================================

This example demonstrates the robust identifier resolution system that allows
users to pass ORN/glomerulus names in various formats.

The resolver automatically normalizes messy inputs like:
- "DL3", "dl3", "ORN_DL3", "ORN-DL3", "Glomerulus DL3"
- "Ir31a", "IR31A", "ORN_Ir31a"
- "Or7a", "OR7A", "ORN_Or7a"

All resolve to their canonical forms (e.g., "ORN_DL3", "ORN_Ir31a").
"""

from pathlib import Path
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.connectomics.pathway_analysis import (
    analyze_single_orn,
    compare_orn_pair,
    find_pathways,
)
from door_toolkit.integration.orn_identifier import (
    normalize_orn_identifier,
    resolve_orn_identifier,
    suggest_orn_identifiers,
    get_available_glomeruli,
)


def main():
    # Load network
    data_path = Path(__file__).parent.parent.parent / "data"
    csv_file = data_path / "interglomerular_crosstalk_pathways.csv"

    if not csv_file.exists():
        print(f"❌ Data file not found: {csv_file}")
        print("Please run the FlyWire extraction script first.")
        return

    print("=" * 70)
    print("ORN/Glomerulus Identifier Resolution Demo")
    print("=" * 70)
    print()

    # Load connectome network
    print("Loading FlyWire connectome network...")
    network = CrossTalkNetwork.from_csv(str(csv_file))
    print(f"✓ Loaded network with {network.data.num_glomeruli} glomeruli\n")

    # Get available glomeruli
    available = get_available_glomeruli(network)
    print(f"Available glomeruli: {len(available)} total")
    print(f"Sample: {sorted(list(available))[:10]}\n")

    print("=" * 70)
    print("1. NORMALIZATION EXAMPLES")
    print("=" * 70)
    print()

    # Demonstrate normalization
    test_inputs = [
        "DL3",
        "dl3",
        "ORN_DL3",
        "ORN-DL3",
        "ORN DL3",
        "Glomerulus DL3",
        "Ir31a",
        "IR31A",
        "ORN_Ir31a",
        "Or7a",
    ]

    print("Normalization (format agnostic):")
    for raw in test_inputs:
        normalized = normalize_orn_identifier(raw)
        print(f"  {raw:20s} → {normalized}")
    print()

    print("=" * 70)
    print("2. RESOLUTION WITH FUZZY MATCHING")
    print("=" * 70)
    print()

    # Test resolution with actual network
    print("Resolving identifiers against FlyWire network:\n")

    test_cases = ["DL3", "dl5", "Ir31a", "Or7a", "va1d"]

    for identifier in test_cases:
        try:
            resolved = resolve_orn_identifier(identifier, available)
            print(f"✓ '{identifier}' → '{resolved}'")
        except ValueError as e:
            print(f"✗ '{identifier}' failed: {e}")
    print()

    print("=" * 70)
    print("3. FUZZY MATCHING WITH SUGGESTIONS")
    print("=" * 70)
    print()

    # Test fuzzy matching
    print("When exact match fails, suggestions are provided:\n")

    typo_input = "DL33"  # User meant DL3
    suggestions = suggest_orn_identifiers(typo_input, available, k=5)

    print(f"Input: '{typo_input}'")
    print("Suggestions:")
    for identifier, score in suggestions:
        print(f"  {identifier:20s} (similarity: {score:.2f})")
    print()

    print("=" * 70)
    print("4. INTEGRATION WITH PATHWAY ANALYSIS")
    print("=" * 70)
    print()

    # Demonstrate that pathway analysis functions now accept messy inputs
    print("analyze_single_orn() now accepts various formats:\n")

    # Test with different input formats - all should work
    formats_to_test = ["DL5", "ORN_DL5", "dl5"]

    for fmt in formats_to_test:
        try:
            result = analyze_single_orn(network, fmt, by_glomerulus=True)
            print(f"✓ analyze_single_orn(network, '{fmt}')")
            print(f"    Found {result.num_pathways} pathways from {fmt}")
        except Exception as e:
            print(f"✗ '{fmt}' failed: {e}")
    print()

    print("=" * 70)
    print("5. COMPARE ORN PAIR WITH MESSY INPUTS")
    print("=" * 70)
    print()

    # Test pair comparison with messy inputs
    print("compare_orn_pair() with different input styles:\n")

    try:
        comparison = compare_orn_pair(
            network,
            "dl5",      # lowercase
            "VA1v",     # mixed case
            by_glomerulus=True
        )
        print(f"✓ compare_orn_pair(network, 'dl5', 'VA1v')")
        print(f"    DL5 → VA1v: {len(comparison.pathways_1_to_2)} pathways")
        print(f"    VA1v → DL5: {len(comparison.pathways_2_to_1)} pathways")
        print(f"    Asymmetry ratio: {comparison.get_asymmetry_ratio():.2f}")
    except Exception as e:
        print(f"✗ Comparison failed: {e}")
    print()

    print("=" * 70)
    print("6. ERROR HANDLING WITH SUGGESTIONS")
    print("=" * 70)
    print()

    # Demonstrate error handling
    print("When identifier cannot be resolved, helpful error is raised:\n")

    try:
        resolve_orn_identifier("XYZ999", available)
    except ValueError as e:
        print(f"Error message:\n{e}\n")

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print("✓ Identifier resolution system successfully handles:")
    print("  - Case-insensitive inputs (DL3, dl3, Dl3)")
    print("  - Multiple separator styles (ORN_DL3, ORN-DL3, ORN DL3)")
    print("  - Prefix variations (DL3, ORN_DL3, Glomerulus DL3)")
    print("  - Receptor names (Ir31a, Or7a, Gr21a)")
    print("  - Fuzzy matching for typos (DL33 → suggests DL3, DL5)")
    print()
    print("✓ All pathway analysis functions now accept messy inputs")
    print("✓ Clear error messages with suggestions when resolution fails")
    print()


if __name__ == "__main__":
    main()
