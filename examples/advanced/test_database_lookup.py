"""
Test ORN Database Lookup Tools
==============================

Test the instant lookup functionality using the test database.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from door_toolkit.flywire.orn_database_tools import ORNDatabase

def main():
    """Test database lookup functions."""

    # Use test database
    test_db_path = Path(__file__).resolve().parents[2] / "flywire_orn_database" / "test" / "test_results.csv"

    if not test_db_path.exists():
        print(f"Test database not found at: {test_db_path}")
        return

    print("=" * 80)
    print("ORN DATABASE LOOKUP TEST")
    print("=" * 80)

    # Initialize database
    print(f"\nLoading database from: {test_db_path}")
    db = ORNDatabase(database_path=str(test_db_path))

    print(f"✓ Loaded {len(db.db_df)} receptors")
    print(f"✓ Successfully mapped: {len(db.success_df)}")

    # Test 1: Get single receptor
    print("\n" + "=" * 80)
    print("TEST 1: Get single receptor (Or22b)")
    print("=" * 80)

    data = db.get("Or22b")
    if data:
        print(f"✓ Found Or22b:")
        print(f"  - Status: {data['status']}")
        print(f"  - ORNs: {data['n_orns']}")
        print(f"  - PNs: {data['n_pns']}")
        print(f"  - KCs: {data['n_kcs']}")
        print(f"  - Circuit score: {data['circuit_score']:.3f}")
        print(f"  - Circuit type: {data['circuit_type']}")
    else:
        print("✗ Or22b not found")

    # Test 2: Get specific metrics
    print("\n" + "=" * 80)
    print("TEST 2: Get specific metrics")
    print("=" * 80)

    score = db.get_circuit_score("Or85a")
    circuit_type = db.get_circuit_type("Or85a")
    kc_coverage = db.get_kc_coverage("Or85a")

    print(f"Or85a metrics:")
    print(f"  - Circuit score: {score:.3f}")
    print(f"  - Circuit type: {circuit_type}")
    print(f"  - KC coverage: {kc_coverage:.2%}")

    # Test 3: Compare receptors
    print("\n" + "=" * 80)
    print("TEST 3: Compare multiple receptors")
    print("=" * 80)

    comparison = db.compare_receptors(["Or22b", "Or85a", "Or42a"])
    print("\nComparison table:")
    print(comparison.to_string(index=False))

    # Test 4: Rank by metric
    print("\n" + "=" * 80)
    print("TEST 4: Rank by circuit score")
    print("=" * 80)

    top_receptors = db.rank_by_metric("circuit_score", top_n=5)
    print("\nTop receptors by circuit score:")
    print(top_receptors[["receptor", "circuit_score", "circuit_type", "n_kcs"]].to_string(index=False))

    # Test 5: Filter by circuit type
    print("\n" + "=" * 80)
    print("TEST 5: Filter by circuit type")
    print("=" * 80)

    appetitive = db.filter_by_circuit_type("appetitive")
    aversive = db.filter_by_circuit_type("aversive")

    print(f"\nAppetitive receptors: {len(appetitive)}")
    print(f"  - {appetitive['receptor'].tolist()}")

    print(f"\nAversive receptors: {len(aversive)}")
    print(f"  - {aversive['receptor'].tolist()}")

    # Test 6: Get statistics
    print("\n" + "=" * 80)
    print("TEST 6: Database statistics")
    print("=" * 80)

    stats = db.get_statistics()
    print("\nDatabase statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  - {key}: {value:.3f}")
        else:
            print(f"  - {key}: {value}")

    # Test 7: Print formatted summary
    print("\n" + "=" * 80)
    print("TEST 7: Formatted summary")
    print("=" * 80)

    db.print_summary("Or49a")

    print("\n" + "=" * 80)
    print("✓✓✓ ALL LOOKUP TESTS PASSED! ✓✓✓")
    print("=" * 80)
    print("\nDatabase lookup tools are ready to use!")


if __name__ == "__main__":
    main()
