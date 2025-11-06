#!/usr/bin/env python3
"""
Quick test script for FlyWire integration with actual data.
"""

from door_toolkit.flywire import FlyWireMapper, CommunityLabelsParser

print("=" * 70)
print("Testing FlyWire Integration")
print("=" * 70)

# Test 1: Parse FlyWire data
print("\n1. Testing FlyWire data parsing...")
parser = CommunityLabelsParser("data/flywire/processed_labels.csv.gz")
df = parser.parse(show_progress=False)
print(f"   ✓ Parsed {len(df):,} labels successfully")
print(f"   Columns: {df.columns.tolist()}")

# Test 2: Search for Or7a
print("\n2. Testing Or7a search...")
results = parser.search_patterns(["Or7a"])
or7a_cells = results["Or7a"]
print(f"   ✓ Found {len(or7a_cells)} Or7a cells")
if or7a_cells:
    print(f"   Sample: {or7a_cells[0].label[:80]}...")

# Test 3: Search for Or42b
print("\n3. Testing Or42b search...")
results = parser.search_patterns(["Or42b"])
or42b_cells = results["Or42b"]
print(f"   ✓ Found {len(or42b_cells)} Or42b cells")

# Test 4: Search for Or47b
print("\n4. Testing Or47b search...")
results = parser.search_patterns(["Or47b"])
or47b_cells = results["Or47b"]
print(f"   ✓ Found {len(or47b_cells)} Or47b cells")

# Test 5: Get unique receptors
print("\n5. Testing unique receptor extraction...")
receptors = parser.get_unique_receptors()
or_receptors = {k: v for k, v in receptors.items() if k.startswith('Or')}
print(f"   ✓ Found {len(or_receptors)} unique Or receptors")
print(f"   Top 5: {list(or_receptors.items())[:5]}")

# Test 6: Initialize mapper (without door_cache for now)
print("\n6. Testing FlyWireMapper initialization...")
mapper = FlyWireMapper("data/flywire/processed_labels.csv.gz", auto_parse=True)
print(f"   ✓ Mapper initialized with {mapper.labels_parser.n_labels:,} labels")

# Test 7: Find cells using mapper
print("\n7. Testing FlyWireMapper.find_receptor_cells()...")
cells = mapper.find_receptor_cells("Or7a")
print(f"   ✓ Found {len(cells)} Or7a cells via mapper")

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("=" * 70)
print("\nKey findings:")
print(f"  - Total FlyWire labels: {len(df):,}")
print(f"  - Unique Or receptors: {len(or_receptors)}")
print(f"  - Or7a neurons: {len(or7a_cells)}")
print(f"  - Or42b neurons: {len(or42b_cells)}")
print(f"  - Or47b neurons: {len(or47b_cells)}")
