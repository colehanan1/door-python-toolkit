"""
FlyWire Integration Example
============================

This example demonstrates how to integrate DoOR receptor data with FlyWire
community labels for spatial and network analysis.

Requirements:
    - DoOR cache (run door-extract first)
    - FlyWire community labels CSV file (processed_labels.csv.gz)
"""

from door_toolkit.flywire import FlyWireMapper

# Example 1: Find cells expressing specific receptors
print("=" * 70)
print("Example 1: Finding Or42b Cells")
print("=" * 70)

# Initialize mapper
mapper = FlyWireMapper(
    community_labels_path="data/processed_labels.csv.gz",
    door_cache_path="door_cache",
    auto_parse=True,  # Automatically parse labels
)

# Find all Or42b-expressing cells
or42b_cells = mapper.find_receptor_cells("Or42b")
print(f"\nFound {len(or42b_cells)} Or42b neurons:")
for i, cell in enumerate(or42b_cells[:5], 1):  # Show first 5
    print(f"  {i}. Root ID: {cell['root_id']}")
    print(f"     Label: {cell['label']}")
    if "position" in cell:
        pos = cell["position"]
        print(f"     Position: ({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f})")


# Example 2: Map all DoOR receptors to FlyWire
print("\n" + "=" * 70)
print("Example 2: Mapping All Receptors")
print("=" * 70)

mappings = mapper.map_door_to_flywire()

print(f"\nMapping Results:")
print(f"  Total receptors mapped: {len(mappings)}")
print(f"  Total cells found: {sum(m.cell_count for m in mappings.values())}")

# Show top 10 receptors by cell count
print(f"\n  Top 10 receptors by cell count:")
sorted_mappings = sorted(
    mappings.items(), key=lambda x: x[1].cell_count, reverse=True
)
for receptor, mapping in sorted_mappings[:10]:
    print(f"    {receptor}: {mapping.cell_count} cells")

# Export mappings
mapper.export_mapping("output/flywire_mapping.json", format="json")
print(f"\n  Exported mappings to output/flywire_mapping.json")


# Example 3: Create spatial activation map
print("\n" + "=" * 70)
print("Example 3: Spatial Activation Map")
print("=" * 70)

odorant = "ethyl butyrate"
spatial_map = mapper.create_spatial_activation_map(odorant)

print(f"\nSpatial map for '{odorant}':")
print(f"  Active receptors: {len(spatial_map.receptor_activations)}")
print(f"  Spatial points: {spatial_map.total_cells}")

# Show top activated receptors
sorted_receptors = sorted(
    spatial_map.receptor_activations.items(),
    key=lambda x: x[1],
    reverse=True,
)
print(f"\n  Top activated receptors:")
for receptor, activation in sorted_receptors[:5]:
    print(f"    {receptor}: {activation:.3f}")

# Export spatial map
import json

with open("output/spatial_map.json", "w") as f:
    json.dump(spatial_map.to_dict(), f, indent=2)
print(f"\n  Exported spatial map to output/spatial_map.json")


# Example 4: Get mapping statistics
print("\n" + "=" * 70)
print("Example 4: Mapping Statistics")
print("=" * 70)

stats = mapper.get_mapping_statistics()
print(f"\nMapping Statistics:")
for key, value in stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")

print("\nDone! Check the output/ directory for exported files.")
