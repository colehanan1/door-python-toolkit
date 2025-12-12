# FlyWire Integration Notes

## Data Format

Your FlyWire data is located at: `data/flywire/processed_labels.csv.gz`

### Structure
```csv
root_id,processed_labels
720575940604352689,"['ORN_DL5; Or7a; FBbt_00067005', 'DL5 / Or7a ORN', ...]"
```

- **Column 1**: `root_id` - FlyWire neuron identifier
- **Column 2**: `processed_labels` - String representation of a list containing cell labels

## What Was Fixed

### 1. Column Name Mapping
Updated [community_labels.py](src/door_toolkit/flywire/community_labels.py:163) to recognize `processed_labels` column:

```python
column_mappings = {
    ...
    "processed_labels": "label",  # FlyWire standard format
    ...
}
```

### 2. Odorant Names
The pathway analyzer already includes correct odorant names:
- ✓ Uses "1-hexanol" (not just "hexanol")
- ✓ Includes variants: ["hexanol", "1-hexanol", "hexan-1-ol"]

### 3. File Paths
Updated examples and CLI to use correct default path:
- ❌ Old: `processed_labels.csv.gz`
- ✅ New: `data/flywire/processed_labels.csv.gz`

## Test Results

```
Total FlyWire labels: 100,013
Unique Or receptors: 36

Sample neuron counts:
- Or7a neurons: 41
- Or42b neurons: 71
- Or47b neurons: 98
```

## Usage Examples

### CLI Commands

```bash
# Find Or7a cells
door-flywire --labels data/flywire/processed_labels.csv.gz --find-receptor Or7a

# Map all receptors to FlyWire (requires door_cache)
door-flywire --labels data/flywire/processed_labels.csv.gz \
  --cache door_cache --map-receptors

# Create spatial activation map
door-flywire --labels data/flywire/processed_labels.csv.gz \
  --cache door_cache --spatial-map "1-hexanol" --output hexanol_map.json
```

### Python API

```python
from door_toolkit.flywire import FlyWireMapper

# Initialize with correct path
mapper = FlyWireMapper(
    community_labels_path="data/flywire/processed_labels.csv.gz",
    door_cache_path="door_cache",
    auto_parse=True
)

# Find specific receptor cells
or7a_cells = mapper.find_receptor_cells("Or7a")
print(f"Found {len(or7a_cells)} Or7a neurons")

# Map all receptors
mappings = mapper.map_door_to_flywire()
print(f"Mapped {len(mappings)} receptors to FlyWire")
```

## Verified Receptors in Your Data

The following Or receptors are confirmed present in your FlyWire dataset:

| Receptor | Cell Count |
|----------|------------|
| Or7a     | 41         |
| Or10a    | 69         |
| Or13a    | 20         |
| Or19     | 39         |
| Or22     | 54         |
| Or23a    | 29         |
| Or42b    | 71         |
| Or47b    | 98         |
| ... (36 total) |

## Pathway Analysis with Correct Odorants

```python
from door_toolkit.pathways import PathwayAnalyzer

analyzer = PathwayAnalyzer("door_cache")

# Trace Or47b→1-hexanol→feeding pathway
pathway = analyzer.trace_or47b_feeding_pathway()
print(f"Pathway strength: {pathway.strength:.3f}")

# The analyzer automatically tries these variants:
# - "hexanol"
# - "1-hexanol"  ✓ (this is in your DoOR cache)
# - "hexan-1-ol"
```

## Troubleshooting

### If receptor search returns 0 results:
1. Check the exact receptor name in your data:
   ```bash
   gunzip -c data/flywire/processed_labels.csv.gz | grep -i "or7a"
   ```

2. Try case-insensitive search:
   ```python
   results = parser.search_patterns(["Or7a"], case_sensitive=False)
   ```

### If odorant not found:
1. List available odorants in your DoOR cache:
   ```bash
   door-extract --list-odorants door_cache --pattern "hex"
   ```

2. Common DoOR odorant names:
   - "1-hexanol" (not "hexanol")
   - "ethyl butyrate" (not "ethyl-butyrate")
   - "acetic acid" (not "acetate")

## Integration Test

Run the integration test to verify everything works:

```bash
python test_flywire_integration.py
```

Expected output:
```
======================================================================
All tests passed! ✓
======================================================================
```

## Next Steps

1. **Map receptors to FlyWire** (if you have door_cache):
   ```bash
   door-flywire --labels data/flywire/processed_labels.csv.gz \
     --cache door_cache --map-receptors --output flywire_mapping.json
   ```

2. **Analyze Or47b pathway**:
   ```bash
   door-pathways --cache door_cache --trace or47b-feeding
   ```

3. **Create spatial maps**:
   ```bash
   door-flywire --labels data/flywire/processed_labels.csv.gz \
     --cache door_cache --spatial-map "1-hexanol" --output hexanol_spatial.json
   ```
