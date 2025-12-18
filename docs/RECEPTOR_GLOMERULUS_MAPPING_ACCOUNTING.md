# Receptor → Glomerulus Mapping Accounting

**Why This Matters**: Preventing confusion between receptor counts and unique glomerulus counts in FlyWire integration analyses.

## The Problem

In FlyWire-DoOR integration, multiple receptor genes can map to the same glomerulus (**many-to-one mapping**). This creates a common source of confusion:

> "We added 11 receptors to the mapping, but the total only increased by 4. Is something broken?"

**Answer**: No! The mapping is working correctly. Those 11 receptors mapped to only 4 unique glomeruli due to many-to-one collapse.

## Scientific Context

### Many-to-One Receptor→Glomerulus Mapping

In the Drosophila olfactory system:
- **Receptors** are genes (e.g., `Or82a`, `Or94a`, `Ir31a`) that encode olfactory receptor proteins
- **Glomeruli** are anatomical structures in the antennal lobe where ORN axons converge
- **Key fact**: Multiple receptors can map to the same glomerulus

**Example**: In FlyWire, both `OR82A` and `OR94A` map to glomerulus `VA6`:
```
OR82A → VA6
OR94A → VA6
```

Result: **+2 receptors, +1 unique glomerulus**

### Why This Happens

1. **Receptor gene duplication**: Some receptors (e.g., `Or82a`/`Or94a`) are paralogs that target the same glomerulus
2. **Developmental biology**: Multiple ORN classes can project to overlapping glomerular regions
3. **FlyWire naming conventions**: Glomeruli are labeled by their anatomical position, not by individual receptors

## The Mapping Accounting System

To eliminate confusion, the toolkit now includes **explicit accounting** for receptor→glomerulus mappings via the `mapping_accounting` module.

### Core Principle

**Every mapping operation MUST report BOTH:**
1. **Receptor counts** (number of receptor genes)
2. **Unique glomerulus counts** (number of distinct glomeruli)

### Key Module: `mapping_accounting.py`

Location: `src/door_toolkit/integration/mapping_accounting.py`

#### Main Function: `compute_mapping_stats()`

```python
from door_toolkit.integration.mapping_accounting import compute_mapping_stats

# Example: Create a receptor→glomerulus mapping
mapping = {
    'OR82A': 'VA6',
    'OR94A': 'VA6',  # Collision: both map to VA6
    'OR7A': 'DL5',
}

# Compute comprehensive statistics
stats = compute_mapping_stats(
    mapping,
    note="Example mapping",
    adult_only=False
)

# Results include BOTH counts:
print(stats['n_receptors_mapped'])                      # 3
print(stats['n_unique_glomeruli_from_mapped_receptors']) # 2 (VA6, DL5)
print(stats['collision_count'])                         # 1 (VA6)
print(stats['collisions'])                              # {'VA6': ['OR82A', 'OR94A']}
```

**Output Fields**:
- `n_receptors_total`: Total number of receptor genes (candidates)
- `n_receptors_mapped`: Number successfully mapped to glomeruli
- `n_receptors_unmapped`: Number that failed to map
- `n_receptors_excluded_larval`: Number excluded as larval-only (if `adult_only=True`)
- `n_unique_glomeruli_from_mapped_receptors`: **CRITICAL** - Number of distinct glomeruli
- `collisions`: Dict of glomeruli with ≥2 receptors (many-to-one)
- `collision_count`: Number of glomeruli with collisions
- `collision_summary`: Human-readable summary (e.g., `"VA6: OR82A, OR94A"`)

#### Helper Functions

```python
from door_toolkit.integration.mapping_accounting import (
    format_mapping_summary,      # Compact summary string
    log_mapping_stats,           # Pretty-printed logging
    build_glomerulus_to_receptors,  # Reverse mapping
    summarize_collisions,        # Find many-to-one cases
    write_mapping_stats_json     # Persist to JSON
)

# Compact summary for reports
summary = format_mapping_summary(stats)
# "3 receptors → 2 unique glomeruli (1 collision)"

# Write to JSON artifact for reproducibility
write_mapping_stats_json("mapping_stats.json", stats)
```

### Adult/Larval Filtering

The `adult_only` parameter automatically excludes larval-only receptors:

```python
stats = compute_mapping_stats(
    mapping,
    adult_only=True  # Exclude larval receptors
)

# Larval receptors excluded and reported separately
print(stats['n_receptors_excluded_larval'])  # Count
print(stats['receptors_excluded_larval'])    # List of names
```

**Larval-only DoOR responding units** (DoOR.mappings `adult=False`, `larva=True`): `OR1A, OR22C, OR24A, OR30A, OR45A, OR45B, OR59A, OR63A, OR74A, OR83A, OR85C, OR94A, OR94B`

Source: DoOR.mappings (DoOR.data v2.0.0), Münch & Galizia 2016 (DOI: 10.1038/srep21841)

## Where This Is Used

The mapping accounting system is integrated throughout the toolkit:

### 1. Integration Module

**`DoORFlyWireIntegrator`** (in `src/door_toolkit/integration/integrator.py`):

```python
from door_toolkit.integration import DoORFlyWireIntegrator

integrator = DoORFlyWireIntegrator(
    door_cache="door_cache",
    connectomics_data="data/interglomerular_crosstalk_pathways.csv"
)

# Mapping stats are automatically computed and logged during initialization
# Output:
# ======================================================================
# RECEPTOR → GLOMERULUS MAPPING STATISTICS
# ======================================================================
#   Receptors mapped (DoOR): 44
#   Unique glomeruli (FlyWire): 42
#   Many-to-one collapses: 2 glomeruli receive ≥2 receptors
#     - VA6: OR82A, OR94A
#     - DM1: OR42B, OR43B
# ======================================================================
```

The stats are stored in `integrator.mapping_stats` for programmatic access.

### 2. Scripts

**`scripts/generate_complete_receptor_mapping.py`**:
- Generates DoOR→FlyWire mapping CSV
- Writes `door_to_flywire_mapping_stats.json` with full accounting
- Prints summary showing receptor vs glomerulus counts

**`scripts/verify_receptor_coverage.py`**:
- Verifies mapping coverage
- Reports receptor counts, unique glomeruli, and collisions
- Shows which glomeruli have many-to-one mapping

### 3. JSON Artifacts

Mapping statistics are persisted to JSON for reproducibility:

**Location**: `data/mappings/door_to_flywire_mapping_stats.json`

**Contents** (example):
```json
{
  "n_receptors_total": 44,
  "n_receptors_mapped": 44,
  "n_unique_glomeruli_from_mapped_receptors": 42,
  "collision_count": 2,
  "collisions": {
    "VA6": ["OR82A", "OR94A"],
    "DM1": ["OR42B", "OR43B"]
  },
  "collision_summary": [
    "VA6: OR82A, OR94A",
    "DM1: OR42B, OR43B"
  ],
  "note": "Complete DoOR → FlyWire mapping",
  "adult_only_mode": false
}
```

## Best Practices

### When Building/Using Mappings

1. **Always use `compute_mapping_stats()`** when creating or modifying receptor→glomerulus mappings
2. **Always report BOTH counts**: receptors AND unique glomeruli
3. **Never say "Total: N"** without specifying whether N refers to receptors or glomeruli
4. **Check for collisions**: Use `collision_summary` to identify many-to-one cases
5. **Write JSON artifacts**: Call `write_mapping_stats_json()` for reproducibility

### Example: Correct Reporting

```python
# ✅ CORRECT: Clear distinction
print(f"Mapped {stats['n_receptors_mapped']} receptors to "
      f"{stats['n_unique_glomeruli_from_mapped_receptors']} unique glomeruli")
# Output: "Mapped 44 receptors to 42 unique glomeruli"

# ❌ WRONG: Ambiguous
print(f"Mapped {len(mapping)} items")  # Items of what? Receptors? Glomeruli?
```

### Example: Handling Unmapped Receptors

```python
stats = compute_mapping_stats(
    receptor_to_glomerulus,
    input_receptors=all_door_receptors,  # Full candidate list
    unmapped_receptors=failed_to_map,    # Explicit unmapped list
    note="Coverage analysis"
)

# Now stats include:
# - n_receptors_total (all candidates)
# - n_receptors_mapped (successfully mapped)
# - n_receptors_unmapped (failed to map)
# - receptors_unmapped (list of names)
```

## FAQ

### Q: Why do I see "44 receptors → 42 unique glomeruli"?

**A**: This indicates **2 many-to-one collapses** (collisions). Check `stats['collision_summary']` to see which glomeruli receive multiple receptors.

Likely: `VA6` (receives `OR82A` + `OR94A`) and possibly `DM1` (receives `OR42B` + `OR43B`).

### Q: When should I use `adult_only=True`?

**A**: Use `adult_only=True` when analyzing adult female *Drosophila* (the default for FlyWire data). This excludes 12 larval-only receptors that are not expressed in the adult antennal lobe.

Use `adult_only=False` for:
- Complete receptor inventories
- Comparative analyses across life stages
- When you're unsure (safer to include all)

### Q: How do I know if my mapping has collisions?

**A**: Check `stats['collision_count']`:
```python
if stats['collision_count'] > 0:
    print("Many-to-one mapping detected!")
    for line in stats['collision_summary']:
        print(f"  {line}")
```

### Q: What if I only care about the final count?

**A**: Use `format_mapping_summary()` for a compact one-liner:
```python
summary = format_mapping_summary(stats)
print(summary)
# "44 receptors → 42 unique glomeruli (2 collisions)"
```

## Implementation Details

### Deterministic Ordering

All lists in mapping stats are **sorted** for reproducibility:
- `receptors_mapped`: Alphabetically sorted
- `receptors_unmapped`: Alphabetically sorted
- `collision_summary`: Sorted by glomerulus name

### Stateless Design

`compute_mapping_stats()` is a **pure function** with no global state. All parameters are explicit:
- No hidden defaults
- No side effects (except logging)
- Fully testable

### Testing

Comprehensive test suite in `tests/test_mapping_accounting.py`:
- 28 test methods across 6 test classes
- Critical test: `test_many_to_one_collapse_case()` validates VA6 collision handling
- Tests for adult-only filtering, JSON persistence, deterministic ordering

Run tests:
```bash
python -m pytest tests/test_mapping_accounting.py -v
```

## References

- **DoOR database**: [Database of Odorant Responses](http://neuro.uni-konstanz.de/DoOR/)
- **FlyWire**: [Adult *Drosophila* brain connectome](https://flywire.ai/)
- **Olfactory receptor mapping**: Couto et al. (2005), Hallem & Carlson (2006), Silbering et al. (2011)
- **Larval receptors**: Kreher et al. (2005), Fishilevich & Vosshall (2005)

## Summary

The mapping accounting system ensures that:
1. **Receptor counts ≠ glomerulus counts** is always explicit
2. **Many-to-one collapses** are detected and reported
3. **JSON artifacts** provide reproducible records
4. **Adult/larval filtering** is transparent and documented

**Bottom line**: You will never again see "we added 11 but total only increased by 4" without immediately understanding why.
