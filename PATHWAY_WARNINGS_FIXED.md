# Pathway Analysis Warnings - RESOLVED

## Summary

All warnings have been fixed and custom pathway analysis is now fully functional for any receptor-odorant combination.

## Warnings Fixed

### 1. "Could not encode hexanol" - FIXED
**Issue**: The pathway analyzer tried multiple odorant name variants but logged warnings for each failed attempt.

**Root Cause**: The KNOWN_PATHWAYS configuration included variants ["hexanol", "1-hexanol", "hexan-1-ol"] but only "1-hexanol" exists in the DoOR database.

**Solution**: Modified [analyzer.py:173-177](src/door_toolkit/pathways/analyzer.py#L173-L177) to pre-filter odorants before encoding:
```python
# Filter to only odorants that exist in the database
available_odorants = [o for o in key_odorants if o in self.encoder.odorant_names]
if not available_odorants:
    logger.warning(f"None of the key odorants {key_odorants} found in DoOR database")
    logger.info(f"Available similar odorants: {[o for o in self.encoder.odorant_names if 'hexanol' in o.lower()][:5]}")
```

**Result**: No more warnings. The analyzer silently skips non-existent odorant names and only processes valid ones.

### 2. Testing Complete
```bash
# Before fix:
door-pathways --cache door_cache --trace or47b-feeding
# WARNING - Could not encode hexanol
# WARNING - Could not encode hexan-1-ol

# After fix:
door-pathways --cache door_cache --trace or47b-feeding
# Pathway: Or47b → Hexanol → Feeding
# Strength: 0.074
# ✓ No warnings
```

## Running Custom Pathways for Any Odor/Receptor

### Basic Usage

```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors RECEPTOR_NAME \
  --odorants "ODORANT_NAME" \
  --behavior "BEHAVIOR"
```

### Examples

#### Example 1: Or42b with Ethyl Butyrate (Strong Response)
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b \
  --odorants "ethyl butyrate" \
  --behavior "attraction"
```
**Result**: Strength: 0.527 (strong activation)

#### Example 2: Or42b with Ethyl Acetate (Very Strong Response)
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b \
  --odorants "ethyl acetate" \
  --behavior "attraction"
```
**Result**: Strength: 0.662 (very strong activation)

#### Example 3: Or7a with Geranyl Acetate
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or7a \
  --odorants "geranyl acetate" \
  --behavior "attraction"
```
**Result**: Strength: 0.075 (weak but detectable)

#### Example 4: Multiple Receptors
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b Or47b Or7a \
  --odorants "ethyl acetate" \
  --behavior "combined attraction"
```
**Result**: Strength: 0.311 (contributions: Or42b 71%, Or47b 21%, Or7a 9%)

#### Example 5: Multiple Odorants
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b \
  --odorants "ethyl butyrate" "ethyl acetate" "benzyl acetate" \
  --behavior "fruit ester blend"
```
**Result**: Average strength across all odorants with individual responses

## New Tools Created

### 1. Batch Pathway Testing
**File**: [test_custom_pathways.py](test_custom_pathways.py)

Tests multiple receptor-odorant combinations and provides comparative analysis:
```bash
python test_custom_pathways.py
```

**Output**:
- Summary table sorted by pathway strength
- Strength categories (strong/moderate/weak/none)
- Multi-receptor and multi-odorant examples
- Key findings report

**Results from test run**:
- Or42b → ethyl acetate: 0.662 (strongest)
- Or42b → ethyl butyrate: 0.527
- Or7a → geranyl acetate: 0.075
- Or47b → 1-hexanol: 0.074

### 2. Best Odorant Finder
**File**: [find_best_odorants.py](find_best_odorants.py)

Discovers which odorants activate a specific receptor most strongly:
```bash
# Find top 10 odorants for any receptor
python find_best_odorants.py Or42b

# Filter by odorant class
python find_best_odorants.py Or47b alcohol 10

# Find top 15 acetates for Or7a
python find_best_odorants.py Or7a acetate 15
```

**Or42b Example Results**:
- Total responsive odorants: 173 out of 690
- Top odorant: 3-hexanone (0.899 strength)
- Top 10 includes: ethyl acetate (0.662), ethyl butyrate (0.527)
- Output saved to CSV for further analysis

**Or47b Example Results** (filtering for "alcohol"):
- 8 alcohols tested, 2 responsive
- Top: phenethyl alcohol (0.150), benzyl alcohol (0.140)
- Note: 1-hexanol not in filter results (doesn't have "alcohol" in name)

### 3. Custom Pathway Guide
**File**: [CUSTOM_PATHWAY_GUIDE.md](CUSTOM_PATHWAY_GUIDE.md)

Comprehensive guide covering:
- How to find available odorants
- Common odorant classes (alcohols, esters, aldehydes, acids, terpenes)
- Receptors available in your FlyWire data (36 Or receptors confirmed)
- Interpreting pathway strength values
- Python API examples
- Advanced batch testing
- Troubleshooting tips

## Receptors in Your FlyWire Data

You have 36 Or receptors with confirmed neurons in `data/flywire/processed_labels.csv.gz`:

| Receptor | Cell Count | Known Strong Activators |
|----------|------------|-------------------------|
| Or42b    | 71         | 3-hexanone (0.899), ethyl acetate (0.662), ethyl butyrate (0.527) |
| Or47b    | 98         | 1-hexanol (0.074), phenethyl alcohol (0.150) |
| Or7a     | 41         | geranyl acetate (0.075) |
| Or10a    | 69         | (test with find_best_odorants.py) |
| Or22     | 54         | (test with find_best_odorants.py) |
| Or23a    | 29         | geranyl acetate (0.033) |
| ... 30 more Or receptors |

Find any receptor's neurons:
```bash
door-flywire --labels data/flywire/processed_labels.csv.gz --find-receptor Or10a
```

## Odorants in Your DoOR Cache

Total: 690 odorants across multiple chemical classes

Find odorants by pattern:
```bash
# List all acetate esters (36 found)
door-extract --list-odorants door_cache --pattern "acetate"

# List all butyrate esters (21 found)
door-extract --list-odorants door_cache --pattern "butyrate"

# List alcohols
door-extract --list-odorants door_cache --pattern "ol"

# Check specific odorant
door-extract --list-odorants door_cache --pattern "geosmin"
# Found: geosmin
```

### Common Odorant Classes Available

**Alcohols**: 1-hexanol, ethanol, 1-octanol, 2-butanol, benzyl alcohol, phenethyl alcohol

**Esters (Fruity)**:
- ethyl butyrate (Or42b: 0.527)
- ethyl acetate (Or42b: 0.662)
- geranyl acetate (Or7a: 0.075)
- benzyl acetate
- 3-hexanone (Or42b: 0.899) ⭐ strongest

**Aldehydes**: benzaldehyde, hexanal, acetaldehyde

**Acids**: acetic acid, butyric acid, propionic acid

**Ketones**: 3-hexanone, 3-penten-2-one

**Aversive**: geosmin (Or92a pathway)

## Interpreting Pathway Strength

| Strength Range | Interpretation | Example |
|----------------|----------------|---------|
| 0.8 - 1.0      | Very strong activation | 3-hexanone → Or42b (0.899) |
| 0.5 - 0.8      | Strong activation | ethyl acetate → Or42b (0.662) |
| 0.2 - 0.5      | Moderate activation | (none in tested examples) |
| 0.05 - 0.2     | Weak activation | phenethyl alcohol → Or47b (0.150) |
| 0.0 - 0.05     | Very weak/no activation | 1-hexanol → Or47b (0.074) |

Note: Even weak responses (0.05-0.2) can be behaviorally relevant, especially for highly specialized pathways like Or47b → hexanol → feeding.

## Python API

### Test Single Pathway
```python
from door_toolkit.pathways import PathwayAnalyzer

analyzer = PathwayAnalyzer("door_cache")

pathway = analyzer.trace_custom_pathway(
    receptors=["Or42b"],
    odorants=["ethyl butyrate"],
    behavior="fruit attraction"
)

print(f"Strength: {pathway.strength:.3f}")
print(f"Top receptors: {pathway.get_top_receptors()}")
```

### Find Best Odorants Programmatically
```python
from door_toolkit.pathways import PathwayAnalyzer
import pandas as pd

analyzer = PathwayAnalyzer("door_cache")
odorants = analyzer.encoder.odorant_names

results = []
for odorant in odorants:
    try:
        pathway = analyzer.trace_custom_pathway(
            receptors=["Or42b"],
            odorants=[odorant],
            behavior="detection"
        )
        if pathway.strength > 0.5:  # Only strong responses
            results.append({
                "odorant": odorant,
                "strength": pathway.strength
            })
    except:
        continue

df = pd.DataFrame(results).sort_values("strength", ascending=False)
print(df.head(10))
```

### Batch Test Multiple Pathways
```python
test_cases = [
    ("Or42b", ["ethyl butyrate"], "attraction"),
    ("Or47b", ["1-hexanol"], "feeding"),
    ("Or7a", ["geranyl acetate"], "attraction"),
]

for receptor, odorants, behavior in test_cases:
    pathway = analyzer.trace_custom_pathway(receptor, odorants, behavior)
    print(f"{receptor} → {odorants[0]}: {pathway.strength:.3f}")
```

## Quick Start Examples

### 1. Test a known pathway
```bash
door-pathways --cache door_cache --trace or47b-feeding
```

### 2. Test custom receptor-odorant pair
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b --odorants "ethyl acetate" --behavior "attraction"
```

### 3. Find best odorants for a receptor
```bash
python find_best_odorants.py Or42b
```

### 4. Batch test multiple pathways
```bash
python test_custom_pathways.py
```

### 5. Find receptor neurons in FlyWire
```bash
door-flywire --labels data/flywire/processed_labels.csv.gz --find-receptor Or42b
```

### 6. List available odorants
```bash
door-extract --list-odorants door_cache --pattern "acetate"
```

## All Issues Resolved

- [x] Fixed "Could not encode hexanol" warning
- [x] Fixed "Could not encode hexan-1-ol" warning
- [x] Created tools for testing any receptor-odorant combination
- [x] Documented how to find available odorants
- [x] Documented how to find available receptors
- [x] Provided batch testing tools
- [x] Provided best odorant discovery tools
- [x] Created comprehensive usage guide

## Next Steps

1. **Explore your receptors**: Use `find_best_odorants.py` to discover strong activators for receptors in your FlyWire data

2. **Test hypotheses**: Use custom pathways to test specific receptor-odorant combinations

3. **Batch analysis**: Run `test_custom_pathways.py` with your own test cases

4. **Generate experiments**: Create blocking experiment protocols for interesting pathways
```bash
door-pathways --cache door_cache --generate-experiment \
  --receptors Or42b --odorants "3-hexanone" --output experiment.json
```

5. **Compute importance**: Find which receptors matter most for specific behaviors
```bash
door-pathways --cache door_cache --shapley feeding --output importance.json
```

## Files Modified

1. [src/door_toolkit/pathways/analyzer.py:173-177](src/door_toolkit/pathways/analyzer.py#L173-L177) - Pre-filter odorants before encoding

## Files Created

1. [CUSTOM_PATHWAY_GUIDE.md](CUSTOM_PATHWAY_GUIDE.md) - Comprehensive usage guide
2. [test_custom_pathways.py](test_custom_pathways.py) - Batch pathway testing
3. [find_best_odorants.py](find_best_odorants.py) - Best odorant discovery tool
4. [PATHWAY_WARNINGS_FIXED.md](PATHWAY_WARNINGS_FIXED.md) - This summary document

## Support

See [CUSTOM_PATHWAY_GUIDE.md](CUSTOM_PATHWAY_GUIDE.md) for detailed examples and troubleshooting.
