# Custom Pathway Analysis Guide

## Summary of Fixes

### Fixed Warnings
The warnings about "Could not encode hexanol" and "Could not encode hexan-1-ol" are now resolved. The code now pre-filters odorants to only those that exist in your DoOR database before attempting to encode them.

## Running Custom Pathways for Any Odorant/Receptor

### Basic Syntax

```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors RECEPTOR_NAME \
  --odorants "ODORANT_NAME" \
  --behavior "BEHAVIOR_DESCRIPTION"
```

### Finding Available Odorants

List odorants by pattern:
```bash
# List all acetate compounds
door-extract --list-odorants door_cache --pattern "acetate"

# List alcohols
door-extract --list-odorants door_cache --pattern "ol"

# List specific odorant
door-extract --list-odorants door_cache --pattern "geosmin"

# List all odorants (690 total)
door-extract --list-odorants door_cache
```

### Example Pathways

#### 1. Or42b → Ethyl Butyrate (Attractive)
Strong activation (0.527) - known attractive pathway
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b \
  --odorants "ethyl butyrate" \
  --behavior "attraction"
```

**Result:**
- Pathway Strength: 0.527
- Receptor Contribution: Or42b (100%)

#### 2. Or7a → Geranyl Acetate (Attractive)
Moderate activation (0.075)
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or7a \
  --odorants "geranyl acetate" \
  --behavior "attraction"
```

**Result:**
- Pathway Strength: 0.075
- Receptor Contribution: Or7a (100%)

#### 3. Or92a → Geosmin (Avoidance)
Note: May show 0.000 if no response data available
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or92a \
  --odorants "geosmin" \
  --behavior "avoidance"
```

#### 4. Or47b → 1-Hexanol (Feeding)
The classic feeding pathway (0.074)
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or47b \
  --odorants "1-hexanol" \
  --behavior "feeding"
```

### Multiple Receptors

Test multiple receptors at once:
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b Or47b Or7a \
  --odorants "ethyl acetate" \
  --behavior "mixed attraction"
```

### Multiple Odorants

Test receptor response to multiple odorants:
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b \
  --odorants "ethyl butyrate" "ethyl acetate" "benzyl acetate" \
  --behavior "fruit ester attraction"
```

### Common Odorant Classes

#### Alcohols
- 1-hexanol
- ethanol
- 1-octanol
- 2-butanol
- benzyl alcohol

#### Esters (Fruity)
- ethyl butyrate (strong Or42b activator)
- ethyl acetate
- geranyl acetate
- benzyl acetate

#### Aldehydes
- benzaldehyde
- hexanal
- acetaldehyde

#### Acids
- acetic acid
- butyric acid
- propionic acid

#### Terpenes
- geranyl acetate
- citronellyl acetate

#### Aversive Compounds
- geosmin

## Receptors Available in Your FlyWire Data

Based on your FlyWire dataset (`data/flywire/processed_labels.csv.gz`), these receptors have confirmed neurons:

| Receptor | Cell Count | Known Function |
|----------|------------|----------------|
| Or7a     | 41         | Aromatic compounds |
| Or10a    | 69         | General odorants |
| Or13a    | 20         | - |
| Or19     | 39         | - |
| Or22     | 54         | - |
| Or23a    | 29         | - |
| Or42b    | 71         | Fruit esters (ethyl butyrate) |
| Or47b    | 98         | Hexanol (feeding) |
| Or92a    | (present)  | Geosmin (avoidance) |
| ... 36 total Or receptors |

You can find any receptor in your FlyWire data:
```bash
door-flywire --labels data/flywire/processed_labels.csv.gz --find-receptor Or10a
```

## Using Pre-defined Pathways

For well-characterized pathways, use the built-in trace commands:

```bash
# Or47b feeding pathway
door-pathways --cache door_cache --trace or47b-feeding

# Or42b fruit ester pathway
door-pathways --cache door_cache --trace or42b

# Or92a geosmin avoidance
door-pathways --cache door_cache --trace or92a-avoidance
```

## Python API for Custom Pathways

```python
from door_toolkit.pathways import PathwayAnalyzer

analyzer = PathwayAnalyzer("door_cache")

# Test any receptor-odorant combination
pathway = analyzer.trace_custom_pathway(
    receptors=["Or42b"],
    odorants=["ethyl butyrate", "ethyl acetate"],
    behavior="fruit attraction"
)

print(f"Pathway strength: {pathway.strength:.3f}")
print(f"Top receptors: {pathway.get_top_receptors(3)}")

# Access detailed responses
for odorant, response in pathway.metadata["odorant_responses"].items():
    print(f"{odorant}: {response:.3f}")
```

## Advanced: Testing Multiple Pathways

Create a batch test script:

```python
from door_toolkit.pathways import PathwayAnalyzer

analyzer = PathwayAnalyzer("door_cache")

# Test suite of receptor-odorant pairs
test_cases = [
    ("Or42b", ["ethyl butyrate"], "attraction"),
    ("Or47b", ["1-hexanol"], "feeding"),
    ("Or7a", ["geranyl acetate"], "attraction"),
    ("Or10a", ["benzaldehyde"], "detection"),
]

results = []
for receptor, odorants, behavior in test_cases:
    pathway = analyzer.trace_custom_pathway([receptor], odorants, behavior)
    results.append({
        "receptor": receptor,
        "odorant": odorants[0],
        "strength": pathway.strength,
        "behavior": behavior
    })

# Compare pathways
import pandas as pd
df = pd.DataFrame(results)
df = df.sort_values("strength", ascending=False)
print(df)
```

## Interpreting Pathway Strength

- **0.8 - 1.0**: Very strong activation (highly responsive)
- **0.5 - 0.8**: Strong activation (responsive)
- **0.2 - 0.5**: Moderate activation (detectable response)
- **0.05 - 0.2**: Weak activation (minimal response)
- **0.0 - 0.05**: Very weak/no activation

Example:
- Or42b → ethyl butyrate = 0.527 (strong activation, known attractive pathway)
- Or47b → 1-hexanol = 0.074 (weak but functional feeding pathway)
- Or7a → geranyl acetate = 0.075 (weak but detectable)

## Troubleshooting

### "Could not encode [odorant]" Warnings (Fixed)
This is now resolved. The code pre-filters odorants before encoding.

### Pathway Strength = 0.000
This means:
1. The odorant may not be in the DoOR database (check with `door-extract --list-odorants`)
2. No response data available for this receptor-odorant pair
3. The receptor name might be incorrect

### Finding Exact Odorant Names
DoOR uses specific naming conventions:
- Use "1-hexanol" not "hexanol"
- Use "ethyl butyrate" not "ethyl-butyrate"
- Use "acetic acid" not "acetate"

Always check available names first:
```bash
door-extract --list-odorants door_cache --pattern "YOUR_SEARCH_TERM"
```

## Next Steps

1. **Explore your FlyWire neurons**: Find which receptors have the most cells
```bash
door-flywire --labels data/flywire/processed_labels.csv.gz --map-receptors
```

2. **Test specific odor responses**: Pick odorants relevant to your experiments
```bash
door-pathways --cache door_cache --custom-pathway \
  --receptors Or42b --odorants "YOUR_ODORANT" --behavior "YOUR_BEHAVIOR"
```

3. **Generate blocking experiments**: Create experimental protocols
```bash
door-pathways --cache door_cache --generate-experiment \
  --receptors Or42b --odorants "ethyl butyrate" --output experiment.json
```

4. **Compute receptor importance**: Find which receptors matter most
```bash
door-pathways --cache door_cache --shapley feeding --output importance.json
```
