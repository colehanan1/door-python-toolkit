# Authoritative DoOR → FlyWire ORN Mapping

This repo maintains a **single authoritative mapping artifact** from **DoOR responding units** (DoOR 2.0; Münch & Galizia 2016, DOI: 10.1038/srep21841) to **FlyWire glomerulus labels** in the canonical form `ORN_<glomerulus>` (FlyWire label conventions: https://github.com/flyconnectome/flywire_annotations).

## Artifacts (Tracked)

- `data/mappings/door_to_flywire_mapping.csv`  
  Single source of truth DoOR→FlyWire mapping with row-level provenance and explicit ambiguity handling.
- `data/mappings/door_to_flywire_manual_overrides.csv`  
  Minimal, explicit overrides for known mismatches (with citable provenance).
- `data/mappings/sensillum_to_receptor_reference.csv`  
  Curated sensillum-unit mappings (ab*/ac*/pb*/at*) with provenance, including multi-glomerulus expansions marked as ambiguous.
- `data/mappings/receptor_inventory.csv`  
  Unified inventory table used downstream (mapping + life-stage metadata + connectivity metrics).

## Mapping Pathways (Supported)

The mapping builder supports three explicit pathways:

1. **Direct gene mapping** (`Or*`, `Ir*`, `Gr*`)  
   Primary source is DoOR’s `DoOR.mappings` table (DoOR.data v2.0.0; DOI: 10.1038/srep21841). A single DoOR glomerulus code (e.g., `DL5`, `DP1m`) is converted to FlyWire’s `ORN_` label.
2. **Sensillum-unit mapping** (`ab*`, `ac*`, `pb*`, `at*`)  
   Uses `data/mappings/sensillum_to_receptor_reference.csv` to map sensillum units to glomerulus labels and (optionally) intermediate receptor genes. Multi-glomerulus cases are **explicitly marked ambiguous**.
3. **Manual overrides**  
   `data/mappings/door_to_flywire_manual_overrides.csv` applies small, explicitly-cited corrections for known mismatches or policy decisions.

## Provenance Columns (Required)

Each mapping row records:
- `source_name`
- `source_year`
- `source_url_or_doi`
- `evidence_note`
- `confidence`

## Strict Validations (Hard-Fail)

The build fails if any of the following are violated:
- Any non-empty FlyWire target does **not** begin with `ORN_`.
- `Ir64a.DC4` does **not** map to `ORN_DC4`.
- `Ir64a.DP1m` does **not** map to `ORN_DP1m`.
- Conflicting/duplicate mappings exist without `is_ambiguous=Yes`.

## Adult-Only Filtering Policy

FlyWire connectomics data are adult-brain annotations. Adult-only analyses must exclude DoOR responding units flagged **larval-only** in DoOR.mappings (`adult=False`, `larva=True`):

`OR1A, OR22C, OR24A, OR30A, OR45A, OR45B, OR59A, OR63A, OR74A, OR83A, OR85C, OR94A, OR94B`

These units remain tracked in `data/mappings/receptor_inventory.csv` but are excluded from adult-only integration by default.

## Or22b Policy + Sensitivity Analysis

- Default mapping is **glomerulus-level**: `Or22b → ORN_DM2`.
- The mapping row is marked with **gene-level uncertainty** in provenance notes.
- A sensitivity analysis mode is available in integration code to **exclude Or22b** when strict gene-level single-cell annotation is required.
  - `DoORFlyWireIntegrator(..., strict_single_cell_annotation=True)`

## Reproducing the Artifacts

1. Generate the authoritative mapping:
   - `python scripts/generate_complete_receptor_mapping.py`
2. Generate the unified receptor inventory:
   - `python scripts/generate_receptor_inventory.py`

## Decision Log (Decision → Evidence → Implementation Change)

- **Use DoOR.mappings as primary mapping source** → DoOR 2.0 is the canonical DoOR receptor/glomerulus reference (DOI: 10.1038/srep21841) → Implemented `src/door_toolkit/integration/door_to_flywire_mapping.py` and cached DoOR.mappings ingestion.
- **Enforce FlyWire label convention `ORN_`** → FlyWire annotation conventions are published (flywire_annotations repo) → Mapping validator rejects any non-empty target not starting with `ORN_`.
- **Treat multi-glomerulus DoOR codes as ambiguity (not “mapped”)** → DoOR encodes ambiguity with `+` and `/` (e.g., `DM5+DM3`, `DL2d/v`) → Mapping artifact stores one row per candidate with `is_ambiguous=Yes`; integration skips these by default.
- **Fix known mismatches via explicit overrides** → DoOR.mappings assigns `Or10a→DL1`, and dotted-suffix IRs must match their suffix → Added `data/mappings/door_to_flywire_manual_overrides.csv` and strict validator checks.
