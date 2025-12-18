"""
Authoritative DoOR → FlyWire ORN Mapping
======================================

Builds a *single source of truth* mapping table from DoOR responding units
("receptors" in DoOR nomenclature) to FlyWire ORN glomerulus labels
(`ORN_<glomerulus>`).

This module is designed for publication-quality, auditable pipelines:
- Deterministic build from explicit inputs (DoOR.mappings + curated tables).
- Explicit provenance per mapping row.
- Strict validations that hard-fail on invalid targets, known mismatches, and
  unresolved conflicts.

Scientific sources (citable in provenance columns):
- DoOR 2.0 / DoOR.data v2.0.0: Münch & Galizia (2016) Sci Rep. DOI: 10.1038/srep21841
- FlyWire label conventions: https://github.com/flyconnectome/flywire_annotations
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from door_toolkit.integration.receptor_identifier import (
    flywire_glomerulus_from_door_code,
    normalize_receptor_identifier,
)

DOOR_2_DOI = "10.1038/srep21841"
FLYWIRE_ANNOTATIONS_URL = "https://github.com/flyconnectome/flywire_annotations"


_PLUS_SPLIT_RE = re.compile(r"[+,]")
_FLYWIRE_ORN_RE = re.compile(r"^ORN_[A-Z]+\\d*[a-z]*$")


@dataclass(frozen=True)
class MappingPaths:
    manual_override: str = "manual_override"
    door_mappings: str = "door_mappings"
    sensillum_reference: str = "sensillum_reference"
    dotted_suffix_inference: str = "dotted_suffix_inference"
    excluded_larval: str = "excluded_larval"
    unmapped: str = "unmapped"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_paths() -> Dict[str, Path]:
    root = repo_root()
    return {
        "door_cache_matrix": root / "door_cache" / "response_matrix_norm.parquet",
        "door_mappings_rdata": root / "data" / "raw" / "DoOR.data-2.0.0" / "data" / "DoOR.mappings.RData",
        "door_mappings_parquet": root / "door_cache" / "door_mappings_full.parquet",
        "manual_overrides": root / "data" / "mappings" / "door_to_flywire_manual_overrides.csv",
        "sensillum_reference": root / "data" / "mappings" / "sensillum_to_receptor_reference.csv",
        "mapping_output": root / "data" / "mappings" / "door_to_flywire_mapping.csv",
    }


def load_door_receptors_from_cache(matrix_path: Path) -> List[str]:
    df = pd.read_parquet(matrix_path)
    if df.shape[0] > df.shape[1]:
        df = df.T
    return sorted(df.index.astype(str).tolist())


def load_door_mappings_full(
    *,
    rdata_path: Optional[Path] = None,
    cache_parquet_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load DoOR.mappings (DoOR.data v2.0.0) with adult/larva flags and metadata.

    Prefers the local DoOR.mappings.RData if available; caches to parquet for speed.
    """
    paths = default_paths()
    rdata = rdata_path or paths["door_mappings_rdata"]
    cache = cache_parquet_path or paths["door_mappings_parquet"]

    if cache.exists():
        return pd.read_parquet(cache)

    if not rdata.exists():
        raise FileNotFoundError(
            f"DoOR.mappings.RData not found at {rdata}. "
            "Provide rdata_path or add DoOR.data-2.0.0 under data/raw/."
        )

    import pyreadr  # optional dependency vendored in many scientific envs

    result = pyreadr.read_r(str(rdata))
    if "DoOR.mappings" not in result:
        raise ValueError(f"Expected key 'DoOR.mappings' in {rdata}; got {list(result.keys())}")

    df = result["DoOR.mappings"].copy()

    # Normalize column naming to stable snake_case-ish identifiers we use elsewhere.
    df = df.rename(
        columns={
            "co.receptor": "co_receptor",
            "sensillum.type": "sensillum_type",
            "dataset.existing": "dataset_existing",
            "code.OSN": "code_osn",
        }
    )

    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def larval_only_receptors_from_door_mappings(door_mappings_df: pd.DataFrame) -> List[str]:
    """
    Operational definition: adult == False and larva == True in DoOR.mappings.
    """
    required = {"receptor", "adult", "larva"}
    if not required.issubset(set(door_mappings_df.columns)):
        return []
    mask = (door_mappings_df["adult"] == False) & (door_mappings_df["larva"] == True)  # noqa: E712
    recs = sorted(set(door_mappings_df.loc[mask, "receptor"].astype(str).tolist()))
    return recs


def _dedupe_preserve_order(values: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


def expand_door_glomerulus_codes(raw: Optional[str]) -> List[str]:
    """
    Expand DoOR glomerulus strings that encode multiple candidates.

    Examples:
      - "DM5+DM3" -> ["DM5", "DM3"]
      - "VL1+VM1+VL2p" -> ["VL1", "VM1", "VL2p"]
      - "DL2d/v" -> ["DL2d", "DL2v"]
      - "DL2d/v+VC3" -> ["DL2d", "DL2v", "VC3"]
    """
    if raw is None:
        return []
    text = str(raw).strip()
    if not text or text == "?":
        return []

    parts = [p.strip() for p in _PLUS_SPLIT_RE.split(text) if p.strip()]
    expanded: List[str] = []
    for part in parts:
        if "/" not in part:
            expanded.append(part)
            continue

        tokens = [t.strip() for t in part.split("/") if t.strip()]
        if not tokens:
            continue

        first = tokens[0]
        match = re.match(r"^([A-Za-z]+\\d+)([A-Za-z]*)$", first)
        if not match:
            # If it doesn't match our expected pattern, keep as-is and let validators decide.
            expanded.append(first)
            expanded.extend(tokens[1:])
            continue

        prefix, first_suffix = match.groups()
        expanded.append(first)

        for tok in tokens[1:]:
            if re.search(r"\\d", tok):
                expanded.append(tok)
            else:
                expanded.append(f"{prefix}{tok}")

    return _dedupe_preserve_order(expanded)


def is_valid_flywire_orn_label(label: str) -> bool:
    if not isinstance(label, str):
        return False
    value = label.strip()
    if not value:
        return False
    return bool(_FLYWIRE_ORN_RE.match(value))


def _truthy_yes(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"yes", "true", "1", "y"}


def validate_door_to_flywire_mapping(
    mapping_df: pd.DataFrame,
    *,
    expected_door_units: Optional[Sequence[str]] = None,
    require_unique_for_units: Optional[Sequence[str]] = None,
) -> None:
    required_cols = {
        "door_name",
        "flywire_glomerulus",
        "mapping_pathway",
        "source_name",
        "source_year",
        "source_url_or_doi",
        "evidence_note",
        "confidence",
        "is_ambiguous",
    }
    missing = sorted(required_cols - set(mapping_df.columns))
    if missing:
        raise ValueError(f"Mapping CSV missing required columns: {missing}")

    # ORN_ prefix enforcement for every non-empty mapping target.
    bad_targets = []
    for _, row in mapping_df.iterrows():
        target = str(row.get("flywire_glomerulus", "") or "").strip()
        if not target:
            continue
        if not target.startswith("ORN_"):
            bad_targets.append((row.get("door_name"), target))
    if bad_targets:
        example = ", ".join([f"{d}→{t}" for d, t in bad_targets[:5]])
        raise ValueError(f"Invalid FlyWire targets (must start with ORN_). Examples: {example}")

    # Dotted suffix receptors must map to their suffix glomerulus (e.g., Ir64a.DC4 → ORN_DC4).
    dotted_bad: List[Tuple[str, str, str]] = []
    for _, row in mapping_df.iterrows():
        door_name = str(row.get("door_name", "") or "").strip()
        target = str(row.get("flywire_glomerulus", "") or "").strip()
        if not door_name or not target:
            continue
        if "." not in door_name:
            continue
        suffix = door_name.split(".", 1)[1]
        # Only enforce this rule for DoOR's dotted *glomerulus* suffix convention
        # (e.g., Ir64a.DC4, Ir64a.DP1m). DoOR also uses "." to join coexpressed
        # gene identifiers (e.g., Gr21a.Gr63a), which must NOT be treated as
        # a glomerulus suffix.
        if not re.match(r"^[A-Z]+\\d+[a-z]*$", suffix):
            continue
        expected = flywire_glomerulus_from_door_code(suffix)
        if expected and target != expected:
            dotted_bad.append((door_name, suffix, target))
    if dotted_bad:
        ex = dotted_bad[0]
        raise ValueError(
            f"Dotted-suffix mapping violation: {ex[0]} has suffix {ex[1]} but maps to {ex[2]}"
        )

    # Known mandatory fixes.
    def _expect_exact(door: str, expected: str) -> None:
        matches = mapping_df[mapping_df["door_name"].astype(str) == door]
        targets = sorted(set(matches["flywire_glomerulus"].astype(str).str.strip().tolist()))
        targets = [t for t in targets if t]
        if targets != [expected]:
            raise ValueError(f"Required mapping missing/incorrect: {door} must map to {expected}; got {targets}")

    _expect_exact("Or10a", "ORN_DL1")
    _expect_exact("Ir64a.DC4", "ORN_DC4")
    _expect_exact("Ir64a.DP1m", "ORN_DP1m")

    # Conflicts/duplicates handling: multiple targets for the same door_name must be explicitly ambiguous.
    grouped = mapping_df.groupby(mapping_df["door_name"].astype(str), dropna=False)
    for door, grp in grouped:
        if not door or door == "nan":
            continue
        targets = [str(v).strip() for v in grp["flywire_glomerulus"].tolist()]
        targets_nonempty = [t for t in targets if t]
        distinct_targets = sorted(set(targets_nonempty))
        if len(grp) == 1:
            continue

        ambiguous_flags = grp["is_ambiguous"].map(_truthy_yes).tolist()
        if distinct_targets and len(distinct_targets) > 1:
            if not all(ambiguous_flags):
                raise ValueError(
                    f"Conflicting mappings for {door} ({distinct_targets}) without is_ambiguous=Yes"
                )

        # Duplicate identical rows are never allowed (auditability).
        dup_cols = ["door_name", "flywire_glomerulus", "mapping_pathway", "source_name", "source_year", "source_url_or_doi"]
        if grp.duplicated(subset=dup_cols).any():
            raise ValueError(f"Duplicate mapping rows detected for {door}; dedupe required")

        # If a door has multiple rows, it must be explicitly marked ambiguous.
        if not all(ambiguous_flags):
            raise ValueError(f"{door} has multiple rows but is_ambiguous is not Yes for all rows")

    if expected_door_units is not None:
        missing_units = sorted(set(expected_door_units) - set(mapping_df["door_name"].astype(str).tolist()))
        if missing_units:
            raise ValueError(f"Mapping is missing DoOR units: {missing_units[:10]}")

    if require_unique_for_units is not None:
        for unit in require_unique_for_units:
            n = int((mapping_df["door_name"].astype(str) == unit).sum())
            if n != 1:
                raise ValueError(f"{unit} must have exactly 1 mapping row; found {n}")


def build_authoritative_door_to_flywire_mapping(
    door_units: Sequence[str],
    *,
    door_mappings_df: pd.DataFrame,
    manual_overrides_df: Optional[pd.DataFrame] = None,
    sensillum_reference_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the authoritative mapping table for the provided DoOR units.

    Strategy (precedence):
      1) Manual overrides (explicit provenance; pathway=manual_override)
      2) DoOR.mappings glomerulus codes (pathway=door_mappings)
      3) Sensillum reference table (pathway=sensillum_reference)
      4) Dotted-suffix inference (pathway=dotted_suffix_inference)
      5) Unmapped / excluded (pathway=unmapped/excluded_larval)
    """
    paths = MappingPaths()

    manual_overrides_df = manual_overrides_df if manual_overrides_df is not None else pd.DataFrame()
    sensillum_reference_df = sensillum_reference_df if sensillum_reference_df is not None else pd.DataFrame()

    def _is_sensillum_unit(name: str) -> bool:
        lowered = str(name).strip().lower()
        return bool(re.match(r"^(ab|ac|pb|at)\\d", lowered)) or lowered.startswith("ac")  # ac1BC, ac3_noOr35a

    # Normalize lookups
    manual_lookup: Dict[str, pd.DataFrame] = {}
    if not manual_overrides_df.empty:
        for k, grp in manual_overrides_df.groupby(manual_overrides_df["door_name"].map(normalize_receptor_identifier)):
            manual_lookup[k] = grp.copy()

    sensillum_lookup: Dict[str, pd.DataFrame] = {}
    if not sensillum_reference_df.empty:
        for k, grp in sensillum_reference_df.groupby(sensillum_reference_df["door_name"].map(normalize_receptor_identifier)):
            sensillum_lookup[k] = grp.copy()

    # DoOR.mappings lookup: keep all rows for a receptor (there can be duplicates).
    if "receptor" not in door_mappings_df.columns:
        raise ValueError("door_mappings_df must contain column 'receptor'")

    door_mappings_by_key: Dict[str, pd.DataFrame] = {}
    for k, grp in door_mappings_df.groupby(
        door_mappings_df["receptor"].map(normalize_receptor_identifier),
        observed=False,
    ):
        door_mappings_by_key[k] = grp.copy()

    # Larval-only set (DoOR adult=False & larva=True)
    larval_only_set = {normalize_receptor_identifier(r) for r in larval_only_receptors_from_door_mappings(door_mappings_df)}

    records: List[Dict[str, object]] = []

    for door_name in door_units:
        key = normalize_receptor_identifier(door_name)

        is_larval = key in larval_only_set
        life_stage = "Larval" if is_larval else "Adult"

        door_comment_context = ""
        door_glomerulus_context = ""

        # 1) Manual overrides
        if key in manual_lookup:
            for _, row in manual_lookup[key].iterrows():
                rec = {c: row.get(c, "") for c in manual_overrides_df.columns}
                rec.setdefault("door_name", door_name)
                rec.setdefault("mapping_pathway", paths.manual_override)
                rec.setdefault("life_stage", life_stage)
                rec.setdefault("is_larval", "Yes" if is_larval else "No")
                rec.setdefault("is_ambiguous", row.get("is_ambiguous", "No"))
                records.append(rec)
            continue

        # 1b) Sensillum-class mapping via curated reference table (pathway b)
        if _is_sensillum_unit(door_name) and key in sensillum_lookup:
            for _, row in sensillum_lookup[key].iterrows():
                rec = {c: row.get(c, "") for c in sensillum_reference_df.columns}
                rec.setdefault("door_name", door_name)
                rec.setdefault("mapping_pathway", paths.sensillum_reference)
                rec.setdefault("life_stage", life_stage)
                rec.setdefault("is_larval", "Yes" if is_larval else "No")
                rec.setdefault("is_ambiguous", row.get("is_ambiguous", "No"))
                records.append(rec)
            continue

        # Larval-only units are tracked but excluded from adult FlyWire mapping by default.
        if is_larval:
            records.append(
                {
                    "door_name": door_name,
                    "flywire_glomerulus": "",
                    "mapping_pathway": paths.excluded_larval,
                    "source_name": "DoOR.mappings (larval-only flag)",
                    "source_year": 2016,
                    "source_url_or_doi": DOOR_2_DOI,
                    "evidence_note": "Larval-only responding unit in DoOR.mappings (adult=False, larva=True); excluded from adult FlyWire analyses.",
                    "confidence": "high",
                    "is_ambiguous": "No",
                    "life_stage": life_stage,
                    "is_larval": "Yes",
                }
            )
            continue

        # 2) DoOR.mappings (glomerulus codes)
        door_rows = door_mappings_by_key.get(key)
        if door_rows is not None and not door_rows.empty and "glomerulus" in door_rows.columns:
            if "comment" in door_rows.columns:
                comments = [str(c or "").strip() for c in door_rows["comment"].tolist() if str(c or "").strip()]
                if comments:
                    door_comment_context = " | ".join(_dedupe_preserve_order(comments))

            gloms_raw = _dedupe_preserve_order([str(g or "").strip() for g in door_rows["glomerulus"].tolist() if str(g or "").strip()])
            door_glomerulus_context = ", ".join(gloms_raw) if gloms_raw else ""

            # If DoOR has a single glomerulus code, use it (expand if it encodes ambiguity).
            if gloms_raw:
                expanded_codes: List[str] = []
                for g in gloms_raw:
                    expanded_codes.extend(expand_door_glomerulus_codes(g) or [])
                expanded_codes = _dedupe_preserve_order(expanded_codes)

                if expanded_codes:
                    if len(expanded_codes) == 1:
                        fw = flywire_glomerulus_from_door_code(expanded_codes[0])
                        if fw:
                            records.append(
                                {
                                    "door_name": door_name,
                                    "flywire_glomerulus": fw,
                                    "mapping_pathway": paths.door_mappings,
                                    "source_name": "DoOR.mappings (DoOR.data v2.0.0)",
                                    "source_year": 2016,
                                    "source_url_or_doi": DOOR_2_DOI,
                                    "evidence_note": "Single glomerulus code in DoOR.mappings; converted to FlyWire ORN_<glomerulus> label convention.",
                                    "confidence": "high",
                                    "is_ambiguous": "No",
                                    "life_stage": life_stage,
                                    "is_larval": "No",
                                }
                            )
                            continue
                    else:
                        for code in expanded_codes:
                            fw = flywire_glomerulus_from_door_code(code)
                            if not fw:
                                continue
                            records.append(
                                {
                                    "door_name": door_name,
                                    "flywire_glomerulus": fw,
                                    "mapping_pathway": paths.door_mappings,
                                    "source_name": "DoOR.mappings (DoOR.data v2.0.0)",
                                    "source_year": 2016,
                                    "source_url_or_doi": DOOR_2_DOI,
                                    "evidence_note": f"Ambiguous DoOR glomerulus code '{gloms_raw[0]}' expanded to candidates.",
                                    "confidence": "ambiguous",
                                    "is_ambiguous": "Yes",
                                    "life_stage": life_stage,
                                    "is_larval": "No",
                                }
                            )
                        if any(r["door_name"] == door_name for r in records):
                            continue

        # 3) Sensillum reference fallback (for non-sensillum units that still need help)
        if (not _is_sensillum_unit(door_name)) and key in sensillum_lookup:
            for _, row in sensillum_lookup[key].iterrows():
                rec = {c: row.get(c, "") for c in sensillum_reference_df.columns}
                rec.setdefault("door_name", door_name)
                rec.setdefault("mapping_pathway", paths.sensillum_reference)
                rec.setdefault("life_stage", life_stage)
                rec.setdefault("is_larval", "No")
                rec.setdefault("is_ambiguous", row.get("is_ambiguous", "No"))
                records.append(rec)
            continue

        # 4) Dotted suffix inference (only if not mapped elsewhere)
        if "." in door_name:
            suffix = door_name.split(".", 1)[1]
            fw = flywire_glomerulus_from_door_code(suffix)
            if fw:
                records.append(
                    {
                        "door_name": door_name,
                        "flywire_glomerulus": fw,
                        "mapping_pathway": paths.dotted_suffix_inference,
                        "source_name": "DoOR receptor identifier (dotted suffix)",
                        "source_year": 2016,
                        "source_url_or_doi": DOOR_2_DOI,
                        "evidence_note": f"Inferred from dotted suffix '{suffix}' (DoOR naming convention).",
                        "confidence": "medium",
                        "is_ambiguous": "No",
                        "life_stage": life_stage,
                        "is_larval": "No",
                    }
                )
                continue

        # 5) Unmapped
        if door_glomerulus_context or door_comment_context:
            note_bits = []
            if door_glomerulus_context:
                note_bits.append(f"DoOR.mappings glomerulus: {door_glomerulus_context}")
            if door_comment_context:
                note_bits.append(door_comment_context)
            evidence_note = " | ".join(note_bits)
            source_name = "DoOR.mappings (DoOR.data v2.0.0)"
            source_year = 2016
            source_url_or_doi = DOOR_2_DOI
        else:
            evidence_note = "No FlyWire ORN_<glomerulus> mapping found after applying overrides and DoOR.mappings lookup."
            source_name = ""
            source_year = ""
            source_url_or_doi = ""

        records.append(
            {
                "door_name": door_name,
                "flywire_glomerulus": "",
                "mapping_pathway": paths.unmapped,
                "source_name": source_name,
                "source_year": source_year,
                "source_url_or_doi": source_url_or_doi,
                "evidence_note": evidence_note,
                "confidence": "unknown",
                "is_ambiguous": "No",
                "life_stage": life_stage,
                "is_larval": "No",
            }
        )

    df = pd.DataFrame.from_records(records)

    # Ensure all required columns exist (stable schema).
    for col in (
        "door_name",
        "flywire_glomerulus",
        "mapping_pathway",
        "source_name",
        "source_year",
        "source_url_or_doi",
        "evidence_note",
        "confidence",
        "is_ambiguous",
        "life_stage",
        "is_larval",
    ):
        if col not in df.columns:
            df[col] = ""

    # Deterministic ordering
    df["door_name"] = df["door_name"].astype(str)
    df["flywire_glomerulus"] = df["flywire_glomerulus"].astype(str)
    df = df.sort_values(["door_name", "flywire_glomerulus"], kind="mergesort").reset_index(drop=True)

    validate_door_to_flywire_mapping(
        df,
        expected_door_units=list(door_units),
        require_unique_for_units=["Or10a", "Ir64a.DC4", "Ir64a.DP1m"],
    )

    return df
