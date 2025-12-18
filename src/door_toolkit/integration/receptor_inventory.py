"""
Receptor Inventory (Single Source of Truth)
===========================================

Builds a unified receptor inventory table that merges:
- DoOR receptor set (from the cached response matrix)
- Receptor → FlyWire glomerulus mapping (normalization-safe)
- Adult/larval life-stage metadata
- Per-receptor FlyWire connectivity coverage metrics

This table is intended to be the canonical "source of truth" artifact written to:
`data/mappings/receptor_inventory.csv` by `scripts/generate_receptor_inventory.py`.

Scientific sources:
- DoOR 2.0 receptor set and nomenclature: Münch & Galizia (2016) Scientific Data.
- FlyWire naming conventions for glomerulus labels: FlyWire official annotations repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from door_toolkit.integration.mapping_accounting import is_larval_receptor
from door_toolkit.integration.receptor_identifier import normalize_receptor_identifier


CONNECTIVITY_COLUMNS: Tuple[str, ...] = (
    "n_orns",
    "n_pns_reached",
    "pct_pns_reached",
    "total_synapses_to_pns",
    "n_kcs_reached",
    "pct_kcs_reached",
    "pathway_intact",
)

REQUIRED_BASE_COLUMNS: Tuple[str, ...] = (
    "receptor_name",
    "is_mapped",
    "flywire_glomerulus",
    "life_stage",
    "is_larval",
    "status",
    "notes",
)


@dataclass(frozen=True)
class MappingResult:
    flywire_glomerulus: str
    mapping_source: str
    notes: str
    is_ambiguous: bool = False
    confidence: str = ""


def _build_connectivity_lookup(connectivity_df: Optional[pd.DataFrame]) -> Dict[str, Dict[str, Any]]:
    if connectivity_df is None or connectivity_df.empty:
        return {}

    if "receptor" not in connectivity_df.columns:
        raise ValueError("connectivity_df must contain a 'receptor' column")

    lookup: Dict[str, Dict[str, Any]] = {}
    for _, row in connectivity_df.iterrows():
        key = normalize_receptor_identifier(row.get("receptor"))
        if not key:
            continue
        lookup[key] = {col: row.get(col) for col in CONNECTIVITY_COLUMNS if col in connectivity_df.columns}
    return lookup


def _build_mapping_rows_lookup(mapping_df: Optional[pd.DataFrame]) -> Dict[str, List[Dict[str, Any]]]:
    if mapping_df is None or mapping_df.empty:
        return {}

    if "door_name" not in mapping_df.columns or "flywire_glomerulus" not in mapping_df.columns:
        raise ValueError("mapping_df must contain columns: 'door_name', 'flywire_glomerulus'")

    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in mapping_df.iterrows():
        key = normalize_receptor_identifier(row.get("door_name"))
        if not key:
            continue
        lookup.setdefault(key, []).append(row.to_dict())
    return lookup


def _truthy_yes(value: object) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"yes", "true", "1", "y"}


def resolve_receptor_mapping_from_authoritative_table(
    receptor_name: str,
    *,
    mapping_rows_lookup: Dict[str, List[Dict[str, Any]]],
) -> MappingResult:
    """
    Resolve receptor → FlyWire glomerulus from the authoritative mapping artifact.

    Rules:
    - Mapped means: exactly one non-empty FlyWire target AND it begins with 'ORN_'.
    - If multiple targets exist (or is_ambiguous is flagged), treat as ambiguous and
      return unmapped with a notes field listing candidates.
    """
    key = normalize_receptor_identifier(receptor_name)
    rows = mapping_rows_lookup.get(key, [])

    if not rows:
        return MappingResult(flywire_glomerulus="", mapping_source="", notes="")

    targets = [str(r.get("flywire_glomerulus", "") or "").strip() for r in rows]
    targets = [t for t in targets if t]
    distinct_targets = sorted(set(targets))

    any_ambiguous_flag = any(_truthy_yes(r.get("is_ambiguous")) for r in rows)

    # Prefer evidence_note/source_name schema; fall back to legacy "notes"/"source" if present.
    def _row_source(row: Dict[str, Any]) -> str:
        source_name = str(row.get("source_name", "") or "").strip()
        year = str(row.get("source_year", "") or "").strip()
        doi = str(row.get("source_url_or_doi", "") or "").strip()
        source = source_name or str(row.get("source", "") or "").strip()
        if year and source:
            source = f"{source} ({year})"
        if doi and source:
            source = f"{source} [{doi}]"
        return source

    def _row_note(row: Dict[str, Any]) -> str:
        return str(row.get("evidence_note", "") or row.get("notes", "") or "").strip()

    if any_ambiguous_flag or len(distinct_targets) > 1:
        sources = sorted({s for s in (_row_source(r) for r in rows) if s})
        notes = sorted({n for n in (_row_note(r) for r in rows) if n})
        combined_notes = "; ".join(notes)
        candidate_list = ", ".join(distinct_targets)
        if combined_notes:
            combined_notes = f"{combined_notes} | Candidates: {candidate_list}"
        else:
            combined_notes = f"Candidates: {candidate_list}"
        return MappingResult(
            flywire_glomerulus="",
            mapping_source="; ".join(sources),
            notes=combined_notes,
            is_ambiguous=True,
            confidence="ambiguous",
        )

    if len(distinct_targets) == 1 and distinct_targets[0].startswith("ORN_"):
        # Choose the first row that provides this target for provenance fields.
        chosen = next((r for r in rows if str(r.get("flywire_glomerulus", "") or "").strip() == distinct_targets[0]), rows[0])
        return MappingResult(
            flywire_glomerulus=distinct_targets[0],
            mapping_source=_row_source(chosen),
            notes=_row_note(chosen),
            is_ambiguous=False,
            confidence=str(chosen.get("confidence", "") or "").strip(),
        )

    # Non-empty targets exist but are not valid ORN_ labels; treat as unmapped.
    notes = "; ".join(sorted({n for n in (_row_note(r) for r in rows) if n}))
    return MappingResult(flywire_glomerulus="", mapping_source="; ".join(sorted({_row_source(r) for r in rows if _row_source(r)})), notes=notes)


def get_larval_receptors_present(door_receptors: Iterable[str]) -> List[str]:
    """Return sorted larval-only receptors present in the provided receptor list."""
    present = [r for r in door_receptors if is_larval_receptor(r)]
    return sorted(present)


def build_receptor_inventory_dataframe(
    door_receptors: List[str],
    *,
    mapping_df: Optional[pd.DataFrame] = None,
    connectivity_df: Optional[pd.DataFrame] = None,
    include_mapping_source_column: bool = True,
) -> pd.DataFrame:
    """
    Build the unified receptor inventory DataFrame.

    Args:
        door_receptors: DoOR receptor identifiers (canonical DoOR naming for output).
        mapping_df: Authoritative DoOR→FlyWire mapping artifact (door_name → ORN_<glomerulus>).
        connectivity_df: Per-receptor FlyWire connectivity metrics table.
        include_mapping_source_column: If True, includes a 'mapping_source' provenance column.
    """
    mapping_lookup = _build_mapping_rows_lookup(mapping_df)
    connectivity_lookup = _build_connectivity_lookup(connectivity_df)

    records: List[Dict[str, Any]] = []

    for receptor in door_receptors:
        mapping_result = resolve_receptor_mapping_from_authoritative_table(
            receptor,
            mapping_rows_lookup=mapping_lookup,
        )

        mapped = bool(mapping_result.flywire_glomerulus) and mapping_result.flywire_glomerulus.startswith("ORN_")
        larval = is_larval_receptor(receptor)

        if larval:
            life_stage = "Larval"
            status = (
                "Mapped (Larval - exclude from adult analysis)"
                if mapped
                else "Unmapped (Larval)"
            )
        else:
            life_stage = "Adult"
            if mapped:
                status = "Mapped (Adult)"
            elif mapping_result.is_ambiguous:
                status = "Ambiguous (Adult - exclude or resolve)"
            else:
                status = "Unmapped (Adult - NEEDS MAPPING)"

        connectivity = connectivity_lookup.get(normalize_receptor_identifier(receptor), {})

        notes = (mapping_result.notes or "").strip()
        if (not larval) and (not mapped) and (not mapping_result.is_ambiguous):
            if not notes:
                notes = "TODO: No FlyWire ORN_<glomerulus> mapping after normalization"
            elif not notes.startswith("TODO:"):
                notes = f"TODO: {notes}"

        record: Dict[str, Any] = {
            "receptor_name": receptor,
            "is_mapped": "Yes" if mapped else "No",
            "flywire_glomerulus": mapping_result.flywire_glomerulus,
            "life_stage": life_stage,
            "is_larval": "Yes" if larval else "No",
            "status": status,
            "notes": notes,
        }

        if include_mapping_source_column:
            record["mapping_source"] = mapping_result.mapping_source

        # Attach connectivity columns (always present in output schema)
        for col in CONNECTIVITY_COLUMNS:
            record[col] = connectivity.get(col, 0 if col != "pathway_intact" else False)

        records.append(record)

    df = pd.DataFrame(records)

    # Column order: required columns first, then optional provenance, then connectivity.
    columns: List[str] = list(REQUIRED_BASE_COLUMNS)
    if include_mapping_source_column and "mapping_source" in df.columns:
        columns.append("mapping_source")
    columns.extend(CONNECTIVITY_COLUMNS)

    # Add any missing columns explicitly so downstream code can rely on them.
    for col in columns:
        if col not in df.columns:
            df[col] = "" if col not in CONNECTIVITY_COLUMNS else (False if col == "pathway_intact" else 0)

    df = df[columns]

    return df


def validate_inventory_schema(df: pd.DataFrame) -> None:
    """Raise ValueError if required inventory columns are missing."""
    missing = [c for c in REQUIRED_BASE_COLUMNS + CONNECTIVITY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Inventory DataFrame missing required columns: {missing}")
