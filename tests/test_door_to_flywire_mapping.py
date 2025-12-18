"""
Tests for authoritative DoOR → FlyWire mapping builder and validators.
"""

import pandas as pd
import pytest

from door_toolkit.integration.door_to_flywire_mapping import (
    build_authoritative_door_to_flywire_mapping,
    validate_door_to_flywire_mapping,
)


def _base_required_rows():
    """Rows required by strict validator (hard requirements)."""
    common = {
        "mapping_pathway": "test",
        "source_name": "test_source",
        "source_year": 2016,
        "source_url_or_doi": "10.1038/srep21841",
        "evidence_note": "test",
        "confidence": "high",
        "is_ambiguous": "No",
    }
    return [
        {**common, "door_name": "Or10a", "flywire_glomerulus": "ORN_DL1"},
        {**common, "door_name": "Ir64a.DC4", "flywire_glomerulus": "ORN_DC4"},
        {**common, "door_name": "Ir64a.DP1m", "flywire_glomerulus": "ORN_DP1m"},
    ]


def test_validate_requires_orn_prefix_for_nonempty_targets():
    rows = _base_required_rows()
    rows.append(
        {
            "door_name": "Or7a",
            "flywire_glomerulus": "DL5",  # invalid: missing ORN_ prefix
            "mapping_pathway": "test",
            "source_name": "test",
            "source_year": 2016,
            "source_url_or_doi": "10.1038/srep21841",
            "evidence_note": "bad target",
            "confidence": "high",
            "is_ambiguous": "No",
        }
    )

    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="must start with ORN_"):
        validate_door_to_flywire_mapping(df)


def test_validate_requires_is_ambiguous_for_conflicts():
    rows = _base_required_rows()
    rows.extend(
        [
            {
                "door_name": "ac1",
                "flywire_glomerulus": "ORN_VL1",
                "mapping_pathway": "test",
                "source_name": "test",
                "source_year": 2016,
                "source_url_or_doi": "10.1038/srep21841",
                "evidence_note": "candidate 1",
                "confidence": "ambiguous",
                "is_ambiguous": "No",  # invalid: conflicts without ambiguity marking
            },
            {
                "door_name": "ac1",
                "flywire_glomerulus": "ORN_VM1",
                "mapping_pathway": "test",
                "source_name": "test",
                "source_year": 2016,
                "source_url_or_doi": "10.1038/srep21841",
                "evidence_note": "candidate 2",
                "confidence": "ambiguous",
                "is_ambiguous": "Yes",
            },
        ]
    )

    df = pd.DataFrame(rows)
    with pytest.raises(ValueError, match="without is_ambiguous=Yes"):
        validate_door_to_flywire_mapping(df)


def test_build_authoritative_applies_manual_overrides_and_larval_exclusion():
    door_units = ["Or10a", "Ir64a.DC4", "Ir64a.DP1m", "Or7a", "Or1a", "ac1"]

    door_mappings_df = pd.DataFrame(
        [
            {"receptor": "Or10a", "glomerulus": "VC3l", "adult": True, "larva": None},
            {"receptor": "Ir64a.DC4", "glomerulus": "DC4", "adult": True, "larva": None},
            {"receptor": "Ir64a.DP1m", "glomerulus": "DP1m", "adult": True, "larva": None},
            {"receptor": "Or7a", "glomerulus": "DL5", "adult": True, "larva": None},
            {"receptor": "Or1a", "glomerulus": "", "adult": False, "larva": True},
        ]
    )

    manual_overrides_df = pd.DataFrame(
        [
            {
                "door_name": "Or10a",
                "flywire_glomerulus": "ORN_DL1",
                "source_name": "manual",
                "source_year": 2016,
                "source_url_or_doi": "10.1038/srep21841",
                "evidence_note": "override",
                "confidence": "high",
                "is_ambiguous": "No",
            }
        ]
    )

    sensillum_reference_df = pd.DataFrame(
        [
            {
                "door_name": "ac1",
                "flywire_glomerulus": "ORN_VL1",
                "source_name": "sensillum_ref",
                "source_year": 2016,
                "source_url_or_doi": "10.1038/srep21841",
                "evidence_note": "candidate VL1",
                "confidence": "ambiguous",
                "is_ambiguous": "Yes",
            },
            {
                "door_name": "ac1",
                "flywire_glomerulus": "ORN_VM1",
                "source_name": "sensillum_ref",
                "source_year": 2016,
                "source_url_or_doi": "10.1038/srep21841",
                "evidence_note": "candidate VM1",
                "confidence": "ambiguous",
                "is_ambiguous": "Yes",
            },
        ]
    )

    df = build_authoritative_door_to_flywire_mapping(
        door_units,
        door_mappings_df=door_mappings_df,
        manual_overrides_df=manual_overrides_df,
        sensillum_reference_df=sensillum_reference_df,
    )

    or10a = df[df["door_name"] == "Or10a"].iloc[0]
    assert or10a["flywire_glomerulus"] == "ORN_DL1"
    assert or10a["mapping_pathway"] == "manual_override"

    or1a = df[df["door_name"] == "Or1a"].iloc[0]
    assert or1a["mapping_pathway"] == "excluded_larval"
    assert or1a["flywire_glomerulus"] == ""
    assert or1a["life_stage"] == "Larval"

    ac1_rows = df[df["door_name"] == "ac1"]
    assert len(ac1_rows) == 2
    assert set(ac1_rows["flywire_glomerulus"]) == {"ORN_VL1", "ORN_VM1"}
    assert set(ac1_rows["is_ambiguous"]) == {"Yes"}

