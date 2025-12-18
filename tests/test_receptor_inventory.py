"""
Tests for unified receptor inventory construction.
"""

import pandas as pd

from door_toolkit.integration.receptor_inventory import (
    build_receptor_inventory_dataframe,
    validate_inventory_schema,
)


def test_build_receptor_inventory_dataframe_schema_and_connectivity_defaults():
    door_receptors = ["Or7a", "Or33b"]

    connectivity_df = pd.DataFrame(
        [
            {
                "receptor": "Or7a",
                "n_orns": 10,
                "n_pns_reached": 2,
                "pct_pns_reached": 0.5,
                "total_synapses_to_pns": 123.0,
                "n_kcs_reached": 20,
                "pct_kcs_reached": 3.0,
                "pathway_intact": True,
            }
        ]
    )

    mapping_df = pd.DataFrame(
        [
            {
                "door_name": "OR7A",  # intentionally different capitalization
                "flywire_glomerulus": "ORN_DL5",
                "source": "manual_table",
                "notes": "test",
            }
        ]
    )

    df = build_receptor_inventory_dataframe(
        door_receptors,
        mapping_df=mapping_df,
        connectivity_df=connectivity_df,
        include_mapping_source_column=True,
    )

    validate_inventory_schema(df)
    assert set(door_receptors) == set(df["receptor_name"])

    or7a = df[df["receptor_name"] == "Or7a"].iloc[0]
    assert or7a["is_mapped"] == "Yes"
    assert or7a["flywire_glomerulus"] == "ORN_DL5"
    assert or7a["n_orns"] == 10
    assert bool(or7a["pathway_intact"]) is True

    or33b = df[df["receptor_name"] == "Or33b"].iloc[0]
    assert or33b["is_mapped"] == "No"
    assert or33b["flywire_glomerulus"] == ""
    assert or33b["n_orns"] == 0
    assert bool(or33b["pathway_intact"]) is False


def test_ambiguous_mapping_is_not_marked_mapped():
    door_receptors = ["Or7a"]

    mapping_df = pd.DataFrame(
        [
            {"door_name": "Or7a", "flywire_glomerulus": "ORN_DL5", "is_ambiguous": "Yes"},
            {"door_name": "Or7a", "flywire_glomerulus": "ORN_FAKE", "is_ambiguous": "Yes"},
        ]
    )

    df = build_receptor_inventory_dataframe(
        door_receptors,
        mapping_df=mapping_df,
        connectivity_df=None,
        include_mapping_source_column=True,
    )

    row = df.iloc[0]
    assert row["is_mapped"] == "No"
    assert row["flywire_glomerulus"] == ""
    assert str(row["status"]).startswith("Ambiguous")
