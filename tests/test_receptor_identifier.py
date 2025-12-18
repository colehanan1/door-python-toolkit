"""
Tests for receptor identifier normalization utilities.
"""

from door_toolkit.integration.receptor_identifier import (
    normalize_receptor_identifier,
    flywire_glomerulus_from_door_code,
)


def test_normalize_receptor_identifier_basic_case_insensitive():
    assert normalize_receptor_identifier("Or7a") == "OR7A"
    assert normalize_receptor_identifier("OR7A") == "OR7A"
    assert normalize_receptor_identifier("  or7a  ") == "OR7A"


def test_normalize_receptor_identifier_handles_dotted_suffixes():
    assert normalize_receptor_identifier("Ir64a.DC4") == "IR64A.DC4"
    assert normalize_receptor_identifier("IR64A.dc4") == "IR64A.DC4"
    assert normalize_receptor_identifier("Ir64a.DP1m") == "IR64A.DP1M"


def test_normalize_receptor_identifier_handles_paired_receptors():
    assert normalize_receptor_identifier("Gr21a.Gr63a") == "GR21A.GR63A"
    assert normalize_receptor_identifier("GR21A.GR63A") == "GR21A.GR63A"
    assert normalize_receptor_identifier("Gr21a+Gr63a") == "GR21A.GR63A"


def test_normalize_receptor_identifier_strips_orn_prefix_if_present():
    assert normalize_receptor_identifier("ORN_Or7a") == "OR7A"


def test_normalize_receptor_identifier_none_and_empty():
    assert normalize_receptor_identifier(None) == ""
    assert normalize_receptor_identifier("") == ""
    assert normalize_receptor_identifier("   ") == ""


def test_flywire_glomerulus_from_door_code_standardizes_case():
    assert flywire_glomerulus_from_door_code("DM2") == "ORN_DM2"
    assert flywire_glomerulus_from_door_code("dp1m") == "ORN_DP1m"
    assert flywire_glomerulus_from_door_code("VC3l") == "ORN_VC3l"
    assert flywire_glomerulus_from_door_code("V") == "ORN_V"


def test_flywire_glomerulus_from_door_code_rejects_unknown_or_ambiguous():
    assert flywire_glomerulus_from_door_code("") == ""
    assert flywire_glomerulus_from_door_code("?") == ""
    assert flywire_glomerulus_from_door_code("DL2d/v+VC3") == ""
