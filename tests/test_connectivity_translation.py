"""Unit tests for DoOR ↔ FlyWire connectivity namespace translation."""

from types import MethodType

import pandas as pd
import pytest

from door_toolkit.integration import DoORFlyWireIntegrator


def _make_stub_integrator():
    """Create a minimal integrator instance without running __init__."""
    integrator = object.__new__(DoORFlyWireIntegrator)
    integrator.door_matrix = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]],
        index=["Or1a", "Or2a"],
        columns=["odor1", "odor2"]
    )
    integrator.door_to_flywire = {"Or1a": "ORN_A", "Or2a": "ORN_B"}
    integrator.flywire_to_door = {"ORN_A": "Or1a", "ORN_B": "Or2a"}
    integrator.receptor_mapping = pd.DataFrame({
        "door_name": ["Or1a", "Or2a"],
        "flywire_glomerulus": ["ORN_A", "ORN_B"]
    })

    def fake_connectivity_matrix(
        self,
        receptor_list=None,
        pathway_type="inhibitory",
        min_synapses=1
    ):
        del self  # Unused
        return pd.DataFrame(
            [[0, 5], [3, 0]],
            index=["ORN_A", "ORN_B"],
            columns=["ORN_A", "ORN_B"]
        )

    integrator.get_connectivity_matrix = MethodType(fake_connectivity_matrix, integrator)
    integrator.get_mapped_receptors = DoORFlyWireIntegrator.get_mapped_receptors.__get__(
        integrator,
        DoORFlyWireIntegrator
    )

    return integrator


def test_get_connectivity_matrix_door_indexed_translates_names():
    """Connectivity matrix is re-indexed to DoOR receptor names."""
    integrator = _make_stub_integrator()

    translated = integrator.get_connectivity_matrix_door_indexed(threshold=2, pathway_type="all")

    assert list(translated.index) == ["Or1a", "Or2a"]
    assert list(translated.columns) == ["Or1a", "Or2a"]
    assert translated.loc["Or1a", "Or2a"] == 5
    assert translated.loc["Or2a", "Or1a"] == 3


def test_get_connectivity_matrix_door_indexed_requires_mappings():
    """Raises ValueError when no FlyWire-to-DoOR mappings exist."""
    integrator = object.__new__(DoORFlyWireIntegrator)
    integrator.door_matrix = pd.DataFrame(
        [[0.1, 0.2]],
        index=["OrX"],
        columns=["odor1", "odor2"]
    )
    integrator.door_to_flywire = {}
    integrator.flywire_to_door = {}
    integrator.receptor_mapping = pd.DataFrame(columns=["door_name", "flywire_glomerulus"])

    def empty_connectivity(self, receptor_list=None, pathway_type="inhibitory", min_synapses=1):
        del self, receptor_list, pathway_type, min_synapses
        return pd.DataFrame()

    integrator.get_connectivity_matrix = MethodType(empty_connectivity, integrator)
    integrator.get_mapped_receptors = DoORFlyWireIntegrator.get_mapped_receptors.__get__(
        integrator,
        DoORFlyWireIntegrator
    )

    with pytest.raises(ValueError):
        integrator.get_connectivity_matrix_door_indexed()

