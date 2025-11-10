"""
Unit tests for duplicate receptor aggregation handling.
"""

import pytest

from door_toolkit.integration.integrator import DoORFlyWireIntegrator


def test_synapse_count_accuracy():
    """
    Verify that Or85e <-> Or98a is in a biologically plausible synapse range.
    """

    integrator = DoORFlyWireIntegrator()

    connectivity_matrix = integrator.get_connectivity_matrix_door_indexed(
        threshold=1,
        pathway_type="inhibitory"
    )

    if "Or85e" in connectivity_matrix.index and "Or98a" in connectivity_matrix.index:
        synapse_count = connectivity_matrix.at["Or85e", "Or98a"]

        assert 400 < synapse_count < 1200, (
            f"Or85e<->Or98a synapse count ({synapse_count}) is inflated. "
            "Expected: 400-1200."
        )

        print(f"PASS: Or85e<->Or98a = {synapse_count} synapses (correct range)")
    else:
        pytest.skip("Or85e or Or98a not present in connectivity matrix.")


def test_no_duplicate_indices():
    """
    Ensure the connectivity matrix has unique receptor indices and columns.
    """

    integrator = DoORFlyWireIntegrator()
    connectivity_matrix = integrator.get_connectivity_matrix_door_indexed(
        threshold=1,
        pathway_type="all"
    )

    assert not connectivity_matrix.index.duplicated().any(), \
        "Connectivity matrix has duplicate row indices."

    assert not connectivity_matrix.columns.duplicated().any(), \
        "Connectivity matrix has duplicate column indices."

    print(f"PASS: No duplicate indices in {connectivity_matrix.shape} matrix")


if __name__ == "__main__":
    test_synapse_count_accuracy()
    test_no_duplicate_indices()
