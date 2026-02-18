"""
Tests for glomerulus feature construction and weight vector regression.

All tests use synthetic fixtures — no internet or DoOR cache required.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from door_toolkit.glomerulus_features import (
    build_design_matrix,
    load_receptor_to_glomerulus_mapping,
    odor_to_receptor_vector,
    receptor_vector_to_glomerulus_vector,
)
from door_toolkit.glomerulus_regression import (
    export_weight_report,
    fit_glomerulus_weight_vector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_mock_encoder(receptor_names, odor_vectors):
    """Create a mock encoder with deterministic encode() and receptor_names."""
    enc = SimpleNamespace()
    enc.receptor_names = receptor_names
    # Store as dict: odor_name_lower → vector
    _lookup = {name.lower(): vec for name, vec in odor_vectors.items()}

    def encode(odor_name, fill_missing=0.0):
        key = odor_name.lower()
        if key not in _lookup:
            raise KeyError(f"Odorant '{odor_name}' not found")
        return _lookup[key]

    enc.encode = encode
    return enc


@pytest.fixture
def simple_mapping():
    """5 receptors mapped to themselves (DoOR names only, no FlyWire conversion)."""
    return {
        "Or1": "Or1",
        "Or2": "Or2",
        "Or3": "Or3",
        "Or4": "Or4",
        "Or5": "Or5",
    }


@pytest.fixture
def simple_receptor_names():
    return ["Or1", "Or2", "Or3", "Or4", "Or5", "OrX"]  # OrX is unmapped


@pytest.fixture
def simple_receptor_vector():
    return np.array([0.2, 0.8, 0.5, 0.1, 0.9, 0.3], dtype=np.float32)


@pytest.fixture
def mock_encoder():
    """Mock encoder with 6 receptors, 4 odors with distinct activation patterns."""
    receptor_names = ["Or1", "Or2", "Or3", "Or4", "Or5", "OrX"]
    odor_vectors = {
        "odor_a": np.array([0.9, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "odor_b": np.array([0.0, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float32),
        "odor_c": np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.0], dtype=np.float32),
        "odor_d": np.array([0.0, 0.0, 0.0, 0.3, 0.7, 0.0], dtype=np.float32),
    }
    return _make_mock_encoder(receptor_names, odor_vectors)


@pytest.fixture
def mapping_csv(tmp_path):
    """Write a minimal mapping CSV for testing load_receptor_to_glomerulus_mapping."""
    csv_path = tmp_path / "mapping.csv"
    df = pd.DataFrame({
        "door_name": ["Or1", "Or2", "Or3", "Or4", "Or5", "OrLarval"],
        "flywire_glomerulus": ["ORN_A", "ORN_A", "ORN_B", "ORN_C", "ORN_C", "ORN_L"],
        "mapping_pathway": ["door_mappings"] * 6,
        "confidence": ["high"] * 6,
        "is_ambiguous": ["No"] * 6,
        "life_stage": ["Adult", "Adult", "Adult", "Adult", "Adult", "Larval"],
        "is_larval": ["No", "No", "No", "No", "No", "Yes"],
    })
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Tests: load_receptor_to_glomerulus_mapping
# ---------------------------------------------------------------------------

class TestLoadMapping:
    def test_adult_only_excludes_larval(self, mapping_csv):
        mapping, meta = load_receptor_to_glomerulus_mapping(str(mapping_csv), adult_only=True)
        assert "OrLarval" not in mapping
        assert meta["n_receptors_mapped"] == 5  # Or1-Or5

    def test_include_larval(self, mapping_csv):
        mapping, meta = load_receptor_to_glomerulus_mapping(str(mapping_csv), adult_only=False)
        assert "OrLarval" in mapping
        assert meta["n_receptors_mapped"] == 6

    def test_meta_has_all_receptors(self, mapping_csv):
        _, meta = load_receptor_to_glomerulus_mapping(str(mapping_csv))
        assert set(meta["all_receptors"]) == {"Or1", "Or2", "Or3", "Or4", "Or5"}


# ---------------------------------------------------------------------------
# Tests: receptor_vector_to_glomerulus_vector
# ---------------------------------------------------------------------------

class TestReceptorToGlomerulus:
    def test_receptor_filter(self, simple_mapping, simple_receptor_names, simple_receptor_vector):
        gvec, gnames, meta = receptor_vector_to_glomerulus_vector(
            simple_receptor_vector, simple_receptor_names, simple_mapping, agg="max"
        )
        # Should output mapped receptors in sorted order (Or1, Or2, Or3, Or4, Or5)
        assert gnames == ["Or1", "Or2", "Or3", "Or4", "Or5"]
        # Responses should match input vector for mapped receptors
        assert np.isclose(gvec[0], 0.2)  # Or1
        assert np.isclose(gvec[4], 0.9)  # Or5

    def test_aggregation_ignored(self, simple_mapping, simple_receptor_names, simple_receptor_vector):
        # With identity mapping, agg parameter is ignored
        gvec_max, _, _ = receptor_vector_to_glomerulus_vector(
            simple_receptor_vector, simple_receptor_names, simple_mapping, agg="max"
        )
        gvec_mean, _, _ = receptor_vector_to_glomerulus_vector(
            simple_receptor_vector, simple_receptor_names, simple_mapping, agg="mean"
        )
        # Should be identical (no aggregation)
        np.testing.assert_array_almost_equal(gvec_max, gvec_mean)

    def test_sum_aggregation_ignored(self, simple_mapping, simple_receptor_names, simple_receptor_vector):
        # agg parameter should be ignored for identity mapping
        gvec_max, _, _ = receptor_vector_to_glomerulus_vector(
            simple_receptor_vector, simple_receptor_names, simple_mapping, agg="max"
        )
        gvec_sum, _, _ = receptor_vector_to_glomerulus_vector(
            simple_receptor_vector, simple_receptor_names, simple_mapping, agg="sum"
        )
        # Should be identical (no aggregation)
        np.testing.assert_array_almost_equal(gvec_max, gvec_sum)

    def test_unmapped_receptors_in_meta(self, simple_mapping, simple_receptor_names, simple_receptor_vector):
        _, _, meta = receptor_vector_to_glomerulus_vector(
            simple_receptor_vector, simple_receptor_names, simple_mapping, agg="max"
        )
        assert "OrX" in meta["unmapped_receptors"]
        assert meta["n_receptors_unmapped"] == 1

    def test_mapping_coverage(self, simple_mapping, simple_receptor_names, simple_receptor_vector):
        _, _, meta = receptor_vector_to_glomerulus_vector(
            simple_receptor_vector, simple_receptor_names, simple_mapping, agg="max"
        )
        assert meta["mapping_coverage_frac"] == pytest.approx(5 / 6)


# ---------------------------------------------------------------------------
# Tests: build_design_matrix
# ---------------------------------------------------------------------------

class TestBuildDesignMatrix:
    def test_all_returns_all_receptors(self, mock_encoder, simple_mapping):
        X, gnames, meta = build_design_matrix(
            ["odor_a", "odor_b"], mock_encoder, simple_mapping,
            feature_set="all", activation_threshold=0.05,
        )
        # Should return all 5 mapped receptors
        assert X.shape == (2, 5)
        assert len(gnames) == 5

    def test_union_vs_intersection_differ(self, mock_encoder, simple_mapping):
        X_u, gnames_u, _ = build_design_matrix(
            ["odor_a", "odor_b"], mock_encoder, simple_mapping,
            feature_set="union", activation_threshold=0.05,
        )
        X_i, gnames_i, _ = build_design_matrix(
            ["odor_a", "odor_b"], mock_encoder, simple_mapping,
            feature_set="intersection", activation_threshold=0.05,
        )
        # odor_a activates: Or1=0.9 (>0.05)
        # odor_b activates: Or3=0.8 (>0.05)
        # Union: Or1 + Or3 (receptors active in any odor)
        # Intersection: none (no receptor active in both)
        assert len(gnames_u) >= len(gnames_i)
        assert len(gnames_u) > 0

    def test_per_odor_active_counts_in_meta(self, mock_encoder, simple_mapping):
        _, _, meta = build_design_matrix(
            ["odor_a", "odor_c"], mock_encoder, simple_mapping,
            feature_set="union", activation_threshold=0.05,
        )
        assert "odor_a" in meta["per_odor_active_counts"]
        assert "odor_c" in meta["per_odor_active_counts"]
        assert all(v >= 0 for v in meta["per_odor_active_counts"].values())

    def test_shape_consistency(self, mock_encoder, simple_mapping):
        odors = ["odor_a", "odor_b", "odor_c", "odor_d"]
        X, gnames, _ = build_design_matrix(
            odors, mock_encoder, simple_mapping,
            feature_set="union", activation_threshold=0.05,
        )
        assert X.shape[0] == 4
        assert X.shape[1] == len(gnames)


# ---------------------------------------------------------------------------
# Tests: fit_glomerulus_weight_vector
# ---------------------------------------------------------------------------

class TestFitWeightVector:
    def _make_data(self, mock_encoder, simple_mapping):
        odors = ["odor_a", "odor_b", "odor_c", "odor_d"]
        y = np.array([0.48, 0.07, 0.04, 0.14])
        X, gnames, _ = build_design_matrix(
            odors, mock_encoder, simple_mapping,
            feature_set="all", activation_threshold=0.05,
        )
        return X, y, gnames

    def test_weights_length(self, mock_encoder, simple_mapping):
        X, y, gnames = self._make_data(mock_encoder, simple_mapping)
        results = fit_glomerulus_weight_vector(X, y, gnames, random_state=0)
        assert len(results["weights"]) == len(gnames)

    def test_sign_classification(self, mock_encoder, simple_mapping):
        X, y, gnames = self._make_data(mock_encoder, simple_mapping)
        results = fit_glomerulus_weight_vector(X, y, gnames, random_state=0, zero_eps=1e-6)
        signs = results["weight_signs"]
        # Every weight should be classified as +1, -1, or 0
        assert set(signs).issubset({-1, 0, 1})

    def test_zero_eps_sensitivity(self, mock_encoder, simple_mapping):
        X, y, gnames = self._make_data(mock_encoder, simple_mapping)
        r_tight = fit_glomerulus_weight_vector(X, y, gnames, random_state=0, zero_eps=1e-10)
        r_loose = fit_glomerulus_weight_vector(X, y, gnames, random_state=0, zero_eps=10.0)
        # With eps=10, all weights should be classified as zero
        assert all(s == 0 for s in r_loose["weight_signs"])
        # Tight eps should have at least as many non-zero signs
        assert sum(s != 0 for s in r_tight["weight_signs"]) >= sum(
            s != 0 for s in r_loose["weight_signs"]
        )

    def test_determinism(self, mock_encoder, simple_mapping):
        X, y, gnames = self._make_data(mock_encoder, simple_mapping)
        r1 = fit_glomerulus_weight_vector(X, y, gnames, random_state=0)
        r2 = fit_glomerulus_weight_vector(X, y, gnames, random_state=0)
        np.testing.assert_array_equal(r1["weights"], r2["weights"])
        assert r1["alpha"] == r2["alpha"]
        assert r1["intercept"] == r2["intercept"]

    def test_returns_required_keys(self, mock_encoder, simple_mapping):
        X, y, receptor_names = self._make_data(mock_encoder, simple_mapping)
        results = fit_glomerulus_weight_vector(X, y, receptor_names)
        required = {"weights", "intercept", "alpha", "y_pred", "r2", "mse",
                     "weight_signs", "receptor_names", "model_name", "random_state",
                     "zero_eps", "standardize", "n_samples", "n_features"}
        assert required.issubset(results.keys())


# ---------------------------------------------------------------------------
# Tests: export_weight_report
# ---------------------------------------------------------------------------

class TestExportWeightReport:
    def test_files_created(self, tmp_path, mock_encoder, simple_mapping):
        odors = ["odor_a", "odor_b", "odor_c", "odor_d"]
        y = np.array([0.48, 0.07, 0.04, 0.14])
        X, gnames, _ = build_design_matrix(
            odors, mock_encoder, simple_mapping, feature_set="all",
        )
        results = fit_glomerulus_weight_vector(X, y, gnames)
        outdir = tmp_path / "out"
        paths = export_weight_report(results, str(outdir))

        assert Path(paths["weights_csv"]).exists()
        assert Path(paths["model_summary_json"]).exists()

    def test_weights_csv_content(self, tmp_path, mock_encoder, simple_mapping):
        odors = ["odor_a", "odor_b", "odor_c", "odor_d"]
        y = np.array([0.48, 0.07, 0.04, 0.14])
        X, gnames, _ = build_design_matrix(
            odors, mock_encoder, simple_mapping, feature_set="all",
        )
        results = fit_glomerulus_weight_vector(X, y, gnames)
        outdir = tmp_path / "out"
        paths = export_weight_report(results, str(outdir))

        df = pd.read_csv(paths["weights_csv"])
        assert set(df.columns) == {"receptor", "weight", "sign", "abs_weight", "rank"}
        assert len(df) == len(gnames)
        # Sorted by abs_weight descending
        assert df["abs_weight"].is_monotonic_decreasing

    def test_model_summary_json(self, tmp_path, mock_encoder, simple_mapping):
        odors = ["odor_a", "odor_b", "odor_c", "odor_d"]
        y = np.array([0.48, 0.07, 0.04, 0.14])
        X, gnames, _ = build_design_matrix(
            odors, mock_encoder, simple_mapping, feature_set="all",
        )
        results = fit_glomerulus_weight_vector(X, y, gnames)
        outdir = tmp_path / "out"
        paths = export_weight_report(results, str(outdir), extra_meta={"foo": "bar"})

        with open(paths["model_summary_json"]) as f:
            summary = json.load(f)
        assert "model" in summary
        assert "r2" in summary
        assert "config" in summary
        assert summary["config"]["foo"] == "bar"


# ---------------------------------------------------------------------------
# Tests: Target Centering
# ---------------------------------------------------------------------------

class TestTargetCentering:
    def _make_data(self, mock_encoder, simple_mapping):
        odors = ["odor_a", "odor_b", "odor_c", "odor_d"]
        y = np.array([0.48, 0.07, 0.04, 0.14])
        X, gnames, _ = build_design_matrix(
            odors, mock_encoder, simple_mapping,
            feature_set="all", activation_threshold=0.05,
        )
        return X, y, gnames

    def test_centered_mode_produces_zero_intercept(self, mock_encoder, simple_mapping):
        """Centered mode should produce intercept ≈ 0."""
        X, y, gnames = self._make_data(mock_encoder, simple_mapping)
        y_mean = float(y.mean())
        y_centered = y - y_mean
        results = fit_glomerulus_weight_vector(
            X, y_centered, gnames,
            target_mode="centered", y_raw=y, y_mean=y_mean,
            random_state=0
        )
        assert abs(results["intercept"]) < 1e-6  # Should be ~ 0
        assert results["target_mode"] == "centered"
        assert results["y_mean"] == pytest.approx(y_mean)
        assert results["y_raw"] == pytest.approx(y.tolist())
        assert results["y_fitted"] == pytest.approx(y_centered.tolist())

    def test_raw_mode_preserves_legacy_behavior(self, mock_encoder, simple_mapping):
        """Raw mode should produce intercept ≈ mean(y)."""
        X, y, gnames = self._make_data(mock_encoder, simple_mapping)
        y_mean = float(y.mean())
        results = fit_glomerulus_weight_vector(
            X, y, gnames,
            target_mode="raw", y_raw=y, y_mean=y_mean,
            random_state=0
        )
        # Intercept should be close to y_mean (within tolerance for small sample)
        assert abs(results["intercept"] - y_mean) < 0.1
        assert results["target_mode"] == "raw"
        assert results["y_raw"] == pytest.approx(y.tolist())
        assert results["y_fitted"] == pytest.approx(y.tolist())

    def test_centered_mode_exports_correctly(self, tmp_path, mock_encoder, simple_mapping):
        """Centered mode metadata should be in model_summary.json."""
        X, y, gnames = self._make_data(mock_encoder, simple_mapping)
        y_mean = float(y.mean())
        y_centered = y - y_mean
        results = fit_glomerulus_weight_vector(
            X, y_centered, gnames,
            target_mode="centered", y_raw=y, y_mean=y_mean,
            random_state=0
        )
        outdir = tmp_path / "out"
        paths = export_weight_report(results, str(outdir))

        with open(paths["model_summary_json"]) as f:
            summary = json.load(f)
        assert summary["target_mode"] == "centered"
        assert summary["y_mean"] == pytest.approx(y_mean)
        assert summary["y_raw"] == pytest.approx(y.tolist())
        assert summary["y_fitted"] == pytest.approx(y_centered.tolist())
        assert abs(summary["intercept"]) < 1e-6
