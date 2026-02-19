"""Tests for door_toolkit.similarity module."""

import numpy as np
import pandas as pd
import pytest

from door_toolkit.similarity import (
    find_similar,
    get_measured_mask,
    pairwise_similarity,
    similarity_matrix,
)


# ---------------------------------------------------------------------------
# get_measured_mask
# ---------------------------------------------------------------------------


class TestGetMeasuredMask:
    def test_all_measured(self):
        vec = np.array([1.0, 2.0, 3.0])
        mask = get_measured_mask(vec)
        assert mask.dtype == bool
        np.testing.assert_array_equal(mask, [True, True, True])

    def test_some_nan(self):
        vec = np.array([1.0, np.nan, 3.0, np.nan])
        mask = get_measured_mask(vec)
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_all_nan(self):
        vec = np.array([np.nan, np.nan])
        mask = get_measured_mask(vec)
        np.testing.assert_array_equal(mask, [False, False])


# ---------------------------------------------------------------------------
# pairwise_similarity – known values
# ---------------------------------------------------------------------------


class TestPairwiseSimilarityKnown:
    """Verify correctness on small hand-computed vectors (no NaN)."""

    def test_cosine_identical(self):
        v = np.array([1.0, 2.0, 3.0])
        res = pairwise_similarity(v, v, method="cosine", min_overlap=1)
        assert res["skipped"] is False
        assert pytest.approx(res["similarity"], abs=1e-9) == 1.0
        assert res["overlap_n"] == 3

    def test_cosine_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        res = pairwise_similarity(a, b, method="cosine", min_overlap=1)
        assert pytest.approx(res["similarity"], abs=1e-9) == 0.0

    def test_cosine_opposite(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        res = pairwise_similarity(a, b, method="cosine", min_overlap=1)
        assert pytest.approx(res["similarity"], abs=1e-9) == -1.0

    def test_pearson_perfect(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 4.0, 6.0])
        res = pairwise_similarity(a, b, method="pearson", min_overlap=1)
        assert pytest.approx(res["similarity"], abs=1e-9) == 1.0

    def test_pearson_negative(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([3.0, 2.0, 1.0])
        res = pairwise_similarity(a, b, method="pearson", min_overlap=1)
        assert pytest.approx(res["similarity"], abs=1e-9) == -1.0

    def test_spearman_perfect(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([10.0, 20.0, 30.0, 40.0])
        res = pairwise_similarity(a, b, method="spearman", min_overlap=1)
        assert pytest.approx(res["similarity"], abs=1e-9) == 1.0

    def test_unknown_method_raises(self):
        a = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="Unknown similarity method"):
            pairwise_similarity(a, a, method="hamming", min_overlap=1)


# ---------------------------------------------------------------------------
# pairwise_similarity – masking
# ---------------------------------------------------------------------------


class TestPairwiseMasking:
    def test_pairwise_uses_shared_dims(self):
        # a measured at [0, 1, 2]; b measured at [1, 2, 3]
        a = np.array([1.0, 2.0, 3.0, np.nan])
        b = np.array([np.nan, 2.0, 3.0, 4.0])
        res = pairwise_similarity(a, b, method="cosine", min_overlap=1)
        assert res["overlap_n"] == 2  # indices 1 and 2
        assert not res["skipped"]
        # Cosine of [2,3] vs [2,3] = 1.0
        assert pytest.approx(res["similarity"], abs=1e-9) == 1.0

    def test_global_mask_applied(self):
        a = np.array([1.0, 2.0, 3.0, 4.0])
        b = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        res = pairwise_similarity(
            a, b, method="cosine", mask_mode="global",
            global_mask=mask, min_overlap=1,
        )
        assert res["overlap_n"] == 2
        assert res["mask_mode"] == "global"

    def test_global_mask_required(self):
        a = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match="global_mask is required"):
            pairwise_similarity(a, a, mask_mode="global", min_overlap=1)


# ---------------------------------------------------------------------------
# min_overlap guard
# ---------------------------------------------------------------------------


class TestMinOverlap:
    def test_skip_when_overlap_too_small(self):
        a = np.array([1.0, np.nan, np.nan])
        b = np.array([1.0, np.nan, np.nan])
        res = pairwise_similarity(a, b, method="cosine", min_overlap=3)
        assert res["skipped"] is True
        assert np.isnan(res["similarity"])
        assert "Overlap (1) < min_overlap (3)" in res["reason"]

    def test_no_skip_when_overlap_sufficient(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        res = pairwise_similarity(a, b, method="cosine", min_overlap=3)
        assert res["skipped"] is False
        assert res["overlap_n"] == 3

    def test_zero_variance_skipped(self):
        a = np.array([5.0, 5.0, 5.0])
        b = np.array([1.0, 2.0, 3.0])
        res = pairwise_similarity(a, b, method="pearson", min_overlap=1)
        assert res["skipped"] is True
        assert "Zero variance" in res["reason"]


# ---------------------------------------------------------------------------
# similarity_matrix
# ---------------------------------------------------------------------------


class TestSimilarityMatrix:
    def test_shape_and_symmetry(self, mock_encoder):
        odors = ["acetic acid", "ethanol", "benzaldehyde"]
        S, meta = similarity_matrix(odors, mock_encoder, method="cosine",
                                    min_overlap=1)
        assert S.shape == (3, 3)
        # Symmetric
        np.testing.assert_array_almost_equal(S.values, S.values.T)
        # Diagonal is 1
        np.testing.assert_array_almost_equal(np.diag(S.values), [1.0, 1.0, 1.0])

    def test_overlap_matrix_shape(self, mock_encoder):
        odors = ["acetic acid", "ethanol"]
        S, meta = similarity_matrix(odors, mock_encoder, method="cosine",
                                    min_overlap=1)
        assert meta["overlap_matrix"].shape == (2, 2)

    def test_coverage_dataframe(self, mock_encoder):
        odors = ["acetic acid", "ethanol"]
        _, meta = similarity_matrix(odors, mock_encoder, method="cosine",
                                    min_overlap=1)
        cov = meta["coverage"]
        assert "n_measured" in cov.columns
        assert "coverage_pct" in cov.columns
        assert len(cov) == 2

    def test_global_mode_overlap(self, mock_encoder):
        odors = ["acetic acid", "ethanol", "benzaldehyde"]
        _, meta = similarity_matrix(odors, mock_encoder, method="cosine",
                                    mask_mode="global", min_overlap=1)
        assert meta["global_overlap_n"] is not None
        # Global overlap should be <= any individual coverage
        for _, row in meta["coverage"].iterrows():
            assert meta["global_overlap_n"] <= row["n_measured"]

    def test_global_overlap_le_pairwise(self, mock_encoder):
        """Global intersection overlap <= pairwise overlap for any pair."""
        odors = ["acetic acid", "ethanol", "benzaldehyde"]
        _, meta_pw = similarity_matrix(odors, mock_encoder, method="cosine",
                                       mask_mode="pairwise", min_overlap=1)
        _, meta_gl = similarity_matrix(odors, mock_encoder, method="cosine",
                                       mask_mode="global", min_overlap=1)
        n = len(odors)
        for i in range(n):
            for j in range(i + 1, n):
                assert (meta_gl["overlap_matrix"].iloc[i, j]
                        <= meta_pw["overlap_matrix"].iloc[i, j])


# ---------------------------------------------------------------------------
# find_similar
# ---------------------------------------------------------------------------


class TestFindSimilar:
    def test_returns_sorted_descending(self, mock_encoder):
        odors = ["acetic acid", "ethanol", "benzaldehyde", "linalool"]
        df = find_similar(odors, "acetic acid", mock_encoder, top_k=3,
                          method="cosine", min_overlap=1)
        assert "target_odor" in df.columns
        assert "similarity" in df.columns
        # Non-NaN values should be sorted descending
        sims = df["similarity"].dropna()
        assert list(sims) == sorted(sims, reverse=True)

    def test_query_excluded(self, mock_encoder):
        odors = ["acetic acid", "ethanol"]
        df = find_similar(odors, "acetic acid", mock_encoder, top_k=10,
                          method="cosine", min_overlap=1)
        assert "acetic acid" not in df["target_odor"].values

    def test_top_k_limits(self, mock_encoder):
        odors = ["acetic acid", "ethanol", "benzaldehyde", "linalool", "citral"]
        df = find_similar(odors, "acetic acid", mock_encoder, top_k=2,
                          method="cosine", min_overlap=1)
        assert len(df) == 2

    def test_has_coverage_columns(self, mock_encoder):
        odors = ["acetic acid", "ethanol"]
        df = find_similar(odors, "acetic acid", mock_encoder, top_k=10,
                          method="cosine", min_overlap=1)
        assert "target_coverage" in df.columns
        assert "query_coverage" in df.columns
        assert "overlap_n" in df.columns
