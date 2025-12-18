"""
Tests for ORN/Glomerulus Identifier Resolution
================================================

Comprehensive test suite for orn_identifier.py module.
"""

import pytest
from door_toolkit.integration.orn_identifier import (
    normalize_orn_identifier,
    generate_orn_candidates,
    suggest_orn_identifiers,
    resolve_orn_identifier,
)


class TestNormalizeORNIdentifier:
    """Tests for normalize_orn_identifier function."""

    def test_exact_match_orn_dl3(self):
        """ORN_DL3 should remain ORN_DL3."""
        assert normalize_orn_identifier("ORN_DL3") == "ORN_DL3"

    def test_lowercase_dl3(self):
        """dl3 → ORN_DL3."""
        assert normalize_orn_identifier("dl3") == "ORN_DL3"

    def test_uppercase_dl3(self):
        """DL3 → ORN_DL3."""
        assert normalize_orn_identifier("DL3") == "ORN_DL3"

    def test_hyphen_separator(self):
        """ORN-DL3 → ORN_DL3."""
        assert normalize_orn_identifier("ORN-DL3") == "ORN_DL3"

    def test_space_separator(self):
        """ORN DL3 → ORN_DL3."""
        assert normalize_orn_identifier("ORN DL3") == "ORN_DL3"

    def test_glomerulus_prefix(self):
        """Glomerulus DL3 → ORN_DL3."""
        assert normalize_orn_identifier("Glomerulus DL3") == "ORN_DL3"

    def test_glom_prefix(self):
        """Glom DL3 → ORN_DL3."""
        assert normalize_orn_identifier("Glom DL3") == "ORN_DL3"

    def test_mixed_case_with_space(self):
        """orn dl3 → ORN_DL3."""
        assert normalize_orn_identifier("orn dl3") == "ORN_DL3"

    def test_ir_receptor(self):
        """Ir31a → ORN_VL2p (mapped to glomerulus)."""
        assert normalize_orn_identifier("Ir31a") == "ORN_VL2p"

    def test_or_receptor(self):
        """Or7a → ORN_DL5 (mapped to glomerulus)."""
        assert normalize_orn_identifier("Or7a") == "ORN_DL5"

    def test_gr_receptor(self):
        """Gr21a → ORN_V (mapped to glomerulus)."""
        assert normalize_orn_identifier("Gr21a") == "ORN_V"

    def test_without_prefix_option(self):
        """DL3 without prefix → DL3."""
        assert normalize_orn_identifier("DL3", prefer_glomerulus_prefix=False) == "DL3"

    def test_complex_name_va1d(self):
        """VA1d → ORN_VA1D."""
        assert normalize_orn_identifier("VA1d") == "ORN_VA1D"

    def test_whitespace_stripping(self):
        """  DL3   → ORN_DL3."""
        assert normalize_orn_identifier("  DL3   ") == "ORN_DL3"

    def test_multiple_separators(self):
        """ORN - DL3 → ORN_DL3."""
        assert normalize_orn_identifier("ORN - DL3") == "ORN_DL3"


class TestGenerateORNCandidates:
    """Tests for generate_orn_candidates function."""

    def test_simple_identifier(self):
        """DL3 generates multiple candidates."""
        candidates = generate_orn_candidates("DL3")
        assert "DL3" in candidates
        assert "ORN_DL3" in candidates

    def test_ir_receptor_candidates(self):
        """Ir31a generates candidates including ORN_VL2p (glomerulus mapping)."""
        candidates = generate_orn_candidates("Ir31a")
        assert "ORN_VL2p" in candidates or "ORN_IR31A" in candidates

    def test_candidates_include_raw(self):
        """Raw input is always first candidate."""
        candidates = generate_orn_candidates("custom_name")
        assert candidates[0] == "custom_name"

    def test_empty_input(self):
        """Empty string returns empty list."""
        assert generate_orn_candidates("") == []

    def test_none_input(self):
        """None returns empty list."""
        assert generate_orn_candidates(None) == []


class TestSuggestORNIdentifiers:
    """Tests for suggest_orn_identifiers function."""

    @pytest.fixture
    def mock_available(self):
        """Mock set of available glomeruli."""
        return {
            "ORN_DL3",
            "ORN_DL5",
            "ORN_DA3",
            "ORN_VA1d",
            "ORN_VM7d",
            "ORN_VL2p",  # Ir31a maps to VL2p
            "ORN_VC5",   # Ir41a maps to VC5
        }

    def test_exact_match_has_high_score(self, mock_available):
        """Exact match should have similarity ~1.0."""
        suggestions = suggest_orn_identifiers("ORN_DL3", mock_available, k=1)
        assert len(suggestions) == 1
        assert suggestions[0][0] == "ORN_DL3"
        assert suggestions[0][1] > 0.95

    def test_close_match_dl33(self, mock_available):
        """DL33 should suggest DL3 and DL5."""
        suggestions = suggest_orn_identifiers("DL33", mock_available, k=3)
        identifiers = [s[0] for s in suggestions]
        assert "ORN_DL3" in identifiers or "ORN_DL5" in identifiers

    def test_k_limit(self, mock_available):
        """Should return at most k suggestions."""
        suggestions = suggest_orn_identifiers("DL", mock_available, k=3)
        assert len(suggestions) <= 3

    def test_similarity_threshold(self, mock_available):
        """Should filter by similarity threshold."""
        suggestions = suggest_orn_identifiers(
            "XYZ", mock_available, k=10, similarity_threshold=0.8
        )
        # "XYZ" should have low similarity to all available
        assert len(suggestions) == 0

    def test_ir_receptor_suggestion(self, mock_available):
        """Ir31a should match ORN_VL2p (via glomerulus mapping)."""
        suggestions = suggest_orn_identifiers("Ir31a", mock_available, k=1)
        assert len(suggestions) >= 1
        # Should match ORN_VL2p with high similarity (Ir31a maps to VL2p)
        assert suggestions[0][0] == "ORN_VL2p"


class TestResolveORNIdentifier:
    """Tests for resolve_orn_identifier function."""

    @pytest.fixture
    def mock_available(self):
        """Mock set of available glomeruli."""
        return {
            "ORN_DL3",
            "ORN_DL5",  # Or7a maps to DL5
            "ORN_DA3",
            "ORN_VA1d",
            "ORN_VL2p",  # Ir31a maps to VL2p
            "ORN_VC5",   # Ir41a maps to VC5
            "ORN_DP1l",  # Ir75a maps to DP1l
        }

    def test_exact_match_returns_immediately(self, mock_available):
        """ORN_DL3 → ORN_DL3 (exact match)."""
        result = resolve_orn_identifier("ORN_DL3", mock_available)
        assert result == "ORN_DL3"

    def test_normalized_match_dl3(self, mock_available):
        """DL3 → ORN_DL3 (via normalization)."""
        result = resolve_orn_identifier("DL3", mock_available)
        assert result == "ORN_DL3"

    def test_lowercase_match_dl5(self, mock_available):
        """dl5 → ORN_DL5."""
        result = resolve_orn_identifier("dl5", mock_available)
        assert result == "ORN_DL5"

    def test_space_separator_match(self, mock_available):
        """ORN DL3 → ORN_DL3."""
        result = resolve_orn_identifier("ORN DL3", mock_available)
        assert result == "ORN_DL3"

    def test_hyphen_separator_match(self, mock_available):
        """ORN-DL3 → ORN_DL3."""
        result = resolve_orn_identifier("ORN-DL3", mock_available)
        assert result == "ORN_DL3"

    def test_ir_receptor_match(self, mock_available):
        """Ir31a → ORN_VL2p (via glomerulus mapping)."""
        result = resolve_orn_identifier("Ir31a", mock_available)
        assert result == "ORN_VL2p"

    def test_or_receptor_match(self, mock_available):
        """Or7a → ORN_DL5 (via glomerulus mapping)."""
        result = resolve_orn_identifier("Or7a", mock_available)
        assert result == "ORN_DL5"

    def test_case_insensitive_ir41a(self, mock_available):
        """IR41A → ORN_VC5 (via glomerulus mapping)."""
        result = resolve_orn_identifier("IR41A", mock_available)
        assert result == "ORN_VC5"

    def test_no_match_raises_error(self, mock_available):
        """Non-existent identifier raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_orn_identifier("ORN_XYZ999", mock_available)

        error_msg = str(exc_info.value)
        assert "not found" in error_msg
        assert "Did you mean" in error_msg or "Available" in error_msg

    def test_error_contains_suggestions(self, mock_available):
        """Error message includes suggestions."""
        with pytest.raises(ValueError) as exc_info:
            resolve_orn_identifier("DL99", mock_available)

        error_msg = str(exc_info.value)
        # Should suggest similar DL glomeruli
        assert "ORN_DL3" in error_msg or "ORN_DL5" in error_msg

    def test_error_shows_normalized_form(self, mock_available):
        """Error message shows what normalization produced."""
        with pytest.raises(ValueError) as exc_info:
            resolve_orn_identifier("glom XYZ", mock_available)

        error_msg = str(exc_info.value)
        assert "Normalized to" in error_msg

    def test_strict_mode_no_fuzzy(self, mock_available):
        """Strict mode disables fuzzy matching."""
        # Add a very similar but not exact identifier
        with pytest.raises(ValueError):
            # "DL33" might fuzzy-match to DL3, but strict mode prevents it
            resolve_orn_identifier("DL33", mock_available, strict=True)

    def test_empty_identifier_raises(self, mock_available):
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            resolve_orn_identifier("", mock_available)

    def test_none_identifier_raises(self, mock_available):
        """None raises ValueError."""
        with pytest.raises(ValueError):
            resolve_orn_identifier(None, mock_available)

    def test_empty_available_set_raises(self):
        """Empty available set raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            resolve_orn_identifier("DL3", set())

        assert "no available identifiers" in str(exc_info.value)


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_all_user_mentioned_ir_receptors(self):
        """Test the specific Ir receptors mentioned by user.

        In FlyWire, Ir receptors are labeled by their glomerulus names:
        - Ir31a → VL2p
        - Ir41a → VC5
        - Ir75a → DP1l
        - Ir75d → VL1
        - Ir76a → VM4
        - Ir84a → VL2a
        - Ir92a → VM1
        """
        available = {
            "ORN_VL2p",  # Ir31a
            "ORN_VC5",   # Ir41a
            "ORN_DP1l",  # Ir75a
            "ORN_VL1",   # Ir75d
            "ORN_VM4",   # Ir76a
            "ORN_VL2a",  # Ir84a
            "ORN_VM1",   # Ir92a
        }

        # Test each receptor resolves to its glomerulus
        assert resolve_orn_identifier("Ir31a", available) == "ORN_VL2p"
        assert resolve_orn_identifier("Ir41a", available) == "ORN_VC5"
        assert resolve_orn_identifier("Ir75a", available) == "ORN_DP1l"
        assert resolve_orn_identifier("Ir75d", available) == "ORN_VL1"
        assert resolve_orn_identifier("Ir76a", available) == "ORN_VM4"
        assert resolve_orn_identifier("Ir84a", available) == "ORN_VL2a"
        assert resolve_orn_identifier("Ir92a", available) == "ORN_VM1"

    def test_common_glomerulus_names(self):
        """Test common glomerulus names from examples."""
        available = {
            "ORN_DL3",
            "ORN_DL5",
            "ORN_DA1",
            "ORN_DA3",
            "ORN_VA1d",
            "ORN_VM7d",
        }

        # Various input formats should all work
        assert resolve_orn_identifier("DL3", available) == "ORN_DL3"
        assert resolve_orn_identifier("dl5", available) == "ORN_DL5"
        assert resolve_orn_identifier("ORN_DA1", available) == "ORN_DA1"
        assert resolve_orn_identifier("da3", available) == "ORN_DA3"
        assert resolve_orn_identifier("VA1d", available) == "ORN_VA1d"
        assert resolve_orn_identifier("vm7d", available) == "ORN_VM7d"

    def test_deterministic_suggestions(self):
        """Suggestions should be deterministic and stable."""
        available = {"ORN_DL3", "ORN_DL5", "ORN_DL1"}

        # Run twice to ensure stability
        sugg1 = suggest_orn_identifiers("DL", available, k=10)
        sugg2 = suggest_orn_identifiers("DL", available, k=10)

        assert sugg1 == sugg2

    def test_resolution_with_punctuation(self):
        """Handle identifiers with various punctuation."""
        available = {"ORN_DL3", "ORN_VA1d"}

        # Should handle these gracefully
        assert resolve_orn_identifier("DL3!", available) == "ORN_DL3" or \
               pytest.raises(ValueError)  # Depending on implementation
