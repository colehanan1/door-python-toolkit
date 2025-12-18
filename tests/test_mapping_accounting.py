"""
Tests for Receptor-to-Glomerulus Mapping Accounting
====================================================

Comprehensive test suite for mapping_accounting module.

Test scenarios:
1. Many-to-one collapse (e.g., OR82A + OR94A → VA6)
2. Mixed mapped/unmapped receptors
3. Adult/larval exclusion filtering
4. Deterministic ordering and stable JSON output
"""

import json
import pytest
from pathlib import Path
import tempfile

from door_toolkit.integration.mapping_accounting import (
    build_glomerulus_to_receptors,
    summarize_collisions,
    compute_mapping_stats,
    is_larval_receptor,
    write_mapping_stats_json,
    log_mapping_stats,
    format_mapping_summary,
    LARVAL_ONLY_RECEPTORS,
)


class TestBuildGlomerulusToReceptors:
    """Tests for reverse mapping construction."""

    def test_one_to_one_mapping(self):
        """Simple 1:1 receptor→glomerulus mapping."""
        mapping = {
            'OR7A': 'DL5',
            'OR42B': 'DM1',
            'IR31A': 'VL2p'
        }

        result = build_glomerulus_to_receptors(mapping)

        assert result == {
            'DL5': ['OR7A'],
            'DM1': ['OR42B'],
            'VL2p': ['IR31A']
        }

    def test_many_to_one_mapping(self):
        """Multiple receptors mapping to same glomerulus (VA6 case)."""
        mapping = {
            'OR82A': 'VA6',
            'OR94A': 'VA6',
            'OR7A': 'DL5'
        }

        result = build_glomerulus_to_receptors(mapping)

        assert result == {
            'VA6': ['OR82A', 'OR94A'],
            'DL5': ['OR7A']
        }
        assert len(result['VA6']) == 2

    def test_deterministic_ordering(self):
        """Receptor lists should be sorted for stability."""
        mapping = {
            'OR94A': 'VA6',  # Unsorted input
            'OR82A': 'VA6',
            'OR7A': 'DL5'
        }

        result = build_glomerulus_to_receptors(mapping)

        # Receptors within glomerulus should be sorted
        assert result['VA6'] == ['OR82A', 'OR94A']  # Alphabetical

    def test_empty_mapping(self):
        """Empty mapping returns empty dict."""
        result = build_glomerulus_to_receptors({})
        assert result == {}


class TestSummarizeCollisions:
    """Tests for collision detection."""

    def test_no_collisions(self):
        """1:1 mapping has no collisions."""
        glom_map = {
            'DL5': ['OR7A'],
            'DM1': ['OR42B']
        }

        result = summarize_collisions(glom_map)

        assert result == {}

    def test_single_collision(self):
        """One glomerulus with multiple receptors."""
        glom_map = {
            'VA6': ['OR82A', 'OR94A'],
            'DL5': ['OR7A']
        }

        result = summarize_collisions(glom_map)

        assert result == {'VA6': ['OR82A', 'OR94A']}

    def test_multiple_collisions(self):
        """Multiple glomeruli with collisions."""
        glom_map = {
            'VA6': ['OR82A', 'OR94A'],
            'DL5': ['OR7A'],
            'DM1': ['OR42B', 'OR43B', 'OR44B']
        }

        result = summarize_collisions(glom_map)

        assert len(result) == 2
        assert 'VA6' in result
        assert 'DM1' in result
        assert len(result['DM1']) == 3

    def test_min_size_parameter(self):
        """Filter collisions by minimum size."""
        glom_map = {
            'VA6': ['OR82A', 'OR94A'],           # 2 receptors
            'DM1': ['OR42B', 'OR43B', 'OR44B']   # 3 receptors
        }

        # Only collisions with ≥3 receptors
        result = summarize_collisions(glom_map, min_size=3)

        assert result == {'DM1': ['OR42B', 'OR43B', 'OR44B']}
        assert 'VA6' not in result


class TestComputeMappingStats:
    """Tests for comprehensive mapping statistics."""

    def test_many_to_one_collapse_case(self):
        """
        Critical test case: OR82A + OR94A → VA6 (many-to-one).

        Expected:
        - n_receptors_total = 2
        - n_unique_glomeruli_from_mapped_receptors = 1 (only VA6)
        - collision_count = 1 (VA6 has 2 receptors)
        """
        mapping = {
            'OR82A': 'VA6',
            'OR94A': 'VA6'
        }

        stats = compute_mapping_stats(mapping, note="VA6 test case", adult_only=False)

        # CRITICAL: Receptor count != unique glomerulus count!
        assert stats['n_receptors_total'] == 2
        assert stats['n_receptors_mapped'] == 2
        assert stats['n_unique_glomeruli_from_mapped_receptors'] == 1  # Only VA6

        # Collision detection
        assert stats['collision_count'] == 1
        assert 'VA6' in stats['collisions']
        assert stats['collisions']['VA6'] == ['OR82A', 'OR94A']
        assert 'VA6: OR82A, OR94A' in stats['collision_summary']

    def test_mixed_mapped_unmapped(self):
        """Mixed mapped and unmapped receptors."""
        mapping = {
            'OR7A': 'DL5',
            'OR42B': 'DM1'
        }

        unmapped = ['OR99Z', 'OR100A']  # Failed to map

        stats = compute_mapping_stats(
            mapping,
            unmapped_receptors=unmapped,
            note="Mixed test",
            adult_only=False
        )

        assert stats['n_receptors_mapped'] == 2
        assert stats['n_receptors_unmapped'] == 2
        assert stats['n_receptors_total'] == 4  # 2 mapped + 2 unmapped
        assert stats['receptors_unmapped'] == ['OR100A', 'OR99Z']  # Sorted

    def test_exclusion_with_input_receptors(self):
        """Explicit input_receptors list with exclusions."""
        mapping = {
            'OR7A': 'DL5',
            'OR42B': 'DM1'
        }

        input_recs = ['OR7A', 'OR42B', 'OR99Z', 'OR100A']
        excluded = ['OR99Z']
        unmapped = ['OR100A']

        stats = compute_mapping_stats(
            mapping,
            input_receptors=input_recs,
            excluded_receptors=excluded,
            unmapped_receptors=unmapped,
            adult_only=False
        )

        assert stats['n_receptors_total'] == 4
        assert stats['n_receptors_mapped'] == 2
        assert stats['n_receptors_unmapped'] == 1
        assert stats['n_receptors_excluded_other'] == 1
        assert stats['receptors_excluded_other'] == ['OR99Z']

    def test_adult_only_filtering(self):
        """Adult-only mode excludes larval receptors."""
        mapping = {
            'OR7A': 'DL5',          # Adult
            'OR1A': 'DA2',          # Larval (in LARVAL_ONLY_RECEPTORS)
            'OR42B': 'DM1'          # Adult
        }

        stats = compute_mapping_stats(mapping, adult_only=True)

        # OR1A should be filtered out
        assert stats['n_receptors_mapped'] == 2  # Only OR7A, OR42B
        assert stats['n_receptors_excluded_larval'] == 1  # OR1A
        assert stats['receptors_excluded_larval'] == ['OR1A']
        assert stats['n_unique_glomeruli_from_mapped_receptors'] == 2  # DL5, DM1 (not DA2)

    def test_adult_only_disabled(self):
        """Adult-only mode disabled includes all receptors."""
        mapping = {
            'OR7A': 'DL5',
            'OR1A': 'DA2',  # Larval
            'OR42B': 'DM1'
        }

        stats = compute_mapping_stats(mapping, adult_only=False)

        # All receptors should be included
        assert stats['n_receptors_mapped'] == 3
        assert stats['n_receptors_excluded_larval'] == 0
        assert stats['n_unique_glomeruli_from_mapped_receptors'] == 3

    def test_custom_larval_list(self):
        """Custom larval list overrides default."""
        mapping = {
            'OR7A': 'DL5',
            'OR42B': 'DM1',
            'CUSTOM_LARVAL': 'VA6'
        }

        custom_larval = {'CUSTOM_LARVAL'}

        stats = compute_mapping_stats(
            mapping,
            adult_only=True,
            custom_larval_list=custom_larval
        )

        assert stats['n_receptors_excluded_larval'] == 1
        assert stats['receptors_excluded_larval'] == ['CUSTOM_LARVAL']

    def test_deterministic_ordering(self):
        """All lists should be sorted for reproducibility."""
        mapping = {
            'OR7A': 'DL5',
            'OR42B': 'DM1',
            'OR33B': 'DA2'
        }

        stats = compute_mapping_stats(mapping, adult_only=False)

        # Receptor lists should be sorted
        assert stats['receptors_mapped'] == sorted(stats['receptors_mapped'])

    def test_collision_summary_format(self):
        """Collision summary should be human-readable."""
        mapping = {
            'OR82A': 'VA6',
            'OR94A': 'VA6',
            'OR7A': 'DL5'
        }

        stats = compute_mapping_stats(mapping, adult_only=False)

        # Should have readable summary
        assert len(stats['collision_summary']) == 1
        assert stats['collision_summary'][0] == 'VA6: OR82A, OR94A'

    def test_empty_mapping(self):
        """Empty mapping returns zero counts."""
        stats = compute_mapping_stats({}, adult_only=False)

        assert stats['n_receptors_total'] == 0
        assert stats['n_receptors_mapped'] == 0
        assert stats['n_unique_glomeruli_from_mapped_receptors'] == 0
        assert stats['collision_count'] == 0


class TestIsLarvalReceptor:
    """Tests for larval receptor detection."""

    def test_known_larval_receptor(self):
        """Receptors in LARVAL_ONLY_RECEPTORS list."""
        assert is_larval_receptor('OR1A') is True
        assert is_larval_receptor('OR45A') is True
        assert is_larval_receptor('OR94A') is True
        assert is_larval_receptor('OR59A') is True

    def test_adult_receptor(self):
        """Adult receptors not in larval list."""
        assert is_larval_receptor('OR7A') is False
        assert is_larval_receptor('OR42B') is False
        assert is_larval_receptor('IR31A') is False

    def test_case_insensitive(self):
        """Case-insensitive matching."""
        assert is_larval_receptor('or45a') is True
        assert is_larval_receptor('Or45A') is True
        assert is_larval_receptor('OR45a') is True

    def test_custom_larval_list(self):
        """Custom list overrides default."""
        custom = {'CUSTOM_LARVAL'}

        assert is_larval_receptor('CUSTOM_LARVAL', custom_larval_list=custom) is True
        assert is_larval_receptor('OR45A', custom_larval_list=custom) is False  # Not in custom


class TestWriteMappingStatsJson:
    """Tests for JSON persistence."""

    def test_writes_valid_json(self):
        """Write stats to JSON and verify format."""
        mapping = {
            'OR82A': 'VA6',
            'OR94A': 'VA6',
            'OR7A': 'DL5'
        }

        stats = compute_mapping_stats(mapping, note="Test", adult_only=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "mapping_stats.json"

            write_mapping_stats_json(output_path, stats)

            # Verify file exists and is valid JSON
            assert output_path.exists()

            with open(output_path, 'r') as f:
                loaded = json.load(f)

            # Verify key fields
            assert loaded['n_receptors_total'] == 3
            assert loaded['n_unique_glomeruli_from_mapped_receptors'] == 2
            assert loaded['note'] == "Test"

    def test_deterministic_json_output(self):
        """JSON output should be deterministic (sorted keys)."""
        mapping = {'OR7A': 'DL5', 'OR42B': 'DM1'}
        stats = compute_mapping_stats(mapping, adult_only=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "stats1.json"

            write_mapping_stats_json(output_path, stats)

            with open(output_path, 'r') as f:
                content1 = f.read()

            # Write again
            output_path2 = Path(tmpdir) / "stats2.json"
            write_mapping_stats_json(output_path2, stats)

            with open(output_path2, 'r') as f:
                content2 = f.read()

            # Content should be identical (sorted keys)
            assert content1 == content2


class TestFormatMappingSummary:
    """Tests for compact summary formatting."""

    def test_no_collisions(self):
        """Summary for 1:1 mapping."""
        mapping = {'OR7A': 'DL5', 'OR42B': 'DM1'}
        stats = compute_mapping_stats(mapping, adult_only=False)

        summary = format_mapping_summary(stats)

        assert summary == "2 receptors → 2 unique glomeruli (no collisions)"

    def test_with_collisions(self):
        """Summary with many-to-one collapse."""
        mapping = {'OR82A': 'VA6', 'OR94A': 'VA6', 'OR7A': 'DL5'}
        stats = compute_mapping_stats(mapping, adult_only=False)

        summary = format_mapping_summary(stats)

        assert summary == "3 receptors → 2 unique glomeruli (1 collision)"

    def test_multiple_collisions(self):
        """Summary with multiple collisions."""
        mapping = {
            'OR82A': 'VA6',
            'OR94A': 'VA6',
            'OR42B': 'DM1',
            'OR43B': 'DM1'
        }
        stats = compute_mapping_stats(mapping, adult_only=False)

        summary = format_mapping_summary(stats)

        assert summary == "4 receptors → 2 unique glomeruli (2 collisions)"
