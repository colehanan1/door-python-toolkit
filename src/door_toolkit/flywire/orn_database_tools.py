"""
ORN-FlyWire Database Lookup Tools
==================================

Instant lookup functions for the complete ORN-FlyWire connectivity database.

Once the database is built, these functions provide zero-latency queries
without needing to re-run FlyWire connectome analysis.

Example:
    >>> from door_toolkit.flywire.orn_database_tools import get_orn_mapping
    >>>
    >>> # Instant lookup!
    >>> data = get_orn_mapping("Or49a")
    >>> print(f"Circuit score: {data['circuit_score']}")
    >>> print(f"KC coverage: {data['kc_coverage']:.2%}")
    >>> print(f"Circuit type: {data['circuit_type']}")
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


class ORNDatabase:
    """
    Lookup interface for complete ORN-FlyWire connectivity database.

    This class provides fast, zero-latency queries to the pre-computed
    database of all 78 DoOR receptors mapped to FlyWire connectome.

    Attributes:
        db_df: DataFrame with all receptor data
        db_dict: Dict mapping receptor names to data
    """

    def __init__(self, database_path: Optional[str] = None):
        """
        Initialize database.

        Args:
            database_path: Path to flywire_orn_complete_v1.0.csv
                          If None, searches for database in package data
        """
        if database_path is None:
            # Try to find database in package
            package_dir = Path(__file__).parent.parent.parent.parent
            db_candidates = [
                package_dir / "flywire_orn_database" / "flywire_orn_complete_v1.0.csv",
                package_dir / "data" / "flywire_orn_complete_v1.0.csv",
            ]

            database_path = None
            for candidate in db_candidates:
                if candidate.exists():
                    database_path = str(candidate)
                    break

            if database_path is None:
                raise FileNotFoundError(
                    "ORN database not found. Please build it first using:\n"
                    "python examples/advanced/build_complete_orn_database.py"
                )

        self.db_path = Path(database_path)
        self.db_df = pd.read_csv(self.db_path)

        # Create dict for fast lookups
        self.db_dict = {
            row["receptor"]: row.to_dict()
            for _, row in self.db_df.iterrows()
        }

        # Filter to successfully mapped receptors
        self.success_df = self.db_df[self.db_df["status"] == "success"]

    def get(self, receptor: str) -> Optional[Dict]:
        """
        Get complete data for a receptor.

        Args:
            receptor: Receptor name (e.g., "Or49a", "Ir75a")

        Returns:
            Dict with all connectivity data, or None if not found

        Example:
            >>> db = ORNDatabase()
            >>> data = db.get("Or49a")
            >>> print(data["circuit_score"])
        """
        return self.db_dict.get(receptor)

    def get_circuit_score(self, receptor: str) -> Optional[float]:
        """Get circuit score for a receptor (0-1)."""
        data = self.get(receptor)
        return data["circuit_score"] if data and data["status"] == "success" else None

    def get_circuit_type(self, receptor: str) -> Optional[str]:
        """Get circuit type: 'appetitive' or 'aversive'."""
        data = self.get(receptor)
        return data["circuit_type"] if data and data["status"] == "success" else None

    def get_kc_coverage(self, receptor: str) -> Optional[float]:
        """Get KC coverage (0-1 fraction of all KCs)."""
        data = self.get(receptor)
        return data["kc_coverage"] if data and data["status"] == "success" else None

    def get_orn_to_pn_strength(self, receptor: str) -> Optional[float]:
        """Get ORN→PN synaptic strength (0-1)."""
        data = self.get(receptor)
        return data["orn_to_pn_strength"] if data and data["status"] == "success" else None

    def get_kc_compartments(self, receptor: str) -> Optional[Dict]:
        """Get KC compartment distribution (α/β, γ, α'β')."""
        data = self.get(receptor)
        if not data or data["status"] != "success":
            return None

        return {
            "alpha_beta": data.get("kc_alpha_beta", 0),
            "gamma": data.get("kc_gamma", 0),
            "alpha_prime_beta_prime": data.get("kc_alpha_prime_beta_prime", 0),
        }

    def list_all_receptors(self) -> List[str]:
        """Get list of all receptors in database."""
        return sorted(self.db_df["receptor"].tolist())

    def list_successful_receptors(self) -> List[str]:
        """Get list of successfully mapped receptors."""
        return sorted(self.success_df["receptor"].tolist())

    def filter_by_circuit_type(self, circuit_type: str) -> pd.DataFrame:
        """
        Filter receptors by circuit type.

        Args:
            circuit_type: "appetitive" or "aversive"

        Returns:
            DataFrame with matching receptors
        """
        return self.success_df[self.success_df["circuit_type"] == circuit_type]

    def filter_by_circuit_score(
        self,
        min_score: float = 0.0,
        max_score: float = 1.0
    ) -> pd.DataFrame:
        """
        Filter receptors by circuit score range.

        Args:
            min_score: Minimum circuit score (0-1)
            max_score: Maximum circuit score (0-1)

        Returns:
            DataFrame with receptors in score range
        """
        return self.success_df[
            (self.success_df["circuit_score"] >= min_score) &
            (self.success_df["circuit_score"] <= max_score)
        ]

    def rank_by_metric(
        self,
        metric: str,
        top_n: Optional[int] = None,
        ascending: bool = False
    ) -> pd.DataFrame:
        """
        Rank receptors by a specific metric.

        Args:
            metric: Column name (e.g., 'circuit_score', 'kc_coverage')
            top_n: Return only top N receptors (None = all)
            ascending: Sort ascending (default: descending)

        Returns:
            Ranked DataFrame

        Example:
            >>> db = ORNDatabase()
            >>> top10 = db.rank_by_metric('kc_coverage', top_n=10)
            >>> print(top10[['receptor', 'kc_coverage']])
        """
        ranked = self.success_df.sort_values(metric, ascending=ascending)

        if top_n:
            ranked = ranked.head(top_n)

        return ranked

    def compare_receptors(self, receptors: List[str]) -> pd.DataFrame:
        """
        Compare multiple receptors side-by-side.

        Args:
            receptors: List of receptor names

        Returns:
            DataFrame with key metrics for comparison

        Example:
            >>> db = ORNDatabase()
            >>> comparison = db.compare_receptors(["Or49a", "Or42a", "Or46a"])
            >>> print(comparison)
        """
        data = [self.get(r) for r in receptors if self.get(r)]

        if not data:
            return pd.DataFrame()

        compare_df = pd.DataFrame(data)

        # Select key columns
        key_columns = [
            "receptor",
            "n_orns",
            "n_pns",
            "n_kcs",
            "orn_to_pn_strength",
            "kc_coverage",
            "alpha_beta_fraction",
            "gamma_fraction",
            "circuit_score",
            "circuit_type",
        ]

        return compare_df[[c for c in key_columns if c in compare_df.columns]]

    def get_statistics(self) -> Dict:
        """Get overall database statistics."""
        stats = {
            "total_receptors": len(self.db_df),
            "successfully_mapped": len(self.success_df),
            "mapping_success_rate": len(self.success_df) / len(self.db_df),
            "appetitive_count": sum(self.success_df["circuit_type"] == "appetitive"),
            "aversive_count": sum(self.success_df["circuit_type"] == "aversive"),
            "mean_circuit_score": self.success_df["circuit_score"].mean(),
            "mean_kc_coverage": self.success_df["kc_coverage"].mean(),
            "mean_orns_per_receptor": self.success_df["n_orns"].mean(),
            "mean_kcs_per_receptor": self.success_df["n_kcs"].mean(),
        }

        return stats

    def print_summary(self, receptor: str):
        """Print formatted summary for a receptor."""
        data = self.get(receptor)

        if not data:
            print(f"Receptor '{receptor}' not found in database.")
            return

        if data["status"] != "success":
            print(f"Receptor '{receptor}' - Status: {data['status']}")
            if "error" in data:
                print(f"Error: {data['error']}")
            return

        print(f"\n{'=' * 60}")
        print(f"{receptor} - FlyWire Mushroom Body Connectivity")
        print(f"{'=' * 60}")
        print(f"\nCircuit Classification:")
        print(f"  Type: {data['circuit_type'].upper()}")
        print(f"  Circuit Score: {data['circuit_score']:.3f}")
        print(f"\nNeuron Counts:")
        print(f"  ORNs: {data['n_orns']}")
        print(f"  PNs contacted: {data['n_pns']}")
        print(f"  KCs contacted: {data['n_kcs']}")
        print(f"  MBONs contacted: {data['n_mbons']}")
        print(f"\nConnectivity Metrics:")
        print(f"  ORN→PN Strength: {data['orn_to_pn_strength']:.2%}")
        print(f"  KC Coverage: {data['kc_coverage']:.2%}")
        print(f"\nKC Compartment Distribution:")
        print(f"  α/β lobe (appetitive): {data.get('kc_alpha_beta', 0)} "
              f"({data.get('alpha_beta_fraction', 0):.1%})")
        print(f"  γ lobe (aversive): {data.get('kc_gamma', 0)} "
              f"({data.get('gamma_fraction', 0):.1%})")
        print(f"  α'β' lobe: {data.get('kc_alpha_prime_beta_prime', 0)}")
        print(f"\n{'=' * 60}\n")


# Convenience functions for quick access

_global_db = None


def _get_database() -> ORNDatabase:
    """Get or initialize global database instance."""
    global _global_db
    if _global_db is None:
        _global_db = ORNDatabase()
    return _global_db


def get_orn_mapping(receptor: str) -> Optional[Dict]:
    """
    Quick lookup: Get complete FlyWire data for a receptor.

    Args:
        receptor: Receptor name (e.g., "Or49a")

    Returns:
        Dict with all connectivity data

    Example:
        >>> data = get_orn_mapping("Or49a")
        >>> print(f"Circuit score: {data['circuit_score']}")
    """
    return _get_database().get(receptor)


def get_circuit_score(receptor: str) -> Optional[float]:
    """Quick lookup: Get circuit score (0-1)."""
    return _get_database().get_circuit_score(receptor)


def get_circuit_type(receptor: str) -> Optional[str]:
    """Quick lookup: Get circuit type ('appetitive' or 'aversive')."""
    return _get_database().get_circuit_type(receptor)


def compare_orns(receptors: List[str]) -> pd.DataFrame:
    """
    Quick comparison of multiple receptors.

    Example:
        >>> comparison = compare_orns(["Or49a", "Or42a", "Or46a"])
        >>> print(comparison)
    """
    return _get_database().compare_receptors(receptors)


def rank_orns_by_metric(metric: str, top_n: int = 10) -> pd.DataFrame:
    """
    Quick ranking by any metric.

    Example:
        >>> top_kc = rank_orns_by_metric('kc_coverage', top_n=10)
        >>> print(top_kc[['receptor', 'kc_coverage']])
    """
    return _get_database().rank_by_metric(metric, top_n=top_n)


def print_orn_summary(receptor: str):
    """Print formatted summary for a receptor."""
    _get_database().print_summary(receptor)
