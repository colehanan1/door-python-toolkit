"""Odorant name to InChIKey mapping with case-insensitive lookup."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class OdorantMapper:
    """
    Maps between common odorant names and InChIKey identifiers.

    Provides case-insensitive lookup and fuzzy matching for odorant names.
    Uses InChIKey as the canonical identifier to eliminate ambiguity.
    """

    def __init__(self, mapping_file: Optional[str] = None) -> None:
        """
        Initialize odorant mapper with case-insensitive lookup.

        Args:
            mapping_file: Path to CSV mapping file. If None, uses default location.
        """
        if mapping_file is None:
            base_dir = Path(__file__).resolve().parents[3] / "data" / "mappings"
            candidates = [
                base_dir / "odorant_name_to_inchikey_complete.csv",
                base_dir / "odorant_name_to_inchikey_full.csv",
                base_dir / "odorant_name_to_inchikey.csv",
            ]
            for candidate in candidates:
                if candidate.exists():
                    mapping_file = candidate
                    break
            else:
                mapping_file = candidates[-1]

        self.mapping_file = Path(mapping_file)

        if not self.mapping_file.exists():
            raise FileNotFoundError(
                f"Odorant mapping file not found: {self.mapping_file}\n"
                "Please ensure data/mappings/odorant_name_to_inchikey.csv exists"
            )

        self.mapping_df = pd.read_csv(self.mapping_file)

        self.name_to_inchikey: Dict[str, str] = {}
        self.inchikey_to_name: Dict[str, str] = {}

        for _, row in self.mapping_df.iterrows():
            inchikey = str(row.get("inchikey", "")).strip()
            if not inchikey:
                continue

            common_name = str(row.get("common_name", "")).strip() or inchikey
            self.inchikey_to_name[inchikey] = common_name

            names = [common_name]
            alt_names = row.get("alternative_names")
            if isinstance(alt_names, str) and alt_names.strip():
                for alt in alt_names.replace(";", ",").split(","):
                    alt = alt.strip()
                    if alt:
                        names.append(alt)

            for name in names:
                for variant in self._normalize_variants(name):
                    self.name_to_inchikey.setdefault(variant, inchikey)

        logger.info("Loaded mapping for %d odorants", len(set(self.name_to_inchikey.values())))

    def is_inchikey(self, identifier: str) -> bool:
        """
        Check whether a string already looks like an InChIKey.
        """
        if not isinstance(identifier, str):
            return False
        ident = identifier.strip()
        return len(ident) == 27 and ident.count("-") >= 2

    def get_inchikey(self, odorant_name: str) -> Optional[str]:
        """
        Get InChIKey for an odorant name (CASE-INSENSITIVE).

        Args:
            odorant_name: Common name (any capitalization)

        Returns:
            InChIKey string or None if not found
        """
        if self.is_inchikey(odorant_name):
            return odorant_name

        for variant in self._normalize_variants(odorant_name):
            inchikey = self.name_to_inchikey.get(variant)
            if inchikey:
                logger.debug("Mapped '%s' → '%s' (variant '%s')", odorant_name, inchikey, variant)
                return inchikey
        return None

    @staticmethod
    def _normalize_variants(name: str) -> List[str]:
        """Generate normalized name variants for flexible matching."""
        normalized = name.lower().strip()
        variants = {normalized}

        if " " in normalized:
            variants.add(normalized.replace(" ", "-"))
            variants.add(normalized.replace(" ", ""))
        if "-" in normalized:
            variants.add(normalized.replace("-", " "))
            variants.add(normalized.replace("-", ""))

        variants.add(normalized.replace(" ", "").replace("-", ""))

        return [variant for variant in variants if variant]

    def get_common_name(self, inchikey: str) -> Optional[str]:
        """Get canonical common name for an InChIKey."""
        return self.inchikey_to_name.get(inchikey)

    def search_by_name(self, query: str, max_results: int = 5) -> List[Tuple[str, str]]:
        """
        Fuzzy search odorant names (case-insensitive).

        Args:
            query: Search string
            max_results: Maximum matches to return

        Returns:
            List of (common_name, inchikey) tuples
        """
        query_lower = query.lower()
        matches: List[Tuple[str, str]] = []
        seen: set[str] = set()

        for name_lower, inchikey in self.name_to_inchikey.items():
            if query_lower in name_lower and inchikey not in seen:
                canonical = self.inchikey_to_name.get(inchikey, name_lower)
                matches.append((canonical, inchikey))
                seen.add(inchikey)

        return matches[:max_results]

    def list_all_odorants(self) -> List[Tuple[str, str]]:
        """
        Get list of all available odorants.

        Returns:
            List of (common_name, inchikey) tuples, sorted by name.
        """
        odorants = [(name, inchikey) for inchikey, name in self.inchikey_to_name.items()]
        return sorted(odorants, key=lambda item: item[0].lower())
