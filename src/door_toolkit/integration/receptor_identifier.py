"""
Receptor Identifier Normalization
================================

Provides a single, canonical normalization function for DoOR receptor identifiers
so the same receptor can be matched across:
- DoOR-loaded receptor names (from the cached response matrix)
- Mapping CSV keys (e.g., data/mappings/door_to_flywire_mapping.csv)
- Inventory generation and downstream analyses

Normalization goal:
- If a mapping exists but capitalization differs, it should still match.
- Handle DoOR naming quirks such as dotted suffixes (e.g., Ir64a.DC4) and paired
  receptors (e.g., Gr21a.Gr63a).

Scientific context:
- DoOR 2.0 receptor set and nomenclature: Münch & Galizia (2016) Scientific Data.
"""

from __future__ import annotations

import re
from typing import Optional

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_receptor_identifier(raw: Optional[str]) -> str:
    """
    Normalize a receptor identifier into a canonical *matching key*.

    This function is intentionally conservative: it preserves DoOR punctuation
    (e.g., "." in Gr21a.Gr63a and Ir64a.DP1m) while making comparisons robust
    to capitalization and incidental whitespace.

    Args:
        raw: Raw receptor identifier (e.g., "Or7a", "OR7A", "Ir64a.DP1m")

    Returns:
        A normalized key suitable for joins/lookup (e.g., "OR7A", "IR64A.DP1M").
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)

    value = raw.strip().strip('"').strip("'").strip()
    if not value:
        return ""

    # Remove incidental whitespace (e.g., "Or 7a" -> "Or7a")
    value = _WHITESPACE_RE.sub("", value)

    # Users occasionally paste glomerulus-prefixed strings; strip if present.
    if value.upper().startswith("ORN_"):
        value = value[4:]

    # DoOR uses "." for paired receptors (e.g., Gr21a.Gr63a). Some upstream
    # metadata uses "+" for co-expression; treat them as equivalent for matching.
    value = value.replace("+", ".")

    return value.upper()


def flywire_glomerulus_from_door_code(raw: Optional[str]) -> str:
    """
    Convert a DoOR glomerulus code (e.g., "DM2", "DP1m") into FlyWire label form.

    Returns an empty string for unknown/ambiguous DoOR codes (e.g., "", "?", "DL2d/v+VC3").
    """
    if raw is None:
        return ""
    if not isinstance(raw, str):
        raw = str(raw)

    code = raw.strip()
    if not code or code == "?":
        return ""

    # DoOR sometimes encodes ambiguous mappings (multiple glomeruli) with "/" or "+".
    if any(sep in code for sep in ("/", "+", ",")):
        return ""

    # Standardize case: letters before digits uppercase, trailing letters lowercase.
    # Examples: "dp1m" -> "DP1m", "vc3l" -> "VC3l", "v" -> "V"
    match = re.match(r"^([A-Za-z]+)(\d+)?([A-Za-z]*)$", code)
    if not match:
        return ""

    prefix, digits, suffix = match.groups()
    if digits is None:
        normalized = prefix.upper()
    else:
        normalized = f"{prefix.upper()}{digits}{suffix.lower()}"

    return f"ORN_{normalized}"
