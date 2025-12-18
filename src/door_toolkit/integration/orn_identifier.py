"""
ORN/Glomerulus Identifier Resolution
=====================================

Robust normalization and fuzzy matching for ORN and glomerulus identifiers.

This module provides functions to resolve messy user inputs (e.g., "DL3", "dl3",
"ORN DL3", "Ir31a") to canonical glomerulus labels (e.g., "ORN_DL3", "ORN_Ir31a")
used internally by the connectomics and FlyWire modules.

Key Features:
- Handles multiple naming conventions (underscores, hyphens, spaces)
- Case-insensitive matching with deterministic normalization
- Fuzzy matching with similarity ranking for suggestions
- Clear error messages with actionable suggestions
- Stateless and testable design

Example:
    >>> available = {"ORN_DL3", "ORN_DL5", "ORN_VA1d"}
    >>> resolve_orn_identifier("dl3", available)
    'ORN_DL3'
    >>> resolve_orn_identifier("DL5", available)
    'ORN_DL5'
    >>> resolve_orn_identifier("Ir31a", {"ORN_Ir31a"})
    'ORN_Ir31a'
"""

import re
import difflib
from typing import List, Set, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# Receptor-to-glomerulus mapping for all olfactory receptors
# In FlyWire, neurons are labeled by their glomerulus name, not receptor name
# Source: FlyWire community labels (ORN_<GLOMERULUS>; <RECEPTOR>)
# Format: receptor name (uppercase) → glomerulus name
RECEPTOR_TO_GLOMERULUS = {
    # Odorant receptors (Or)
    'OR2A': 'DA4m',
    'OR7A': 'DL5',
    'OR9A': 'VM3',
    'OR10A': 'DL1',
    'OR13A': 'DC2',
    'OR23A': 'DA3',
    'OR33A': 'DA2',
    'OR33C': 'VC1',
    'OR42A': 'VM7d',
    'OR42B': 'DM1',
    'OR43A': 'DA4l',
    'OR43B': 'VM2',
    'OR46A': 'VA7l',
    'OR47A': 'DM3',
    'OR47B': 'VA1v',
    'OR49A': 'DL4',
    'OR49B': 'VA5',
    'OR59B': 'DM4',
    'OR59C': 'VM7v',
    'OR67A': 'DM6',
    'OR67B': 'VA3',
    'OR67C': 'VC4',
    'OR67D': 'DA1',
    'OR69A': 'D',
    'OR71A': 'VC2',
    'OR82A': 'VA6',
    'OR83C': 'DC3',
    'OR85A': 'DM5',
    'OR85B': 'VM5d',
    'OR85D': 'VA4',
    'OR88A': 'VA1d',
    'OR92A': 'VA2',
    'OR98A': 'VM5v',
    # Ionotropic receptors (Ir)
    'IR31A': 'VL2p',
    'IR41A': 'VC5',
    'IR64A': 'DC4',  # Note: Ir64a projects to both DC4 and DP1m
    'IR75A': 'DP1l',
    'IR75B': 'DL2d',
    'IR75C': 'DL2v',
    'IR75D': 'VL1',
    'IR76A': 'VM4',
    'IR84A': 'VL2a',
    'IR92A': 'VM1',
    # Gustatory receptors (Gr)
    'GR21A': 'V',
}


def normalize_orn_identifier(raw: str, prefer_glomerulus_prefix: bool = True) -> str:
    """
    Normalize an ORN/glomerulus identifier to canonical format.

    This function standardizes various input formats to the canonical "ORN_<GLOM>"
    format used internally. It handles case variations, different separators, and
    common prefix/suffix patterns.

    Normalization steps:
    1. Strip whitespace and convert to uppercase
    2. Remove common prefixes/suffixes: "GLOMERULUS", "GLOM", "ORN " (space-separated)
    3. Replace hyphens and spaces with underscores
    4. Re-apply canonical prefix "ORN_" if not present

    Args:
        raw: Raw identifier string (e.g., "DL3", "orn_dl3", "ORN-DL3")
        prefer_glomerulus_prefix: If True, ensure "ORN_" prefix is added

    Returns:
        Normalized identifier (e.g., "ORN_DL3")

    Example:
        >>> normalize_orn_identifier("dl3")
        'ORN_DL3'
        >>> normalize_orn_identifier("ORN DL5")
        'ORN_DL5'
        >>> normalize_orn_identifier("Ir31a")
        'ORN_IR31A'
    """
    if not raw or not isinstance(raw, str):
        return raw

    # Step 1: Basic cleanup
    normalized = raw.strip().upper()

    # Step 2: Remove common prefixes/suffixes (case-insensitive patterns)
    # Remove "GLOMERULUS", "GLOM" if they appear as separate words
    normalized = re.sub(r'\bGLOMERULUS\b', '', normalized)
    normalized = re.sub(r'\bGLOM\b', '', normalized)

    # Remove "ORN" only if it's followed by space, hyphen, or at start
    # This preserves "ORN_" but removes "ORN " or "ORN-"
    normalized = re.sub(r'^ORN[\s\-]+', '', normalized)

    # Step 3: Replace separators with underscores
    normalized = re.sub(r'[\s\-]+', '_', normalized)

    # Step 4: Remove leading/trailing underscores
    normalized = normalized.strip('_')

    # Step 5: Map receptors to their glomerulus names (FlyWire convention)
    # Check if this looks like a receptor (pattern: OR/IR/GR + number + letter)
    if re.match(r'^(OR|IR|GR)\d+[A-Z]$', normalized):
        # Try to find the glomerulus mapping
        if normalized in RECEPTOR_TO_GLOMERULUS:
            glom = RECEPTOR_TO_GLOMERULUS[normalized]
            logger.info(f"Mapped receptor '{raw}' → glomerulus '{glom}'")
            normalized = glom

    # Step 6: Re-apply canonical prefix if needed
    if prefer_glomerulus_prefix and not normalized.startswith('ORN_'):
        normalized = f'ORN_{normalized}'

    return normalized


def generate_orn_candidates(raw: str) -> List[str]:
    """
    Generate candidate identifiers for matching.

    This function produces multiple plausible variations of the input identifier
    to maximize chances of finding a match in the available set.

    Candidates generated:
    1. Raw input (as-is)
    2. Normalized with "ORN_" prefix
    3. Normalized without prefix
    4. Uppercase raw
    5. Lowercase raw
    6. With underscores instead of hyphens/spaces

    Args:
        raw: Raw identifier string

    Returns:
        List of candidate strings, ordered by likelihood

    Example:
        >>> generate_orn_candidates("DL3")
        ['DL3', 'ORN_DL3', 'DL3', 'DL3', 'dl3', 'DL3']
    """
    if not raw or not isinstance(raw, str):
        return []

    candidates = []

    # Candidate 1: Raw input
    candidates.append(raw.strip())

    # Candidate 2: Normalized with ORN_ prefix
    normalized_with_prefix = normalize_orn_identifier(raw, prefer_glomerulus_prefix=True)
    candidates.append(normalized_with_prefix)

    # Candidate 3: Normalized without prefix
    normalized_without_prefix = normalize_orn_identifier(raw, prefer_glomerulus_prefix=False)
    if normalized_without_prefix != normalized_with_prefix:
        candidates.append(normalized_without_prefix)

    # Candidate 4: Uppercase raw
    upper_raw = raw.strip().upper()
    if upper_raw not in candidates:
        candidates.append(upper_raw)

    # Candidate 5: Lowercase raw
    lower_raw = raw.strip().lower()
    if lower_raw not in candidates:
        candidates.append(lower_raw)

    # Candidate 6: Replace common separators
    underscore_variant = re.sub(r'[\s\-]+', '_', raw.strip())
    if underscore_variant not in candidates:
        candidates.append(underscore_variant)

    # Candidate 7: For Ir/Or receptors, try with ORN_ prefix
    if re.match(r'^(Ir|Or|Gr)\d+[a-z]', raw.strip(), re.IGNORECASE):
        receptor_with_prefix = f"ORN_{raw.strip()}"
        if receptor_with_prefix not in candidates:
            candidates.append(receptor_with_prefix)

    return candidates


def suggest_orn_identifiers(
    raw: str,
    available: Set[str],
    k: int = 10,
    similarity_threshold: float = 0.3
) -> List[Tuple[str, float]]:
    """
    Suggest similar ORN identifiers from available set using fuzzy matching.

    Uses difflib's SequenceMatcher to compute string similarity and returns
    the top k matches sorted by similarity score (descending).

    Args:
        raw: Raw identifier to match
        available: Set of available canonical identifiers
        k: Maximum number of suggestions to return
        similarity_threshold: Minimum similarity score (0.0-1.0) to include

    Returns:
        List of (identifier, similarity_score) tuples, sorted by score descending

    Example:
        >>> available = {"ORN_DL3", "ORN_DL5", "ORN_DA3"}
        >>> suggest_orn_identifiers("DL33", available, k=2)
        [('ORN_DL3', 0.857), ('ORN_DL5', 0.714)]
    """
    if not raw or not available:
        return []

    # Normalize input for comparison
    normalized_raw = normalize_orn_identifier(raw, prefer_glomerulus_prefix=True)

    # Compute similarity scores
    similarities = []
    for candidate in available:
        # Try matching against both the candidate and its base name (without ORN_)
        score1 = difflib.SequenceMatcher(None, normalized_raw, candidate).ratio()

        # Also try matching the raw input against the glomerulus part only
        glom_part = candidate.replace('ORN_', '')
        score2 = difflib.SequenceMatcher(None, raw.upper(), glom_part).ratio()

        # Use the higher score
        score = max(score1, score2)

        if score >= similarity_threshold:
            similarities.append((candidate, score))

    # Sort by score descending, then alphabetically for ties
    similarities.sort(key=lambda x: (-x[1], x[0]))

    return similarities[:k]


def resolve_orn_identifier(
    raw: str,
    available: Set[str],
    *,
    prefer_glomerulus_prefix: bool = True,
    fuzzy_threshold: float = 0.85,
    strict: bool = False
) -> str:
    """
    Resolve a raw ORN identifier to a canonical form from the available set.

    This is the main entry point for identifier resolution. It attempts to match
    the input against available identifiers using several strategies:

    1. Exact match (any candidate)
    2. Fuzzy match (if similarity >= fuzzy_threshold and not strict mode)
    3. Raise ValueError with suggestions if no match found

    Resolution behavior:
    - If any candidate matches available exactly: return it immediately
    - If fuzzy_threshold met and not strict: return best match with debug log
    - Otherwise: raise ValueError with top suggestions and diagnostic info

    Args:
        raw: Raw identifier string (e.g., "DL3", "Ir31a")
        available: Set of canonical identifiers to match against
        prefer_glomerulus_prefix: If True, normalize to "ORN_<GLOM>" format
        fuzzy_threshold: Minimum similarity (0.0-1.0) for automatic fuzzy matching
        strict: If True, only allow exact matches (no fuzzy matching)

    Returns:
        Canonical identifier from available set

    Raises:
        ValueError: If no match found, with suggestions and diagnostic info

    Example:
        >>> available = {"ORN_DL3", "ORN_DL5"}
        >>> resolve_orn_identifier("dl3", available)
        'ORN_DL3'
        >>> resolve_orn_identifier("DL99", available)  # doctest: +SKIP
        ValueError: ORN identifier 'DL99' not found. Did you mean: ORN_DL3, ORN_DL5?
    """
    if not raw or not isinstance(raw, str):
        raise ValueError(f"Invalid ORN identifier: {raw}")

    if not available:
        raise ValueError(
            f"Cannot resolve ORN identifier '{raw}': no available identifiers provided"
        )

    # Step 1: Try exact match with all candidates
    candidates = generate_orn_candidates(raw)

    for candidate in candidates:
        if candidate in available:
            # Exact match found
            if candidate != raw:
                logger.debug(
                    f"Resolved ORN identifier '{raw}' → '{candidate}' (exact match)"
                )
            return candidate

    # Step 2: Try fuzzy matching (if not strict mode)
    if not strict:
        suggestions = suggest_orn_identifiers(
            raw, available, k=1, similarity_threshold=fuzzy_threshold
        )

        if suggestions:
            best_match, score = suggestions[0]
            logger.info(
                f"Fuzzy-matched ORN identifier '{raw}' → '{best_match}' "
                f"(similarity: {score:.2f})"
            )
            return best_match

    # Step 3: No match found - raise with suggestions
    suggestions = suggest_orn_identifiers(raw, available, k=10, similarity_threshold=0.3)

    error_msg_parts = [
        f"ORN identifier '{raw}' not found in available glomeruli."
    ]

    # Show what was tried
    normalized = normalize_orn_identifier(raw, prefer_glomerulus_prefix=True)
    error_msg_parts.append(
        f"\nNormalized to: '{normalized}'"
    )
    error_msg_parts.append(
        f"Tried candidates: {candidates[:5]}"  # Show first 5
    )

    # Show suggestions
    if suggestions:
        error_msg_parts.append("\nDid you mean one of these?")
        for identifier, score in suggestions[:10]:
            error_msg_parts.append(f"  - {identifier} (similarity: {score:.2f})")
    else:
        error_msg_parts.append(
            f"\nNo similar identifiers found. Available: {sorted(list(available))[:20]}"
        )

    raise ValueError("\n".join(error_msg_parts))


def get_available_glomeruli(network) -> Set[str]:
    """
    Extract available glomerulus identifiers from a CrossTalkNetwork.

    This is a convenience function to get the set of valid identifiers for
    resolution from a network object.

    Args:
        network: CrossTalkNetwork instance

    Returns:
        Set of glomerulus identifier strings

    Example:
        >>> from door_toolkit.connectomics import CrossTalkNetwork
        >>> network = CrossTalkNetwork.from_csv('pathways.csv')
        >>> available = get_available_glomeruli(network)
        >>> "ORN_DL3" in available
        True
    """
    if hasattr(network, 'data') and hasattr(network.data, 'glomeruli'):
        return set(network.data.glomeruli)
    elif hasattr(network, 'glomeruli'):
        return set(network.glomeruli)
    else:
        raise ValueError(
            "Cannot extract glomeruli from network object. "
            "Expected CrossTalkNetwork with 'data.glomeruli' or 'glomeruli' attribute."
        )


# Convenience function for common use case
def resolve_glomerulus(
    identifier: str,
    network,
    strict: bool = False
) -> str:
    """
    Convenience function to resolve a glomerulus identifier using a network.

    This combines get_available_glomeruli and resolve_orn_identifier into a
    single function call for the common use case.

    Args:
        identifier: Raw glomerulus identifier
        network: CrossTalkNetwork instance
        strict: If True, only allow exact matches

    Returns:
        Canonical glomerulus identifier

    Example:
        >>> from door_toolkit.connectomics import CrossTalkNetwork
        >>> network = CrossTalkNetwork.from_csv('pathways.csv')
        >>> resolve_glomerulus("DL3", network)
        'ORN_DL3'
    """
    available = get_available_glomeruli(network)
    return resolve_orn_identifier(identifier, available, strict=strict)
