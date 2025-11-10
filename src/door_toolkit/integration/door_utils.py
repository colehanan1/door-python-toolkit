"""
DoOR Data Utilities
==================

Utilities for loading and processing DoOR 2.0 olfactory response data.

Citation:
    Münch, D. & Galizia, C. G. DoOR 2.0 - Comprehensive Mapping of Drosophila
    melanogaster Odorant Responses. Sci. Rep. 6, 21841 (2016).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

from .odorant_mapper import OdorantMapper

_odorant_mapper: Optional[OdorantMapper] = None


def _get_odorant_mapper() -> OdorantMapper:
    """Return a cached OdorantMapper instance."""
    global _odorant_mapper
    if _odorant_mapper is None:
        _odorant_mapper = OdorantMapper()
    return _odorant_mapper


def load_door_response_matrix(cache_path: str = "door_cache") -> pd.DataFrame:
    """
    Load DoOR 2.0 response matrix.

    The response matrix contains normalized responses [0, 1] range for:
    - Rows: 78 responding units (receptors/glomeruli)
    - Columns: 693 odorants (identified by InChIKey or name)
    - Values: Normalized responses (negative values = inhibitory)

    Args:
        cache_path: Path to DoOR cache directory

    Returns:
        DataFrame with receptors as rows, odorants as columns

    Example:
        >>> door_matrix = load_door_response_matrix("door_cache")
        >>> or47b_responses = door_matrix.loc['Or47b']
        >>> print(f"Or47b responds to {(or47b_responses > 0.5).sum()} odorants")
    """
    cache_path = Path(cache_path)

    # Try to load from cache
    matrix_file = cache_path / "response_matrix_norm.parquet"

    if not matrix_file.exists():
        raise FileNotFoundError(
            f"DoOR response matrix not found at {matrix_file}. "
            f"Please run DoOR extraction first or ensure cache_path is correct."
        )

    logger.info(f"Loading DoOR response matrix from {matrix_file}")
    response_matrix = pd.read_parquet(matrix_file)

    # Ensure receptors are rows (responding units)
    # DoOR convention: receptors as index, odorants as columns
    if response_matrix.shape[0] > response_matrix.shape[1]:
        logger.warning("Matrix appears transposed, transposing to receptors × odorants")
        response_matrix = response_matrix.T

    logger.info(f"Loaded DoOR matrix: {response_matrix.shape[0]} receptors × {response_matrix.shape[1]} odorants")

    return response_matrix


def load_receptor_mapping(mapping_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load DoOR receptor → FlyWire glomerulus mapping.

    Args:
        mapping_path: Path to mapping CSV. If None, uses default location.

    Returns:
        DataFrame with columns: door_name, flywire_glomerulus, receptor_type,
                                sensillum, tuning_class, notes

    Example:
        >>> mapping = load_receptor_mapping()
        >>> or47b_glom = mapping[mapping['door_name'] == 'Or47b']['flywire_glomerulus'].values[0]
        >>> print(f"Or47b maps to {or47b_glom}")
    """
    if mapping_path is None:
        # Use default location
        mapping_path = Path(__file__).parent.parent.parent.parent / "data" / "mappings" / "door_to_flywire_mapping.csv"

    mapping_path = Path(mapping_path)

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Receptor mapping not found at {mapping_path}. "
            f"Please ensure the mapping file exists."
        )

    logger.info(f"Loading receptor mapping from {mapping_path}")
    mapping = pd.read_csv(mapping_path)

    logger.info(f"Loaded mapping for {len(mapping)} receptors")

    return mapping


def map_door_to_flywire(door_names: List[str], mapping: Optional[pd.DataFrame] = None) -> Dict[str, str]:
    """
    Map DoOR receptor names to FlyWire glomerulus identifiers.

    Args:
        door_names: List of DoOR receptor names (e.g., ['Or47b', 'Or42b'])
        mapping: Receptor mapping DataFrame. If None, loads from default.

    Returns:
        Dictionary mapping DoOR names to FlyWire glomeruli

    Example:
        >>> door_names = ['Or47b', 'Or42b', 'Or69a']
        >>> flywire_map = map_door_to_flywire(door_names)
        >>> print(flywire_map)
        {'Or47b': 'ORN_VA1d', 'Or42b': 'ORN_DM1', 'Or69a': 'ORN_D'}
    """
    if mapping is None:
        mapping = load_receptor_mapping()

    # Create mapping dictionary
    door_to_flywire = {}
    missing = []

    for door_name in door_names:
        matches = mapping[mapping['door_name'] == door_name]
        if len(matches) > 0:
            door_to_flywire[door_name] = matches.iloc[0]['flywire_glomerulus']
        else:
            missing.append(door_name)
            logger.warning(f"No mapping found for {door_name}")

    if missing:
        logger.warning(f"Missing mappings for {len(missing)} receptors: {missing[:5]}...")

    logger.info(f"Mapped {len(door_to_flywire)}/{len(door_names)} receptors to FlyWire glomeruli")

    return door_to_flywire


def calculate_lifetime_kurtosis(response_vector: np.ndarray) -> float:
    """
    Calculate Lifetime Kurtosis (LTK) as per Münch & Galizia 2016.

    LTK measures receptor tuning breadth:
    - High LTK (>20): Narrowly tuned specialist (e.g., Or82a: 63.88, Or47b: 33.12)
    - Low/Negative LTK (<0): Broadly tuned generalist (e.g., Or69a: -0.26, Or35a: -0.44)

    Formula:
        LTK = (1/n * Σ((r_i - μ)^4)) / (1/n * Σ((r_i - μ)^2))^2 - 3

    Where:
        r_i = response to odorant i
        μ = mean response across all odorants
        n = number of odorants

    Args:
        response_vector: Array of responses to all odorants (can contain mixed types)

    Returns:
        LTK value (float)

    Example:
        >>> or47b_responses = door_matrix.loc['Or47b'].values
        >>> ltk = calculate_lifetime_kurtosis(or47b_responses)
        >>> print(f"Or47b LTK: {ltk:.2f} (narrow tuning expected ~33)")

    Reference:
        Münch & Galizia (2016) Sci Rep 6:21841
        https://doi.org/10.1038/srep21841
    """
    # Convert to numeric, coerce errors to NaN (handles strings like 'SFR')
    if isinstance(response_vector, pd.Series):
        responses = pd.to_numeric(response_vector, errors='coerce').values
    else:
        responses = pd.to_numeric(response_vector, errors='coerce')

    # Remove NaN and non-numeric values
    responses = responses[~np.isnan(responses)]

    # Need at least 4 data points for kurtosis calculation
    if len(responses) < 4:
        return np.nan

    n = len(responses)
    mean_resp = np.mean(responses)

    # Calculate moments
    deviations = responses - mean_resp

    # Fourth moment
    numerator = np.sum(deviations**4) / n

    # Second moment squared
    denominator = (np.sum(deviations**2) / n)**2

    # Avoid division by zero (all responses identical)
    if denominator == 0:
        return np.nan

    ltk = (numerator / denominator) - 3

    return ltk


def calculate_tuning_correlation_matrix(response_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate pairwise tuning correlation between all receptors.

    Uses Pearson correlation to quantify similarity in odor response profiles.
    High correlation = similar tuning, Low/negative = dissimilar tuning.

    Args:
        response_matrix: DoOR response matrix (receptors × odorants)

    Returns:
        Correlation matrix (receptors × receptors)

    Example:
        >>> door_matrix = load_door_response_matrix()
        >>> tuning_corr = calculate_tuning_correlation_matrix(door_matrix)
        >>> print(f"Or47b vs Or42b correlation: {tuning_corr.loc['Or47b', 'Or42b']:.3f}")
    """
    logger.info("Calculating tuning correlation matrix...")

    # Calculate Pearson correlation
    tuning_corr = response_matrix.T.corr()

    logger.info(f"Calculated correlation matrix: {tuning_corr.shape[0]} × {tuning_corr.shape[1]}")

    return tuning_corr


def get_odorant_activated_receptors(
    odorant_name: str,
    response_matrix: pd.DataFrame,
    activation_threshold: float = 0.3,
    use_abs_value: bool = False
) -> pd.Series:
    """
    Get receptors activated by an odorant (case-insensitive name support).

    Args:
        odorant_name: Common name OR InChIKey (any capitalization)
        response_matrix: DoOR response matrix (receptors × odorants)
        activation_threshold: Activation threshold
        use_abs_value: Include inhibitory responses if True

    Returns:
        pd.Series of activated receptors with response values

    Raises:
        ValueError: If odorant is missing from mapping or DoOR matrix.
    """
    try:
        mapper = _get_odorant_mapper()
    except FileNotFoundError as exc:
        logger.error("Failed to load odorant mapper: %s", exc)
        raise

    inchikey = mapper.get_inchikey(odorant_name)

    if inchikey is None:
        suggestions = mapper.search_by_name(odorant_name, max_results=5)
        if suggestions:
            suggest_str = ", ".join(f"'{name}'" for name, _ in suggestions)
            raise ValueError(
                f"Odorant '{odorant_name}' not found in mapping.\n"
                f"💡 Did you mean: {suggest_str}?\n"
                "💡 Use --list-odorants to see all available names"
            )
        raise ValueError(
            f"Odorant '{odorant_name}' not found.\n"
            "💡 Use --list-odorants to see all available names"
        )

    if inchikey not in response_matrix.columns:
        common_name = mapper.get_common_name(inchikey) or odorant_name
        raise ValueError(
            f"Odorant '{common_name}' (InChIKey: {inchikey}) "
            "not found in DoOR response matrix.\n"
            "This odorant is in the mapping but not in the DoOR database."
        )

    responses = pd.to_numeric(response_matrix[inchikey], errors="coerce")

    if use_abs_value:
        activated = responses[responses.abs() > activation_threshold]
    else:
        activated = responses[responses > activation_threshold]

    activated = activated.dropna().sort_values(ascending=False)
    common_name = mapper.get_common_name(inchikey) or odorant_name

    logger.info(
        "Odorant '%s' (InChIKey: %s) activates %d/%d receptors (threshold=%.2f)",
        common_name,
        inchikey,
        len(activated),
        len(responses),
        activation_threshold,
    )

    return activated


def classify_receptor_tuning(ltk_value: float) -> str:
    """
    Classify receptor tuning breadth based on LTK value.

    Args:
        ltk_value: Lifetime Kurtosis value

    Returns:
        Tuning class string: 'very_narrow', 'narrow', 'moderate', 'broad', 'very_broad'

    Example:
        >>> ltk = calculate_lifetime_kurtosis(or47b_responses)
        >>> tuning = classify_receptor_tuning(ltk)
        >>> print(f"Or47b tuning: {tuning}")
    """
    if ltk_value > 40:
        return "very_narrow"
    elif ltk_value > 20:
        return "narrow"
    elif ltk_value > 0:
        return "moderate"
    elif ltk_value > -10:
        return "broad"
    else:
        return "very_broad"


def get_receptor_best_ligands(
    response_matrix: pd.DataFrame,
    receptor_name: str,
    top_n: int = 10
) -> pd.Series:
    """
    Get top N best ligands for a receptor.

    Args:
        response_matrix: DoOR response matrix
        receptor_name: Receptor name (e.g., 'Or47b')
        top_n: Number of top ligands to return

    Returns:
        Series with odorant names and response values, sorted by response

    Example:
        >>> door_matrix = load_door_response_matrix()
        >>> best = get_receptor_best_ligands(door_matrix, 'Or47b', 10)
        >>> print("Or47b best ligands:")
        >>> for odor, resp in best.items():
        >>>     print(f"  {odor}: {resp:.3f}")
    """
    if receptor_name not in response_matrix.index:
        raise ValueError(f"Receptor '{receptor_name}' not found in response matrix")

    responses = response_matrix.loc[receptor_name]
    best_ligands = responses.nlargest(top_n)

    logger.info(f"Top {top_n} ligands for {receptor_name}:")
    for i, (odor, resp) in enumerate(best_ligands.items(), 1):
        logger.info(f"  {i}. {odor}: {resp:.3f}")

    return best_ligands
