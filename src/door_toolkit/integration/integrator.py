"""
DoOR-FlyWire Integration
========================

Main class for integrating olfactory response data with connectomics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import logging

from door_toolkit import DoOREncoder
from door_toolkit.connectomics import CrossTalkNetwork
from door_toolkit.integration.door_utils import (
    load_door_response_matrix,
    load_receptor_mapping,
    calculate_tuning_correlation_matrix,
    calculate_lifetime_kurtosis,
    map_door_to_flywire
)
from door_toolkit.integration.mapping_accounting import (
    compute_mapping_stats,
    log_mapping_stats,
    format_mapping_summary,
    is_larval_receptor,
)
from door_toolkit.integration.receptor_identifier import normalize_receptor_identifier

logger = logging.getLogger(__name__)


class DoORFlyWireIntegrator:
    """
    Main class for structure-function integration analysis.

    Combines DoOR 2.0 olfactory response data with FlyWire connectomics to reveal
    how network structure relates to functional odor coding.

    Attributes:
        door_encoder: DoOREncoder instance for odorant encoding
        network: CrossTalkNetwork instance for connectomics
        door_matrix: DoOR response matrix (receptors × odorants)
        receptor_mapping: Mapping between DoOR and FlyWire nomenclature
        tuning_correlation: Receptor tuning correlation matrix

    Example:
        >>> integrator = DoORFlyWireIntegrator(
        ...     door_cache="door_cache",
        ...     connectomics_data="data/interglomerular_crosstalk_pathways.csv"
        ... )
        >>>
        >>> # Run analyses
        >>> tuning_results = integrator.analyze_tuning_vs_connectivity()
        >>> odor_net = integrator.extract_odor_network("acetic acid")
    """

    def __init__(
        self,
        door_cache: str = "door_cache",
        connectomics_data: str = "data/interglomerular_crosstalk_pathways.csv",
        mapping_path: Optional[str] = None,
        strict_single_cell_annotation: bool = False,
    ):
        """
        Initialize integrator.

        Args:
            door_cache: Path to DoOR cache directory
            connectomics_data: Path to FlyWire connectomics CSV
            mapping_path: Path to receptor mapping CSV (optional)
            strict_single_cell_annotation: If True, exclude mappings that require
                glomerulus-level interpretation for gene-level claims (currently
                `Or22b → ORN_DM2` is excluded as a sensitivity analysis).
        """
        logger.info("Initializing DoOR-FlyWire integrator...")
        self.strict_single_cell_annotation = strict_single_cell_annotation

        # Load DoOR encoder
        logger.info("Loading DoOR encoder...")
        self.door_encoder = DoOREncoder(door_cache, use_torch=False)

        # Load DoOR response matrix
        logger.info("Loading DoOR response matrix...")
        self.door_matrix = load_door_response_matrix(door_cache)

        # Load receptor mapping
        if mapping_path is None:
            base_dir = Path(__file__).resolve().parents[3] / "data" / "mappings"
            authoritative_path = base_dir / "door_to_flywire_mapping.csv"
            deprecated_complete_path = base_dir / "door_to_flywire_mapping_complete.csv"

            if authoritative_path.exists():
                mapping_path = str(authoritative_path)
                logger.info("Using authoritative DoOR→FlyWire mapping (%s)", authoritative_path)
            elif deprecated_complete_path.exists():
                mapping_path = str(deprecated_complete_path)
                logger.warning(
                    "Using deprecated mapping file (%s). "
                    "Regenerate authoritative mapping with scripts/generate_complete_receptor_mapping.py.",
                    deprecated_complete_path,
                )
            else:
                logger.error(
                    "No receptor mapping file found. Expected %s",
                    authoritative_path,
                )
                mapping_path = None

        logger.info("Loading receptor mapping...")
        self.receptor_mapping = load_receptor_mapping(mapping_path)

        # Load connectomics network
        logger.info("Loading FlyWire connectomics network...")
        self.network = CrossTalkNetwork.from_csv(connectomics_data)

        # Calculate tuning correlations
        logger.info("Calculating tuning correlations...")
        self.tuning_correlation = calculate_tuning_correlation_matrix(self.door_matrix)

        # Create mapping dictionaries
        self._create_mappings()

        mapped_receptors = sorted(
            set(self.door_matrix.index) & set(self.door_to_flywire.keys())
        )
        if mapped_receptors:
            self.tuning_correlation_matrix = self.tuning_correlation.loc[
                mapped_receptors,
                mapped_receptors
            ]
        else:
            self.tuning_correlation_matrix = self.tuning_correlation

        logger.info("Integration initialization complete!")
        logger.info(f"  DoOR: {self.door_matrix.shape[0]} receptors × {self.door_matrix.shape[1]} odorants")
        logger.info(f"  FlyWire: {len(self.network.data.glomeruli)} glomeruli, {len(self.network.data.pathways)} pathways")
        logger.info(
            f"  Integration: {self.mapping_stats['n_receptors_mapped']} receptors → "
            f"{self.mapping_stats['n_unique_glomeruli_from_mapped_receptors']} unique glomeruli"
        )

    def _create_mappings(self):
        """Create bidirectional mapping dictionaries."""
        # DoOR → FlyWire (normalization-safe)
        # Key insight: receptor identifiers may differ only by capitalization across
        # DoOR cache vs mapping CSV. Normalize both sides so mappings aren't missed.
        self.door_to_flywire = {}
        self.flywire_to_door = {}

        if "door_name" not in self.receptor_mapping.columns or "flywire_glomerulus" not in self.receptor_mapping.columns:
            raise ValueError("Mapping CSV must contain columns: door_name, flywire_glomerulus")

        normalized_mapping: Dict[str, str] = {}
        ambiguous_keys: set[str] = set()

        grouped = self.receptor_mapping.groupby(
            self.receptor_mapping["door_name"].map(normalize_receptor_identifier),
            dropna=False,
        )

        for key, grp in grouped:
            if not key:
                continue

            targets = [
                str(v).strip()
                for v in grp["flywire_glomerulus"].tolist()
                if v is not None and str(v).strip() != ""
            ]
            if not targets:
                continue

            distinct_targets = sorted(set(targets))

            # If the mapping artifact explicitly marks ambiguity, skip it by default:
            # adult-brain analyses require a single glomerulus label per DoOR unit.
            if "is_ambiguous" in grp.columns:
                flagged = grp["is_ambiguous"].astype(str).str.strip().str.lower().isin({"yes", "true", "1", "y"})
                if bool(flagged.any()):
                    ambiguous_keys.add(key)
                    continue

            if len(distinct_targets) != 1:
                ambiguous_keys.add(key)
                continue

            target = distinct_targets[0]
            if not target.startswith("ORN_"):
                # Inventory definition: mapped means mapped to a valid FlyWire ORN_* label.
                continue

            normalized_mapping[key] = target

        if ambiguous_keys:
            preview = sorted(ambiguous_keys)[:10]
            logger.info(
                "Skipping %d ambiguous DoOR units (multi-glomerulus mappings): %s",
                len(ambiguous_keys),
                preview,
            )

        # Prefer DoOR-cache receptor naming for keys so downstream tuning matrices line up.
        for receptor in self.door_matrix.index:
            if self.strict_single_cell_annotation and normalize_receptor_identifier(receptor) == "OR22B":
                # Or22a/Or22b both target DM2; FlyWire ORN_ labels are glomerulus-level.
                # In strict mode we exclude Or22b to avoid gene-level over-interpretation.
                continue
            key = normalize_receptor_identifier(receptor)
            flywire_name = normalized_mapping.get(key)
            if flywire_name:
                self.door_to_flywire[receptor] = flywire_name
                # flywire_to_door is inherently many-to-one; keep first for determinism.
                self.flywire_to_door.setdefault(flywire_name, receptor)

        # Compute mapping statistics to prevent receptor/glomerulus count confusion
        # NOTE: DoOR receptors map to FlyWire glomeruli (many-to-one possible)
        self.mapping_stats = compute_mapping_stats(
            self.door_to_flywire,
            note="DoOR → FlyWire integration mapping",
            adult_only=False  # DoOR includes all receptors
        )

        # Log mapping statistics with CLEAR distinction between receptors and glomeruli
        logger.info("")  # Blank line for readability
        logger.info("=" * 70)
        logger.info("RECEPTOR → GLOMERULUS MAPPING STATISTICS")
        logger.info("=" * 70)
        logger.info(f"  Receptors mapped (DoOR): {self.mapping_stats['n_receptors_mapped']}")
        logger.info(f"  Unique glomeruli (FlyWire): {self.mapping_stats['n_unique_glomeruli_from_mapped_receptors']}")

        if self.mapping_stats['collision_count'] > 0:
            logger.info(f"  Many-to-one collapses: {self.mapping_stats['collision_count']} glomeruli receive ≥2 receptors")
            for collision_line in self.mapping_stats['collision_summary'][:5]:  # Show first 5
                logger.info(f"    - {collision_line}")
            if self.mapping_stats['collision_count'] > 5:
                logger.info(f"    ... and {self.mapping_stats['collision_count'] - 5} more")
        else:
            logger.info("  1:1 mapping (no collisions)")

        logger.info("=" * 70)
        logger.info("")

    def get_mapped_receptors(self, *, adult_only: bool = True) -> List[str]:
        """
        Get list of receptors that have both DoOR data and FlyWire connectivity.

        By default, this returns **adult-only** receptors because FlyWire connectomics
        data are adult brain annotations. Larval-only receptors are explicitly excluded
        to prevent accidental inclusion in adult-only analyses.

        Returns:
            List of receptor names (DoOR nomenclature)
        """
        # Receptors in DoOR matrix
        door_receptors = set(self.door_matrix.index)

        # Receptors with FlyWire mapping
        mapped_receptors = set(self.door_to_flywire.keys())

        # Intersection
        available = door_receptors & mapped_receptors

        if adult_only:
            excluded_larval = sorted([r for r in available if is_larval_receptor(r)])
            if excluded_larval:
                logger.info(
                    "Adult-only mode: excluding %d larval receptors: %s",
                    len(excluded_larval),
                    excluded_larval,
                )
            available = {r for r in available if not is_larval_receptor(r)}

        logger.info(
            "Found %d receptors with both DoOR and FlyWire data%s",
            len(available),
            " (adult-only)" if adult_only else "",
        )

        return sorted(list(available))

    def get_connectivity_matrix(
        self,
        receptor_list: Optional[List[str]] = None,
        pathway_type: str = "all",
        min_synapses: int = 5
    ) -> pd.DataFrame:
        """
        Get connectivity matrix between glomeruli.

        Args:
            receptor_list: List of DoOR receptor names. If None, uses all mapped receptors.
            pathway_type: 'all', 'inhibitory' (LN-mediated), or 'excitatory' (PN-mediated)
            min_synapses: Minimum synapse threshold

        Returns:
            Connectivity matrix (glomeruli × glomeruli) with synapse counts
        """
        if receptor_list is None:
            receptor_list = self.get_mapped_receptors(adult_only=True)

        # Map to FlyWire glomeruli
        glom_list = [self.door_to_flywire[r] for r in receptor_list if r in self.door_to_flywire]
        glom_index = {glom: idx for idx, glom in enumerate(glom_list)}

        # Set threshold
        self.network.set_min_synapse_threshold(min_synapses)

        # Build connectivity matrix
        connectivity = np.zeros((len(glom_list), len(glom_list)))

        for i, source_glom in enumerate(glom_list):
            # Get pathways from this glomerulus
            try:
                pathways = self.network.get_pathways_from_orn(source_glom, by_glomerulus=True)

                for pathway in pathways:
                    # Find target glomerulus
                    target_glom = pathway.get('target_glomerulus') or pathway.get('level2_glomerulus')
                    if not target_glom:
                        continue

                    j = glom_index.get(target_glom)
                    if j is None:
                        continue

                    synapses = pathway.get('synapse_count_step2', 0)
                    level1_category = pathway.get('level1_category')

                    if pathway_type == "inhibitory" and level1_category == "Local_Neuron":
                        connectivity[i, j] += synapses
                    elif pathway_type == "excitatory" and level1_category == "Projection_Neuron":
                        connectivity[i, j] += synapses
                    elif pathway_type == "all":
                        connectivity[i, j] += synapses
            except Exception as e:
                logger.warning(f"Could not get pathways for {source_glom}: {e}")

        # Create DataFrame
        conn_df = pd.DataFrame(
            connectivity,
            index=glom_list,
            columns=glom_list
        )

        return conn_df

    def get_connectivity_matrix_door_indexed(
        self,
        threshold: int = 1,
        pathway_type: str = "inhibitory"
    ) -> pd.DataFrame:
        """
        Build a connectivity matrix indexed by DoOR receptor names.

        Tuning correlation matrices are already expressed in the DoOR namespace
        (e.g., Or7a, Or47b). FlyWire connectivity matrices, however, are based
        on glomerulus labels (e.g., ORN_DL5). This helper translates the raw
        FlyWire matrix so both datasets share identical indices, enabling
        element-wise comparisons.

        Args:
            threshold: Minimum synapse count required for including a pathway.
            pathway_type: 'inhibitory', 'excitatory', or 'all'.

        Returns:
            pd.DataFrame: Connectivity matrix whose rows/columns are DoOR
            receptor names.

        Raises:
            ValueError: If no overlapping receptors can be mapped or if the raw
            connectivity matrix is empty.
        """
        logger.info(
            "Building DoOR-indexed connectivity matrix "
            f"(pathway_type={pathway_type}, threshold={threshold})"
        )

        mapped_receptors = self.get_mapped_receptors()
        if not mapped_receptors:
            raise ValueError(
                "No receptors have both DoOR responses and FlyWire mappings."
            )

        flywire_connectivity = self.get_connectivity_matrix(
            receptor_list=mapped_receptors,
            pathway_type=pathway_type,
            min_synapses=threshold
        )

        if flywire_connectivity.empty:
            raise ValueError("FlyWire connectivity matrix is empty.")

        index_sample = flywire_connectivity.index[:5].tolist() if len(flywire_connectivity.index) > 0 else []
        logger.debug(
            "Raw FlyWire connectivity matrix shape: %s, index sample: %s",
            flywire_connectivity.shape,
            index_sample
        )

        # Translate FlyWire glomerulus identifiers to DoOR receptor names
        flywire_to_door_mapping: Dict[str, str] = {}

        for flywire_name in flywire_connectivity.index:
            door_name = self.flywire_to_door.get(flywire_name)
            if door_name:
                flywire_to_door_mapping[flywire_name] = door_name
            else:
                logger.debug("No DoOR mapping for FlyWire glomerulus: %s", flywire_name)

        if not flywire_to_door_mapping:
            raise ValueError(
                "No FlyWire glomeruli could be mapped to DoOR receptors. "
                "Verify that receptor mappings are loaded correctly."
            )

        mapped_flywire_names = list(flywire_to_door_mapping.keys())
        connectivity_subset = flywire_connectivity.loc[
            mapped_flywire_names,
            mapped_flywire_names
        ].copy()

        door_names_index = [flywire_to_door_mapping[name] for name in connectivity_subset.index]
        door_names_columns = [flywire_to_door_mapping[name] for name in connectivity_subset.columns]

        connectivity_subset.index = door_names_index
        connectivity_subset.columns = door_names_columns

        # ========================================================================
        # DUPLICATE RECEPTOR DIAGNOSTIC BLOCK
        # ========================================================================
        if connectivity_subset.index.duplicated().any():
            logger.warning(
                "Duplicate receptor indices detected after remapping; analyzing root cause..."
            )

            duplicated_receptors = connectivity_subset.index[
                connectivity_subset.index.duplicated(keep=False)
            ]
            unique_duplicates = duplicated_receptors.unique()

            logger.info("   %d receptors have duplicate entries:", len(unique_duplicates))

            for receptor in unique_duplicates[:5]:
                matching_glom = [
                    (door_name, glom)
                    for glom, door_name in self.flywire_to_door.items()
                    if door_name == receptor
                ]

                receptor_rows = connectivity_subset.loc[receptor]

                if isinstance(receptor_rows, pd.DataFrame):
                    n_copies = len(receptor_rows)
                    sample_values = receptor_rows.iloc[:, :3].values

                    logger.info("   - %s: %d glomeruli variants", receptor, n_copies)
                    logger.info("       FlyWire mappings: %s", matching_glom)
                    logger.info("       Connectivity values (sample):")

                    for i, row in enumerate(sample_values):
                        logger.info("         Copy %d: %s", i + 1, row[:3])

                    if n_copies > 1:
                        first_row = receptor_rows.iloc[0]
                        all_identical = all(
                            receptor_rows.iloc[i].equals(first_row)
                            for i in range(1, n_copies)
                        )

                        if all_identical:
                            logger.warning(
                                "       → TRUE DUPLICATE (identical values) - use .first() or .drop_duplicates()"
                            )
                        else:
                            logger.warning(
                                "       → BIOLOGICAL VARIANTS (different values) - use .max() or .sum()"
                            )
        # ========================================================================

        if connectivity_subset.index.duplicated().any():
            logger.warning(
                "Duplicate receptor indices detected after remapping; taking MAX (strongest pathway)."
            )

            duplicated_count = connectivity_subset.index.duplicated().sum()
            logger.info("   Aggregating %d duplicate rows using MAX", duplicated_count)

            connectivity_subset = connectivity_subset.groupby(level=0, axis=0).max()

            logger.info("   ✅ Aggregated to %d unique receptors", len(connectivity_subset))

        if connectivity_subset.columns.duplicated().any():
            logger.warning(
                "Duplicate receptor columns detected after remapping; taking MAX (strongest pathway)."
            )

            duplicated_count = connectivity_subset.columns.duplicated().sum()
            logger.info("   Aggregating %d duplicate columns using MAX", duplicated_count)

            connectivity_subset = connectivity_subset.T.groupby(level=0).max().T

            logger.info(
                "   ✅ Aggregated to %d unique receptors",
                connectivity_subset.shape[1]
            )

        logger.info(
            "✅ Created DoOR-indexed connectivity matrix: %s (%d receptors)",
            connectivity_subset.shape,
            len(connectivity_subset.index)
        )
        logger.debug("DoOR index sample: %s", door_names_index[:5])

        return connectivity_subset

    def calculate_all_ltk(self) -> pd.Series:
        """
        Calculate Lifetime Kurtosis for all receptors in DoOR matrix.

        Returns:
            Series with receptor names and LTK values

        Note:
            Automatically handles non-numeric values (like 'SFR') in DoOR matrix
            by converting to numeric and coercing errors to NaN.
        """
        logger.info("Calculating LTK for all receptors...")

        ltk_values = {}
        skipped = 0

        for receptor in self.door_matrix.index:
            # Get responses (may contain non-numeric values)
            responses = self.door_matrix.loc[receptor]

            # Calculate LTK (function handles conversion to numeric)
            ltk = calculate_lifetime_kurtosis(responses)

            if np.isnan(ltk):
                skipped += 1

            ltk_values[receptor] = ltk

        ltk_series = pd.Series(ltk_values)

        # Filter out NaN values for statistics
        valid_ltk = ltk_series.dropna()

        logger.info(f"Calculated LTK for {len(ltk_series)} receptors")
        logger.info(f"  Valid LTK values: {len(valid_ltk)}")
        logger.info(f"  Skipped (insufficient data): {skipped}")
        if len(valid_ltk) > 0:
            logger.info(f"  Mean LTK: {valid_ltk.mean():.2f}")
            logger.info(f"  Range: {valid_ltk.min():.2f} to {valid_ltk.max():.2f}")

        return ltk_series

    def get_glomerulus_tuning_correlation(
        self,
        glom1: str,
        glom2: str
    ) -> float:
        """
        Get tuning correlation between two glomeruli.

        Args:
            glom1: FlyWire glomerulus name (e.g., 'ORN_DL5')
            glom2: FlyWire glomerulus name

        Returns:
            Pearson correlation coefficient
        """
        # Map to DoOR names
        if glom1 not in self.flywire_to_door or glom2 not in self.flywire_to_door:
            return np.nan

        receptor1 = self.flywire_to_door[glom1]
        receptor2 = self.flywire_to_door[glom2]

        if receptor1 not in self.tuning_correlation.index or receptor2 not in self.tuning_correlation.columns:
            return np.nan

        return self.tuning_correlation.loc[receptor1, receptor2]

    def summary(self) -> str:
        """Generate summary string of integrated dataset."""
        mapped_receptors = self.get_mapped_receptors()

        lines = [
            "DoOR-FlyWire Integration Summary",
            "=" * 60,
            f"DoOR database:",
            f"  Receptors: {self.door_matrix.shape[0]}",
            f"  Odorants: {self.door_matrix.shape[1]}",
            f"  Data points: {self.door_matrix.size:,}",
            "",
            f"FlyWire connectomics:",
            f"  Glomeruli: {len(self.network.data.glomeruli)}",
            f"  Pathways: {len(self.network.data.pathways):,}",
            f"  Neurons: {len(self.network.data.neurons):,}",
            "",
            f"Receptor → Glomerulus Mapping:",
            f"  Mapped receptors: {self.mapping_stats['n_receptors_mapped']}",
            f"  Unique glomeruli: {self.mapping_stats['n_unique_glomeruli_from_mapped_receptors']}",
            f"  Coverage: {100 * len(mapped_receptors) / self.door_matrix.shape[0]:.1f}%",
            f"  Collisions: {self.mapping_stats['collision_count']} glomeruli receive ≥2 receptors",
        ]

        return "\n".join(lines)
