#!/usr/bin/env python3
"""
Analysis 1: Tuning Correlation vs Connectivity Strength
=======================================================

Tests fundamental hypotheses about structure-function relationships:

H1 (Lateral Inhibition): Dissimilar tuning → Strong LN-mediated inhibition
H2 (Cooperative Coding): Similar tuning → Weak inhibition, possibly PN excitation

Citation:
    Münch, D. & Galizia, C. G. DoOR 2.0 - Comprehensive Mapping of Drosophila
    melanogaster Odorant Responses. Sci. Rep. 6, 21841 (2016).

Usage:
    python scripts/analysis_1_tuning_vs_connectivity.py --threshold 10 --pathway-type inhibitory

Note:
    This analysis now includes a namespace translation layer so DoOR receptor
    names (Or7a) align with FlyWire glomerulus labels (ORN_DL5), preventing
    silent mismatches during correlation tests.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr
from scipy.stats import mannwhitneyu
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from door_toolkit.integration import DoORFlyWireIntegrator
from door_toolkit.integration.door_utils import classify_receptor_tuning, calculate_lifetime_kurtosis

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analysis 1: Test tuning correlation vs connectivity strength",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--door-cache",
        type=str,
        default="door_cache",
        help="Path to DoOR cache directory (default: door_cache)"
    )

    parser.add_argument(
        "--connectomics-data",
        type=str,
        default="data/interglomerular_crosstalk_pathways.csv",
        help="Path to connectomics CSV (default: data/interglomerular_crosstalk_pathways.csv)"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Minimum synapse threshold (default: 1). Higher values (>10) may result in insufficient data. Start with 1-5 for exploratory analysis."
    )

    parser.add_argument(
        "--pathway-type",
        type=str,
        choices=["all", "inhibitory", "excitatory"],
        default="inhibitory",
        help="Pathway type to analyze (default: inhibitory)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/integration/analysis_1",
        help="Output directory (default: output/integration/analysis_1)"
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip plotting (faster for quick analysis)"
    )

    return parser.parse_args()


def analyze_tuning_vs_connectivity(integrator, args):
    """
    Main analysis function.

    Tests correlation between odor tuning similarity and cross-talk connectivity.
    """
    print("=" * 70)
    print("Analysis 1: Tuning Correlation vs Connectivity Strength")
    print("=" * 70)
    print()

    # Get mapped receptors
    receptors = integrator.get_mapped_receptors()
    logger.info(f"Analyzing {len(receptors)} receptors with both DoOR and FlyWire data")

    # Calculate LTK for classification
    ltk_values = integrator.calculate_all_ltk()

    # Build DoOR-indexed connectivity matrix
    logger.info(
        f"Building connectivity matrix (pathway_type={args.pathway_type}, "
        f"threshold={args.threshold})..."
    )

    try:
        connectivity_matrix = integrator.get_connectivity_matrix_door_indexed(
            threshold=args.threshold,
            pathway_type=args.pathway_type
        )

        if connectivity_matrix.index.duplicated().any():
            duplicates = connectivity_matrix.index[connectivity_matrix.index.duplicated()].unique()
            logger.error("❌ CRITICAL: Connectivity matrix has duplicate indices: %s", list(duplicates))
            logger.error("   Fix receptor mapping entries before rerunning.")
            return None

        if connectivity_matrix.columns.duplicated().any():
            duplicates = connectivity_matrix.columns[connectivity_matrix.columns.duplicated()].unique()
            logger.error("❌ CRITICAL: Connectivity matrix has duplicate columns: %s", list(duplicates))
            return None

        logger.info(f"✅ Connectivity matrix shape: {connectivity_matrix.shape}")
        index_preview = list(connectivity_matrix.index[:5])
        logger.info("   Index type: DoOR receptor names")
        logger.info("   Sample indices: %s", index_preview)
    except ValueError as exc:
        logger.error(f"❌ Failed to build connectivity matrix: {exc}")
        raise

    # Get tuning correlation matrix
    logger.info("Extracting tuning correlation matrix...")
    tuning_correlation_matrix = integrator.tuning_correlation_matrix

    # Diagnose namespace compatibility before proceeding
    n_overlap = log_matrix_diagnostics(
        tuning_correlation_matrix,
        connectivity_matrix,
        logger
    )

    if n_overlap < 2:
        logger.error(
            f"❌ CRITICAL: Only {n_overlap} overlapping receptors. "
            f"Cannot perform correlation analysis (need ≥2)."
        )
        raise ValueError(
            f"Insufficient receptor overlap ({n_overlap}). "
            f"Check namespace consistency between DoOR and FlyWire data."
        )

    shared_receptors = sorted(
        set(tuning_correlation_matrix.index) & set(connectivity_matrix.index)
    )
    tuning_subset = tuning_correlation_matrix.loc[
        shared_receptors,
        shared_receptors
    ]
    connectivity_subset = connectivity_matrix.loc[
        shared_receptors,
        shared_receptors
    ]

    logger.info(f"Analyzing {len(shared_receptors)} receptors present in both matrices")

    results_list = []

    for i, receptor1 in enumerate(shared_receptors):
        for receptor2 in shared_receptors[i + 1:]:
            try:
                tuning_val = tuning_subset.at[receptor1, receptor2]
                conn_val = connectivity_subset.at[receptor1, receptor2]
            except KeyError as exc:
                logger.debug("Skipping %s ↔ %s: %s", receptor1, receptor2, exc)
                continue

            if conn_val > 0 and not np.isnan(tuning_val):
                results_list.append({
                    'receptor1': receptor1,
                    'receptor2': receptor2,
                    'tuning_correlation': float(tuning_val),
                    'connectivity_strength': float(conn_val)
                })

    results_df = pd.DataFrame(results_list).sort_values(
        'connectivity_strength',
        ascending=False
    )

    logger.info(f"Found {len(results_df)} valid receptor pairs with connectivity > 0")

    if len(results_df) == 0:
        logger.error(
            f"\n❌ CRITICAL: No connected receptor pairs found!\n"
            f"   Threshold: {args.threshold} synapses\n"
            f"   Pathway type: {args.pathway_type}\n"
            f"   Shared receptors: {len(shared_receptors)}\n\n"
            f"   💡 Try lowering --threshold or using --pathway-type all"
        )
        return None

    if len(results_df) < 2:
        logger.error(
            f"\n❌ INSUFFICIENT DATA: Only {len(results_df)} glomerulus pairs with connectivity > 0.\n"
            f"   Statistical tests require at least 2 pairs.\n\n"
            f"   Current settings:\n"
            f"      Threshold: {args.threshold} synapses\n"
            f"      Pathway type: {args.pathway_type}\n"
            f"      Overlapping receptors: {len(shared_receptors)}\n\n"
            f"   💡 SOLUTIONS:\n"
            f"      1. Lower --threshold (try: 1-5 instead of {args.threshold})\n"
            f"      2. Use --pathway-type all (includes inhibitory + excitatory)\n"
            f"      3. Check receptor mapping completeness\n"
        )

        diagnostic_df = pd.DataFrame({
            'threshold': [args.threshold],
            'pathway_type': [args.pathway_type],
            'n_overlapping_receptors': [len(shared_receptors)],
            'n_valid_pairs': [len(results_df)],
            'error': ['Insufficient data for analysis']
        })

        output_file = Path(args.output_dir) / "diagnostic_report.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_df.to_csv(output_file, index=False)

        logger.info(f"Saved diagnostic report to {output_file}")
        return None

    tuning_nonzero = results_df['tuning_correlation'].to_numpy()
    connectivity_nonzero = results_df['connectivity_strength'].to_numpy()

    logger.info(f"Analyzing {len(results_df)} receptor pairs with connectivity > 0")

    logger.info("\n🔥 Top 5 Strongest Connections:")
    for _, row in results_df.head(5).iterrows():
        logger.info(
            "   %s ↔ %s: %.0f synapses (tuning corr: %.3f)",
            row['receptor1'],
            row['receptor2'],
            row['connectivity_strength'],
            row['tuning_correlation'],
        )

    # Calculate correlations
    print()
    print("Correlation Analysis:")
    print("-" * 70)

    # Spearman (robust to outliers)
    rho_spearman, p_spearman = spearmanr(tuning_nonzero, connectivity_nonzero)
    print(f"Spearman ρ = {rho_spearman:.4f}, p = {p_spearman:.4e}")

    # Pearson (assumes linearity)
    r_pearson, p_pearson = pearsonr(tuning_nonzero, connectivity_nonzero)
    print(f"Pearson r = {r_pearson:.4f}, p = {p_pearson:.4e}")

    print()

    # Hypothesis testing
    print("Hypothesis Testing:")
    print("-" * 70)

    # Split by tuning similarity
    median_tuning = np.median(tuning_nonzero)
    similar_mask = tuning_nonzero > median_tuning
    dissimilar_mask = tuning_nonzero <= median_tuning

    conn_similar = connectivity_nonzero[similar_mask]
    conn_dissimilar = connectivity_nonzero[dissimilar_mask]

    # Mann-Whitney U test
    u_stat, p_mw = mannwhitneyu(conn_similar, conn_dissimilar, alternative='two-sided')

    print(f"Similar tuning (n={len(conn_similar)}):")
    print(f"  Mean connectivity: {np.mean(conn_similar):.2f} synapses")
    print(f"  Median connectivity: {np.median(conn_similar):.2f} synapses")

    print(f"Dissimilar tuning (n={len(conn_dissimilar)}):")
    print(f"  Mean connectivity: {np.mean(conn_dissimilar):.2f} synapses")
    print(f"  Median connectivity: {np.median(conn_dissimilar):.2f} synapses")

    print(f"\nMann-Whitney U test: U = {u_stat:.0f}, p = {p_mw:.4e}")

    # Effect size (Cohen's d)
    effect_size = (np.mean(conn_similar) - np.mean(conn_dissimilar)) / \
                  np.sqrt((np.std(conn_similar)**2 + np.std(conn_dissimilar)**2) / 2)
    print(f"Effect size (Cohen's d): {effect_size:.3f}")

    print()

    # Interpretation
    print("Biological Interpretation:")
    print("-" * 70)

    if args.pathway_type == "inhibitory":
        if rho_spearman < -0.1 and p_spearman < 0.05:
            print("✓ LATERAL INHIBITION CONFIRMED:")
            print("  Dissimilar tuning → Stronger LN-mediated inhibition")
            print("  Supports contrast enhancement hypothesis")
        elif rho_spearman > 0.1 and p_spearman < 0.05:
            print("✗ UNEXPECTED: Similar tuning → Stronger inhibition")
            print("  May indicate competitive inhibition within odor class")
        else:
            print("○ NO CLEAR RELATIONSHIP:")
            print("  Cross-talk may be independent of tuning similarity")
    elif args.pathway_type == "excitatory":
        if rho_spearman > 0.1 and p_spearman < 0.05:
            print("✓ COOPERATIVE CODING CONFIRMED:")
            print("  Similar tuning → Stronger PN-mediated excitation")
            print("  Supports cooperative encoding hypothesis")
    else:
        print(f"Mixed pathway analysis (all types included)")

    print()

    # Classify by LTK
    print("Analysis by Receptor Tuning Class:")
    print("-" * 70)

    # Get LTK for each glomerulus pair
    specialist_pairs = []
    generalist_pairs = []

    for i, receptor1 in enumerate(shared_receptors):
        ltk1 = ltk_values.get(receptor1, 0)

        for j, receptor2 in enumerate(shared_receptors):
            if j <= i:  # Skip lower triangle and diagonal
                continue

            ltk2 = ltk_values.get(receptor2, 0)

            conn_strength = connectivity_subset.iloc[i, j]
            if conn_strength == 0:
                continue

            # Classify
            is_specialist = (ltk1 > 20) or (ltk2 > 20)
            is_generalist = (ltk1 < 0) or (ltk2 < 0)

            if is_specialist:
                specialist_pairs.append(conn_strength)
            if is_generalist:
                generalist_pairs.append(conn_strength)

    if len(specialist_pairs) > 0:
        print(f"Specialist pairs (n={len(specialist_pairs)}):")
        print(f"  Mean connectivity: {np.mean(specialist_pairs):.2f} synapses")

    if len(generalist_pairs) > 0:
        print(f"Generalist pairs (n={len(generalist_pairs)}):")
        print(f"  Mean connectivity: {np.mean(generalist_pairs):.2f} synapses")

    # Test difference
    if len(specialist_pairs) > 0 and len(generalist_pairs) > 0:
        u_spec, p_spec = mannwhitneyu(specialist_pairs, generalist_pairs, alternative='two-sided')
        print(f"\nSpecialist vs Generalist: U = {u_spec:.0f}, p = {p_spec:.4e}")

    print()

    # Save results
    if not args.no_plot:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create plots
        create_plots(
            tuning_nonzero,
            connectivity_nonzero,
            connectivity_subset,
            output_dir,
            args
        )

    # Save summary CSV with receptor identities
    output_base = Path(args.output_dir)
    if output_base.name != "analysis_1":
        csv_dir = output_base / "analysis_1"
    else:
        csv_dir = output_base

    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "tuning_vs_connectivity_data.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"Saved data to {csv_path}")

    return {
        'spearman_rho': rho_spearman,
        'spearman_p': p_spearman,
        'pearson_r': r_pearson,
        'pearson_p': p_pearson,
        'n_pairs': len(tuning_nonzero),
        'effect_size': effect_size
    }


def log_matrix_diagnostics(
    tuning_matrix: pd.DataFrame,
    connectivity_matrix: pd.DataFrame,
    logger
) -> int:
    """
    Log diagnostics comparing tuning and connectivity matrices.

    Args:
        tuning_matrix: DoOR tuning correlation matrix (DoOR namespace).
        connectivity_matrix: Connectivity matrix indexed by DoOR names.
        logger: Logger instance for emitting diagnostics.

    Returns:
        Number of overlapping receptors between the two matrices.
    """
    separator = "=" * 70
    logger.info("\n%s", separator)
    logger.info("MATRIX COMPATIBILITY DIAGNOSTICS")
    logger.info("%s", separator)

    tuning_index_type = type(tuning_matrix.index[0]).__name__ if len(tuning_matrix.index) > 0 else "None"
    connectivity_index_type = type(connectivity_matrix.index[0]).__name__ if len(connectivity_matrix.index) > 0 else "None"

    logger.info("\nTuning Correlation Matrix:")
    logger.info("   Shape: %s", tuning_matrix.shape)
    logger.info("   Index type: %s", tuning_index_type)
    logger.info("   Sample indices: %s", list(tuning_matrix.index[:5]))
    logger.info("   Total receptors: %d", len(tuning_matrix.index))

    logger.info("\nConnectivity Matrix:")
    logger.info("   Shape: %s", connectivity_matrix.shape)
    logger.info("   Index type: %s", connectivity_index_type)
    logger.info("   Sample indices: %s", list(connectivity_matrix.index[:5]))
    logger.info("   Total glomeruli: %d", len(connectivity_matrix.index))

    tuning_names = set(tuning_matrix.index)
    connectivity_names = set(connectivity_matrix.index)
    overlap = tuning_names & connectivity_names

    logger.info("\nIndex Overlap Analysis:")
    logger.info("   Receptors in tuning matrix: %d", len(tuning_names))
    logger.info("   Receptors in connectivity matrix: %d", len(connectivity_names))
    logger.info("   Overlapping receptors: %d", len(overlap))

    if overlap:
        logger.info("   Found %d common receptors", len(overlap))
        logger.info("   Sample overlap: %s", list(overlap)[:5])
    else:
        logger.error("   NO OVERLAP! Namespace mismatch detected!")
        logger.error("   Tuning uses: %s", list(tuning_names)[:5])
        logger.error("   Connectivity uses: %s", list(connectivity_names)[:5])

    only_tuning = tuning_names - connectivity_names
    if only_tuning:
        logger.info("\n   %d receptors only in tuning matrix:", len(only_tuning))
        logger.info("   %s", list(only_tuning)[:10])

    only_connectivity = connectivity_names - tuning_names
    if only_connectivity:
        logger.info("\n   %d receptors only in connectivity matrix:", len(only_connectivity))
        logger.info("   %s", list(only_connectivity)[:10])

    logger.info("%s\n", separator)

    return len(overlap)


def create_plots(tuning, connectivity, conn_matrix, output_dir, args):
    """Create visualization plots."""
    logger.info("Creating plots...")

    # Set style
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.2)

    # Plot 1: Scatter plot with regression line
    fig, ax = plt.subplots(figsize=(10, 8))

    # Color by connectivity strength
    scatter = ax.scatter(
        tuning,
        connectivity,
        c=connectivity,
        cmap='viridis',
        s=50,
        alpha=0.6,
        edgecolors='black',
        linewidths=0.5
    )

    # Add regression line
    z = np.polyfit(tuning, connectivity, 1)
    p = np.poly1d(z)
    x_line = np.linspace(tuning.min(), tuning.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, label=f'Linear fit: y={z[0]:.2f}x+{z[1]:.2f}')

    # Labels
    ax.set_xlabel('Odor Tuning Correlation', fontsize=14)
    ax.set_ylabel(f'Connectivity Strength ({args.pathway_type} pathways, synapses)', fontsize=14)
    ax.set_title(f'Analysis 1: Tuning Similarity vs Connectivity\n(threshold={args.threshold} synapses)', fontsize=16)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Synapse Count', fontsize=12)

    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "scatter_tuning_vs_connectivity.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved scatter plot to {plot_path}")
    plt.close()

    # Plot 2: Heatmap of connectivity matrix
    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        conn_matrix,
        cmap='YlOrRd',
        cbar_kws={'label': 'Synapse Count'},
        square=True,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title(f'Connectivity Matrix ({args.pathway_type} pathways)', fontsize=16)
    ax.set_xlabel('Target Glomerulus', fontsize=12)
    ax.set_ylabel('Source Glomerulus', fontsize=12)

    plt.tight_layout()
    plot_path = output_dir / "heatmap_connectivity_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved heatmap to {plot_path}")
    plt.close()

    logger.info("Plotting complete!")


def main():
    """Main entry point."""
    args = parse_args()

    print()
    print("DoOR-FlyWire Integration: Analysis 1")
    print("=" * 70)
    print()

    # Initialize integrator
    logger.info("Initializing integrator...")
    integrator = DoORFlyWireIntegrator(
        door_cache=args.door_cache,
        connectomics_data=args.connectomics_data
    )

    print()
    print(integrator.summary())
    print()

    # Run analysis
    results = analyze_tuning_vs_connectivity(integrator, args)

    print()
    print("=" * 70)
    print("Analysis 1 Complete!")
    print("=" * 70)
    print()
    print(f"Results saved to: {args.output_dir}")
    print()
    print("Citation:")
    print("  Münch, D. & Galizia, C. G. DoOR 2.0 - Comprehensive Mapping of Drosophila")
    print("  melanogaster Odorant Responses. Sci. Rep. 6, 21841 (2016).")
    print()


if __name__ == "__main__":
    main()
