"""
Multi-Condition Sensitivity Analysis: Hub Receptor Identification
==================================================================

This script performs sensitivity analysis across ALL behavioral conditions
(opto_hex, opto_EB, opto_benz_1) to identify hub receptors that appear
across multiple odorant training paradigms.

Key Questions:
1. Which receptors appear as IMPORTANT across multiple conditions?
2. Is Or49a a generalist hub receptor or condition-specific?
3. Are learning circuits modular (condition-specific) or canonical (shared)?

Example usage:
    python examples/multi_condition_sensitivity_analysis.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")

# Import locally
import importlib.util
spec = importlib.util.spec_from_file_location(
    "receptor_sensitivity_analysis",
    Path(__file__).parent / "receptor_sensitivity_analysis.py"
)
rsa_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rsa_module)
ReceptorSensitivityAnalyzer = rsa_module.ReceptorSensitivityAnalyzer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultiConditionAnalyzer:
    """Analyze receptor importance across multiple behavioral conditions."""

    def __init__(
        self,
        doorcache_path: str,
        behavior_csv_path: str,
        results_dir: str = "behavioral_prediction_results",
    ):
        """Initialize multi-condition analyzer."""
        self.doorcache_path = Path(doorcache_path)
        self.behavior_csv_path = Path(behavior_csv_path)
        self.results_dir = Path(results_dir)

        # Initialize sensitivity analyzer
        self.analyzer = ReceptorSensitivityAnalyzer(
            doorcache_path=str(doorcache_path),
            behavior_csv_path=str(behavior_csv_path),
            results_dir=str(results_dir),
        )

        logger.info("Initialized MultiConditionAnalyzer")

    def run_all_conditions(
        self,
        conditions: List[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Run sensitivity analysis on all specified conditions.

        Args:
            conditions: List of condition names (default: hex, EB, benz_1)

        Returns:
            Dictionary mapping condition -> sensitivity DataFrame
        """
        if conditions is None:
            conditions = ["opto_hex", "opto_EB", "opto_benz_1"]

        logger.info("=" * 80)
        logger.info("MULTI-CONDITION SENSITIVITY ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Conditions to analyze: {conditions}")
        logger.info("")

        sensitivity_results = {}

        for condition in conditions:
            logger.info(f"\n{'='*80}")
            logger.info(f"ANALYZING CONDITION: {condition}")
            logger.info(f"{'='*80}\n")

            try:
                # Run sensitivity analysis
                self.analyzer.run_analysis(
                    condition_name=condition,
                    output_dir=str(self.results_dir),
                )

                # Load results
                results_path = self.results_dir / f"{condition}_sensitivity_results.csv"
                sensitivity_df = pd.read_csv(results_path)
                sensitivity_results[condition] = sensitivity_df

                logger.info(f"✓ {condition} complete!")

            except Exception as e:
                logger.error(f"✗ {condition} failed: {e}")
                continue

        logger.info(f"\n{'='*80}")
        logger.info(f"COMPLETED {len(sensitivity_results)}/{len(conditions)} CONDITIONS")
        logger.info(f"{'='*80}\n")

        return sensitivity_results

    def create_master_comparison_table(
        self,
        sensitivity_results: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Create master comparison table across all conditions.

        Args:
            sensitivity_results: Dict of condition -> sensitivity DataFrame

        Returns:
            Master comparison DataFrame
        """
        logger.info("Creating master comparison table...")

        # Collect all unique receptors
        all_receptors = set()
        for df in sensitivity_results.values():
            all_receptors.update(df["receptor"].values)

        all_receptors = sorted(all_receptors)

        # Build master table
        rows = []

        for receptor in all_receptors:
            row = {"receptor": receptor}

            # Add importance scores for each condition
            for condition, df in sensitivity_results.items():
                receptor_row = df[df["receptor"] == receptor]

                if len(receptor_row) > 0:
                    importance = receptor_row["importance_score"].values[0]
                    percent = receptor_row["percent_increase"].values[0]
                    row[f"{condition}_importance"] = importance
                    row[f"{condition}_percent"] = percent
                else:
                    row[f"{condition}_importance"] = np.nan
                    row[f"{condition}_percent"] = np.nan

            # Calculate hub score (frequency of appearance as important)
            appearances = 0
            total_importance = 0

            for condition in sensitivity_results.keys():
                importance = row.get(f"{condition}_importance", np.nan)
                if not np.isnan(importance) and importance > 0.10:  # >10% threshold
                    appearances += 1
                    total_importance += importance

            row["appearances"] = appearances
            row["hub_score"] = appearances / len(sensitivity_results)
            row["mean_importance"] = total_importance / max(appearances, 1)

            # Calculate robustness (consistency across conditions)
            importances = [
                row[f"{c}_importance"]
                for c in sensitivity_results.keys()
                if not np.isnan(row.get(f"{c}_importance", np.nan))
            ]

            if len(importances) > 0:
                row["robustness"] = min(importances) / max(importances) if max(importances) > 0 else 0
            else:
                row["robustness"] = 0

            rows.append(row)

        # Create DataFrame
        master_df = pd.DataFrame(rows)

        # Sort by hub score (descending)
        master_df = master_df.sort_values(
            ["hub_score", "mean_importance"], ascending=[False, False]
        )

        # Add classification
        def classify_receptor(row):
            if row["hub_score"] >= 0.67:  # Appears in 2+ conditions
                if row["mean_importance"] > 0.30:
                    return "STRONG HUB"
                elif row["mean_importance"] > 0.10:
                    return "MODERATE HUB"
                else:
                    return "WEAK HUB"
            elif row["hub_score"] > 0:
                return "SPECIALIST"
            else:
                return "NOISE"

        master_df["classification"] = master_df.apply(classify_receptor, axis=1)

        logger.info(f"Master table created with {len(master_df)} receptors")

        return master_df

    def generate_hub_receptor_ranking(
        self,
        master_df: pd.DataFrame,
        output_path: str = None,
    ) -> pd.DataFrame:
        """
        Generate hub receptor ranking table.

        Args:
            master_df: Master comparison DataFrame
            output_path: Path to save ranking

        Returns:
            Hub receptor ranking DataFrame
        """
        logger.info("Generating hub receptor ranking...")

        # Filter for receptors with hub_score > 0
        hub_df = master_df[master_df["hub_score"] > 0].copy()

        # Select columns for ranking
        ranking_cols = [
            "receptor",
            "hub_score",
            "appearances",
            "mean_importance",
            "robustness",
            "classification",
        ]

        # Add per-condition importance
        for col in master_df.columns:
            if col.endswith("_importance"):
                ranking_cols.append(col)

        hub_ranking = hub_df[ranking_cols].copy()

        # Add recommendation
        def get_recommendation(row):
            if row["classification"] == "STRONG HUB":
                return "TEST FIRST ⭐⭐⭐ (high priority across conditions)"
            elif row["classification"] == "MODERATE HUB":
                return "TEST SECOND ⭐⭐ (moderate priority)"
            elif row["classification"] == "WEAK HUB":
                return "TEST THIRD ⭐ (low priority)"
            elif row["classification"] == "SPECIALIST":
                return "TEST CONDITION-SPECIFIC (single condition)"
            else:
                return "SKIP (noise)"

        hub_ranking["recommendation"] = hub_ranking.apply(get_recommendation, axis=1)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            hub_ranking.to_csv(output_path, index=False)
            logger.info(f"Saved hub ranking to {output_path}")

        return hub_ranking

    def plot_importance_heatmap(
        self,
        master_df: pd.DataFrame,
        save_to: str,
        dpi: int = 300,
    ):
        """
        Plot heatmap of importance scores across conditions.

        Args:
            master_df: Master comparison DataFrame
            save_to: Path to save figure
            dpi: Resolution
        """
        logger.info("Generating importance heatmap...")

        # Extract importance columns
        conditions = []
        for col in master_df.columns:
            if col.endswith("_importance"):
                condition = col.replace("_importance", "")
                conditions.append(condition)

        # Filter receptors with at least one appearance
        plot_df = master_df[master_df["hub_score"] > 0].copy()

        # Create matrix for heatmap
        matrix_data = []
        receptor_labels = []

        for _, row in plot_df.iterrows():
            receptor_labels.append(row["receptor"])
            importance_values = [
                row[f"{condition}_importance"] * 100  # Convert to percentage
                for condition in conditions
            ]
            matrix_data.append(importance_values)

        matrix = np.array(matrix_data)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, max(8, len(receptor_labels) * 0.4)))

        # Use diverging colormap centered at 0
        im = ax.imshow(
            matrix,
            cmap="RdYlGn",
            aspect="auto",
            vmin=-50,
            vmax=50,
            interpolation="nearest",
        )

        # Set ticks
        ax.set_xticks(range(len(conditions)))
        ax.set_yticks(range(len(receptor_labels)))
        ax.set_xticklabels(conditions, rotation=45, ha="right", fontsize=11)
        ax.set_yticklabels(receptor_labels, fontsize=10)

        # Add text annotations
        for i in range(len(receptor_labels)):
            for j in range(len(conditions)):
                value = matrix[i, j]
                if not np.isnan(value):
                    text_color = "white" if abs(value) > 25 else "black"
                    ax.text(
                        j, i, f"{value:.1f}%",
                        ha="center", va="center",
                        color=text_color, fontsize=9,
                    )

        ax.set_xlabel("Condition", fontsize=13)
        ax.set_ylabel("Receptor", fontsize=13)
        ax.set_title(
            "Receptor Importance Across Conditions\n"
            "Green = Positive (critical), Red = Negative (redundant)",
            fontsize=14, pad=15,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, label="Importance Score (%)")
        cbar.ax.axhline(0, color="black", linewidth=2, linestyle="--", alpha=0.5)

        plt.tight_layout()

        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved importance heatmap to {save_path}")
        plt.close()

    def plot_hub_score_barplot(
        self,
        hub_ranking: pd.DataFrame,
        save_to: str,
        dpi: int = 300,
    ):
        """
        Plot bar chart of hub scores.

        Args:
            hub_ranking: Hub ranking DataFrame
            save_to: Path to save figure
            dpi: Resolution
        """
        logger.info("Generating hub score bar plot...")

        fig, ax = plt.subplots(figsize=(12, 7))

        receptors = hub_ranking["receptor"].values
        hub_scores = hub_ranking["hub_score"].values
        classifications = hub_ranking["classification"].values

        # Color by classification
        color_map = {
            "STRONG HUB": "darkgreen",
            "MODERATE HUB": "orange",
            "WEAK HUB": "gold",
            "SPECIALIST": "steelblue",
            "NOISE": "gray",
        }
        colors = [color_map.get(c, "gray") for c in classifications]

        ax.barh(range(len(receptors)), hub_scores, color=colors, alpha=0.8, edgecolor="black")

        ax.set_yticks(range(len(receptors)))
        ax.set_yticklabels(receptors, fontsize=10)
        ax.set_xlabel("Hub Score (Frequency Across Conditions)", fontsize=12)
        ax.set_ylabel("Receptor", fontsize=12)
        ax.set_title(
            "Hub Receptor Ranking\n"
            "Score = (# conditions where important) / (total conditions)",
            fontsize=14,
        )
        ax.set_xlim(0, 1.1)

        # Add threshold lines
        ax.axvline(0.67, color="red", linestyle="--", alpha=0.5, linewidth=2, label="Hub threshold (2/3)")
        ax.axvline(0.33, color="orange", linestyle="--", alpha=0.5, linewidth=2, label="Specialist (1/3)")

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor="darkgreen", label="Strong Hub (score ≥ 0.67, high importance)"),
            Patch(facecolor="orange", label="Moderate Hub (score ≥ 0.67, moderate importance)"),
            Patch(facecolor="steelblue", label="Specialist (score = 0.33)"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()

        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved hub score bar plot to {save_path}")
        plt.close()

    def plot_receptor_radar(
        self,
        master_df: pd.DataFrame,
        receptor_name: str,
        save_to: str,
        dpi: int = 300,
    ):
        """
        Plot radar/spider diagram for a specific receptor across conditions.

        Args:
            master_df: Master comparison DataFrame
            receptor_name: Receptor to plot
            save_to: Path to save figure
            dpi: Resolution
        """
        logger.info(f"Generating radar plot for {receptor_name}...")

        # Extract receptor row
        receptor_row = master_df[master_df["receptor"] == receptor_name]

        if len(receptor_row) == 0:
            logger.warning(f"Receptor {receptor_name} not found")
            return

        receptor_row = receptor_row.iloc[0]

        # Extract conditions and importance scores
        conditions = []
        importances = []

        for col in master_df.columns:
            if col.endswith("_importance"):
                condition = col.replace("_importance", "").replace("opto_", "")
                importance = receptor_row[col]

                if not np.isnan(importance):
                    conditions.append(condition)
                    importances.append(importance * 100)  # Convert to percentage

        if len(conditions) == 0:
            logger.warning(f"No data for {receptor_name}")
            return

        # Create radar plot
        angles = np.linspace(0, 2 * np.pi, len(conditions), endpoint=False).tolist()
        importances_plot = importances + [importances[0]]  # Close the loop
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))

        ax.plot(angles, importances_plot, "o-", linewidth=2, color="steelblue")
        ax.fill(angles, importances_plot, alpha=0.25, color="steelblue")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(conditions, fontsize=12)
        ax.set_ylim(-50, 100)
        ax.set_yticks([-50, 0, 50, 100])
        ax.set_yticklabels(["-50%", "0%", "50%", "100%"], fontsize=10)

        # Add reference lines
        ax.axhline(0, color="black", linestyle="-", linewidth=1, alpha=0.3)
        ax.axhline(50, color="red", linestyle="--", linewidth=1, alpha=0.3)
        ax.axhline(10, color="orange", linestyle="--", linewidth=1, alpha=0.3)

        ax.set_title(
            f"Receptor {receptor_name} - Importance Across Conditions\n"
            f"Hub Score: {receptor_row['hub_score']:.2f} | "
            f"Classification: {receptor_row['classification']}",
            fontsize=14, pad=20,
        )

        ax.grid(True)

        plt.tight_layout()

        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved radar plot to {save_path}")
        plt.close()

    def generate_multi_condition_report(
        self,
        master_df: pd.DataFrame,
        hub_ranking: pd.DataFrame,
        save_to: str,
    ):
        """
        Generate comprehensive multi-condition analysis report.

        Args:
            master_df: Master comparison DataFrame
            hub_ranking: Hub ranking DataFrame
            save_to: Path to save report
        """
        logger.info("Generating multi-condition report...")

        lines = [
            "=" * 80,
            "MULTI-CONDITION SENSITIVITY ANALYSIS REPORT",
            "=" * 80,
            f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This report identifies hub receptors that appear as important across",
            "multiple behavioral conditions (opto_hex, opto_EB, opto_benz_1).",
            "",
            "=" * 80,
            "EXECUTIVE SUMMARY",
            "=" * 80,
            "",
        ]

        # Count receptors by classification
        strong_hubs = master_df[master_df["classification"] == "STRONG HUB"]
        moderate_hubs = master_df[master_df["classification"] == "MODERATE HUB"]
        weak_hubs = master_df[master_df["classification"] == "WEAK HUB"]
        specialists = master_df[master_df["classification"] == "SPECIALIST"]

        lines.append(f"Total receptors analyzed: {len(master_df)}")
        lines.append(f"Strong hubs (appear in 2-3 conditions, high importance): {len(strong_hubs)}")
        lines.append(f"Moderate hubs (appear in 2-3 conditions, moderate importance): {len(moderate_hubs)}")
        lines.append(f"Weak hubs (appear in 2-3 conditions, low importance): {len(weak_hubs)}")
        lines.append(f"Specialists (appear in 1 condition): {len(specialists)}")
        lines.append("")

        # Top hub receptors
        lines.append("=" * 80)
        lines.append("TOP HUB RECEPTORS (Appear Across Multiple Conditions)")
        lines.append("=" * 80)
        lines.append("")

        top_hubs = hub_ranking.head(5)

        lines.append(f"{'Receptor':<12} | {'Hub Score':<10} | {'Appearances':<12} | {'Mean Imp':<10} | {'Classification':<15}")
        lines.append("-" * 80)

        for idx, row in top_hubs.iterrows():
            receptor = row['receptor']
            hub_score = row['hub_score']
            appearances = row['appearances']
            mean_imp = row['mean_importance']
            classification = row['classification']

            lines.append(
                f"{receptor:<12} | "
                f"{hub_score:>10.2f} | "
                f"{appearances:>12d} | "
                f"{mean_imp:>10.2f} | "
                f"{classification:<15}"
            )

        lines.append("")

        # Detailed findings for top hub
        if len(top_hubs) > 0:
            top_receptor = top_hubs.iloc[0]["receptor"]
            top_row = master_df[master_df["receptor"] == top_receptor].iloc[0]

            lines.append("=" * 80)
            lines.append(f"SPOTLIGHT: {top_receptor} (Top Hub Receptor)")
            lines.append("=" * 80)
            lines.append("")
            lines.append(f"Hub Score: {top_row['hub_score']:.2f}")
            lines.append(f"Classification: {top_row['classification']}")
            lines.append(f"Appearances: {top_row['appearances']} / 3 conditions")
            lines.append(f"Mean Importance: {top_row['mean_importance']:.2%}")
            lines.append(f"Robustness: {top_row['robustness']:.2f}")
            lines.append("")
            lines.append("Per-Condition Breakdown:")

            for col in master_df.columns:
                if col.endswith("_importance"):
                    condition = col.replace("_importance", "")
                    importance = top_row[col]
                    if not np.isnan(importance):
                        lines.append(f"  {condition:15s}: {importance:>7.2%}")

            lines.append("")

        # Experimental recommendations
        lines.append("=" * 80)
        lines.append("EXPERIMENTAL RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")

        lines.append("PRIORITY 1: Test Strong Hub Receptors")
        if len(strong_hubs) > 0:
            for idx, row in strong_hubs.iterrows():
                receptor = row['receptor']
                lines.append(f"  - {receptor}: Test across ALL conditions (generalist)")
        else:
            lines.append("  - None identified")

        lines.append("")
        lines.append("PRIORITY 2: Test Moderate Hubs")
        if len(moderate_hubs) > 0:
            for idx, row in moderate_hubs.iterrows():
                receptor = row['receptor']
                lines.append(f"  - {receptor}: Secondary priority")
        else:
            lines.append("  - None identified")

        lines.append("")
        lines.append("PRIORITY 3: Test Specialists (Condition-Specific)")
        if len(specialists) > 0:
            for idx, row in specialists.iterrows():
                receptor = row['receptor']
                lines.append(f"  - {receptor}: Test in specific condition only")
        else:
            lines.append("  - None identified")

        lines.append("")

        # Biological interpretation
        lines.append("=" * 80)
        lines.append("BIOLOGICAL INTERPRETATION")
        lines.append("=" * 80)
        lines.append("")

        if len(strong_hubs) > 0:
            lines.append("FINDING: Strong hub receptors detected!")
            lines.append("  Interpretation: Learning uses a CANONICAL generalist circuit")
            lines.append("  Prediction: Silencing hub receptors will impair ALL conditions")
            lines.append("  Implication: Core sensory gates for learned behavior exist")
        elif len(moderate_hubs) > 0:
            lines.append("FINDING: Moderate hubs + specialists detected")
            lines.append("  Interpretation: Learning uses MIXED strategy (generalists + specialists)")
            lines.append("  Prediction: Hub receptors important, but specialists fine-tune")
            lines.append("  Implication: Combination of canonical + modular circuits")
        else:
            lines.append("FINDING: Mostly specialists, no strong hubs")
            lines.append("  Interpretation: Learning circuits are MODULAR (condition-specific)")
            lines.append("  Prediction: Each condition uses different receptors")
            lines.append("  Implication: No universal learning code, circuits are flexible")

        lines.append("")

        # Caveats
        lines.append("=" * 80)
        lines.append("CAVEATS & LIMITATIONS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("1. Small sample size (n=7 odorants per condition)")
        lines.append("2. Negative importance scores in some conditions (overfitting)")
        lines.append("3. Different lambda values used (hex/benz: 0.1, EB: 0.0001)")
        lines.append("4. Receptor correlations may confound importance estimates")
        lines.append("5. Need experimental validation with optogenetics")
        lines.append("")

        # Next steps
        lines.append("=" * 80)
        lines.append("NEXT STEPS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("1. Map top hub receptors to FlyWire mushroom body connectivity")
        lines.append("2. Design optogenetic experiments targeting hub receptors first")
        lines.append("3. Test hub receptors across ALL 3 conditions (validate generalist role)")
        lines.append("4. Collect more behavioral data (aim for n≥20 per condition)")
        lines.append("5. Run diagnostic analysis on EB and benz_1 (check for redundancy)")
        lines.append("")
        lines.append("=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)

        # Save report
        report_text = "\n".join(lines)

        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report_text)

        logger.info(f"Saved multi-condition report to {save_path}")

        return report_text

    def run_complete_analysis(
        self,
        conditions: List[str] = None,
        output_dir: str = None,
    ):
        """Run complete multi-condition sensitivity analysis."""

        if conditions is None:
            conditions = ["opto_hex", "opto_EB", "opto_benz_1"]

        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("COMPLETE MULTI-CONDITION SENSITIVITY ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Conditions: {conditions}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("")

        # Step 1: Run sensitivity on all conditions
        logger.info("\nStep 1: Running sensitivity analysis on all conditions...")
        sensitivity_results = self.run_all_conditions(conditions=conditions)

        # Step 2: Create master comparison table
        logger.info("\nStep 2: Creating master comparison table...")
        master_df = self.create_master_comparison_table(sensitivity_results)
        master_df.to_csv(output_dir / "multi_condition_comparison.csv", index=False)

        # Step 3: Generate hub receptor ranking
        logger.info("\nStep 3: Generating hub receptor ranking...")
        hub_ranking = self.generate_hub_receptor_ranking(
            master_df=master_df,
            output_path=str(output_dir / "hub_receptor_ranking.csv"),
        )

        # Step 4: Generate visualizations
        logger.info("\nStep 4: Generating visualizations...")
        self.plot_importance_heatmap(
            master_df=master_df,
            save_to=str(output_dir / "multi_condition_importance_heatmap.png"),
        )

        self.plot_hub_score_barplot(
            hub_ranking=hub_ranking,
            save_to=str(output_dir / "hub_score_barplot.png"),
        )

        # Generate radar plots for top 3 hub receptors
        top_hubs = hub_ranking.head(3)
        for _, row in top_hubs.iterrows():
            receptor = row["receptor"]
            self.plot_receptor_radar(
                master_df=master_df,
                receptor_name=receptor,
                save_to=str(output_dir / f"radar_{receptor}.png"),
            )

        # Step 5: Generate report
        logger.info("\nStep 5: Generating multi-condition report...")
        self.generate_multi_condition_report(
            master_df=master_df,
            hub_ranking=hub_ranking,
            save_to=str(output_dir / "multi_condition_analysis_report.txt"),
        )

        logger.info("\n" + "=" * 80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nGenerated files in {output_dir}:")
        logger.info("  - multi_condition_comparison.csv")
        logger.info("  - hub_receptor_ranking.csv")
        logger.info("  - multi_condition_importance_heatmap.png")
        logger.info("  - hub_score_barplot.png")
        logger.info("  - radar_*.png (top 3 hub receptors)")
        logger.info("  - multi_condition_analysis_report.txt")
        logger.info("")

        # Print top hubs
        print("\n" + "=" * 80)
        print("TOP HUB RECEPTORS")
        print("=" * 80)
        for i, (_, row) in enumerate(hub_ranking.head(5).iterrows(), 1):
            print(f"{i}. {row['receptor']:<12} - Hub Score: {row['hub_score']:.2f} - {row['classification']}")
        print("=" * 80)


def main():
    """Run complete multi-condition sensitivity analysis."""

    DOORCACHE_PATH = "door_cache"
    BEHAVIOR_CSV_PATH = (
        "/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions(Strictest)/"
        "reaction_rates_summary_unordered.csv"
    )
    OUTPUT_DIR = "behavioral_prediction_results"

    analyzer = MultiConditionAnalyzer(
        doorcache_path=DOORCACHE_PATH,
        behavior_csv_path=BEHAVIOR_CSV_PATH,
        results_dir=OUTPUT_DIR,
    )

    # Run complete analysis on all 3 conditions
    analyzer.run_complete_analysis(
        conditions=["opto_hex", "opto_EB", "opto_benz_1"],
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
