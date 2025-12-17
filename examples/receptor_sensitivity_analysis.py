"""
Receptor Sensitivity Analysis for Behavioral Prediction
========================================================

This script performs sensitivity analysis on LASSO-selected receptors to identify
which receptors are CRITICAL vs REDUNDANT for predicting learned behavior (PER).

Workflow:
1. Load LASSO model results and behavioral data
2. Fit baseline model with all sparse receptors
3. Drop each receptor one-by-one and measure performance impact
4. Calculate importance scores (% MSE increase when receptor removed)
5. Create priority matrix for experimental validation
6. Generate visualizations and comprehensive report

Key Metrics:
- Importance Score: (MSE_without_R - Baseline_MSE) / Baseline_MSE
- Higher score = more critical receptor
- Score > 50%: CRITICAL (test with optogenetics first)
- Score 10-50%: IMPORTANT (secondary target)
- Score < 10%: REDUNDANT (skip in experiments)

Example usage:
    python examples/receptor_sensitivity_analysis.py
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, LeaveOneOut
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

from door_toolkit.pathways import LassoBehavioralPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ReceptorSensitivityAnalyzer:
    """
    Analyze receptor importance through systematic dropout analysis.

    This class implements sensitivity analysis by iteratively removing each
    receptor and measuring the impact on prediction accuracy.
    """

    def __init__(
        self,
        doorcache_path: str,
        behavior_csv_path: str,
        results_dir: str = "behavioral_prediction_results",
    ):
        """
        Initialize sensitivity analyzer.

        Args:
            doorcache_path: Path to DoOR cache directory
            behavior_csv_path: Path to reaction_rates_summary_unordered.csv
            results_dir: Directory containing LASSO model results
        """
        self.doorcache_path = Path(doorcache_path)
        self.behavior_csv_path = Path(behavior_csv_path)
        self.results_dir = Path(results_dir)

        # Initialize predictor
        self.predictor = LassoBehavioralPredictor(
            doorcache_path=str(doorcache_path),
            behavior_csv_path=str(behavior_csv_path),
            scale_features=True,
            scale_targets=False,
        )

        logger.info("Initialized ReceptorSensitivityAnalyzer")

    def load_model_results(self, condition_name: str) -> Dict:
        """
        Load saved LASSO model results.

        Args:
            condition_name: Condition name (e.g., "opto_hex")

        Returns:
            Dictionary with model metadata and results
        """
        model_path = self.results_dir / f"{condition_name}_model.json"

        if not model_path.exists():
            raise FileNotFoundError(f"Model results not found: {model_path}")

        with open(model_path, "r") as f:
            results = json.load(f)

        logger.info(f"Loaded model results for {condition_name}")
        logger.info(f"  λ = {results['lambda_value']:.6f}")
        logger.info(f"  Baseline MSE = {results['cv_mse']:.6f}")
        logger.info(f"  Receptors selected: {results['n_receptors_selected']}")

        return results

    def prepare_feature_matrix(
        self,
        condition_name: str,
        trained_odorant: str,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        Prepare feature matrix for sensitivity analysis.

        Args:
            condition_name: Condition name
            trained_odorant: Trained odorant name

        Returns:
            Tuple of (X, y, test_odorants, receptor_names)
        """
        # Fit the model to get feature matrix
        results = self.predictor.fit_behavior(
            condition_name=condition_name,
            trained_odorant=trained_odorant,
            lambda_range=[0.1],  # Use a single lambda for consistency
            prediction_mode="test_odorant",
        )

        X = results.feature_matrix
        y = results.actual_per
        test_odorants = results.test_odorants
        receptor_names = results.receptor_names

        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Test odorants: {len(test_odorants)}")

        return X, y, test_odorants, receptor_names

    def fit_baseline_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        receptor_indices: List[int],
        lambda_alpha: float,
    ) -> Tuple[float, float, np.ndarray]:
        """
        Fit baseline LASSO model with all sparse receptors.

        Args:
            X: Feature matrix (n_samples, n_receptors)
            y: Target vector (n_samples,)
            receptor_indices: Indices of selected receptors
            lambda_alpha: LASSO regularization parameter

        Returns:
            Tuple of (baseline_mse, baseline_r2, y_pred)
        """
        # Extract features for selected receptors only
        X_sparse = X[:, receptor_indices]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sparse)

        # Fit LASSO with Leave-One-Out CV
        lasso = Lasso(alpha=lambda_alpha, max_iter=10000, random_state=42)

        # Use LOO-CV for small samples
        loo = LeaveOneOut()
        cv_scores_r2 = cross_val_score(lasso, X_scaled, y, cv=loo, scoring="r2")
        cv_scores_mse = cross_val_score(
            lasso, X_scaled, y, cv=loo, scoring="neg_mean_squared_error"
        )

        baseline_r2 = float(np.mean(cv_scores_r2))
        baseline_mse = float(-np.mean(cv_scores_mse))

        # Fit on full data for predictions
        lasso.fit(X_scaled, y)
        y_pred = lasso.predict(X_scaled)

        logger.info(f"Baseline Model Performance:")
        logger.info(f"  Receptors used: {len(receptor_indices)}")
        logger.info(f"  Baseline MSE: {baseline_mse:.6f}")
        logger.info(f"  Baseline R²: {baseline_r2:.6f}")

        return baseline_mse, baseline_r2, y_pred

    def sensitivity_loop(
        self,
        X: np.ndarray,
        y: np.ndarray,
        receptor_names: List[str],
        sparse_receptors: List[str],
        lambda_alpha: float,
        baseline_mse: float,
    ) -> pd.DataFrame:
        """
        Perform sensitivity analysis by dropping each receptor iteratively.

        Args:
            X: Feature matrix (n_samples, n_receptors)
            y: Target vector (n_samples,)
            receptor_names: All receptor names
            sparse_receptors: Selected sparse receptors
            lambda_alpha: LASSO regularization parameter
            baseline_mse: Baseline MSE with all receptors

        Returns:
            DataFrame with sensitivity results
        """
        results_list = []

        # Get indices of sparse receptors
        receptor_name_to_idx = {name: i for i, name in enumerate(receptor_names)}
        sparse_indices = [receptor_name_to_idx[r] for r in sparse_receptors]

        logger.info(f"\nStarting sensitivity loop for {len(sparse_receptors)} receptors...")

        for receptor_to_drop in sparse_receptors:
            logger.info(f"\n  Analyzing: {receptor_to_drop}")

            # Create feature matrix WITHOUT this receptor
            remaining_indices = [
                idx for idx in sparse_indices
                if receptor_names[idx] != receptor_to_drop
            ]

            X_without = X[:, remaining_indices]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_without)

            # Re-fit LASSO with SAME lambda
            lasso = Lasso(alpha=lambda_alpha, max_iter=10000, random_state=42)

            # Use LOO-CV
            loo = LeaveOneOut()
            cv_scores_mse = cross_val_score(
                lasso, X_scaled, y, cv=loo, scoring="neg_mean_squared_error"
            )
            mse_without = float(-np.mean(cv_scores_mse))

            # Calculate importance score
            importance = (mse_without - baseline_mse) / baseline_mse
            percent_increase = importance * 100

            logger.info(f"    MSE without: {mse_without:.6f}")
            logger.info(f"    Importance: {percent_increase:.1f}% increase")

            # Determine criticality
            if importance > 0.5:
                criticality = "CRITICAL"
                priority = 1
            elif importance > 0.1:
                criticality = "IMPORTANT"
                priority = 2
            else:
                criticality = "REDUNDANT"
                priority = 3

            results_list.append({
                "receptor": receptor_to_drop,
                "mse_baseline": baseline_mse,
                "mse_without": mse_without,
                "importance_score": importance,
                "percent_increase": percent_increase,
                "criticality": criticality,
                "priority": priority,
            })

        # Create DataFrame and sort by importance
        df = pd.DataFrame(results_list)
        df = df.sort_values("importance_score", ascending=False)

        return df

    def create_priority_matrix(
        self,
        sensitivity_df: pd.DataFrame,
        lasso_weights: Dict[str, float],
        condition_name: str,
        all_conditions_receptors: Dict[str, List[str]] = None,
    ) -> pd.DataFrame:
        """
        Create priority matrix combining sensitivity, weights, and multi-condition counts.

        Args:
            sensitivity_df: Sensitivity analysis results
            lasso_weights: LASSO weights for each receptor
            condition_name: Current condition name
            all_conditions_receptors: Dict mapping condition -> list of receptors

        Returns:
            Priority matrix DataFrame
        """
        priority_data = []

        for _, row in sensitivity_df.iterrows():
            receptor = row["receptor"]

            # Get LASSO weight
            weight = lasso_weights.get(receptor, 0.0)

            # Count appearances in other conditions
            appears_in = 1  # At least in current condition
            if all_conditions_receptors:
                for cond, receptors in all_conditions_receptors.items():
                    if cond != condition_name and receptor in receptors:
                        appears_in += 1

            # Calculate composite score
            # 50% importance, 30% abs(weight), 20% multi-condition
            composite_score = (
                row["importance_score"] * 0.5 +
                abs(weight) * 0.3 +
                (appears_in / 5.0) * 0.2  # Normalize by max 5 conditions
            )

            # Determine experimental recommendation
            if composite_score > 0.6:
                recommendation = "Test FIRST! ⭐⭐⭐"
            elif composite_score > 0.3:
                recommendation = "Test SECOND ⭐⭐"
            else:
                recommendation = "Skip ✓"

            priority_data.append({
                "receptor": receptor,
                "importance_score": row["importance_score"],
                "lasso_weight": weight,
                "appears_in_conditions": appears_in,
                "composite_score": composite_score,
                "recommendation": recommendation,
            })

        priority_df = pd.DataFrame(priority_data)
        priority_df = priority_df.sort_values("composite_score", ascending=False)

        return priority_df

    def plot_importance_bar(
        self,
        sensitivity_df: pd.DataFrame,
        save_to: str,
        dpi: int = 300,
    ):
        """
        Create bar chart of importance scores.

        Args:
            sensitivity_df: Sensitivity results
            save_to: Path to save figure
            dpi: Resolution
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        receptors = sensitivity_df["receptor"].values
        importance = sensitivity_df["percent_increase"].values

        # Color by criticality
        colors = []
        for crit in sensitivity_df["criticality"]:
            if crit == "CRITICAL":
                colors.append("red")
            elif crit == "IMPORTANT":
                colors.append("orange")
            else:
                colors.append("gray")

        ax.barh(range(len(receptors)), importance, color=colors, alpha=0.7, edgecolor="k")

        ax.set_yticks(range(len(receptors)))
        ax.set_yticklabels(receptors)
        ax.set_xlabel("Importance Score (% MSE increase when removed)", fontsize=12)
        ax.set_ylabel("Receptor", fontsize=12)
        ax.set_title(
            "Sensitivity Analysis: Which Receptors Are Critical?\n"
            "Red = CRITICAL (>50%), Orange = IMPORTANT (10-50%), Gray = REDUNDANT (<10%)",
            fontsize=13,
        )
        ax.axvline(50, color="red", linestyle="--", linewidth=1, alpha=0.5, label="50% threshold")
        ax.axvline(10, color="orange", linestyle="--", linewidth=1, alpha=0.5, label="10% threshold")
        ax.grid(alpha=0.3, axis="x")
        ax.legend()

        plt.tight_layout()

        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved importance bar chart to {save_path}")
        plt.close()

    def plot_sensitivity_vs_weight(
        self,
        sensitivity_df: pd.DataFrame,
        lasso_weights: Dict[str, float],
        save_to: str,
        dpi: int = 300,
    ):
        """
        Create scatter plot of sensitivity vs LASSO weight.

        Args:
            sensitivity_df: Sensitivity results
            lasso_weights: LASSO weights
            save_to: Path to save figure
            dpi: Resolution
        """
        fig, ax = plt.subplots(figsize=(10, 7))

        receptors = sensitivity_df["receptor"].values
        importance = sensitivity_df["importance_score"].values
        weights = [abs(lasso_weights.get(r, 0.0)) for r in receptors]

        # Color by criticality
        colors = []
        for crit in sensitivity_df["criticality"]:
            if crit == "CRITICAL":
                colors.append("red")
            elif crit == "IMPORTANT":
                colors.append("orange")
            else:
                colors.append("gray")

        ax.scatter(weights, importance, c=colors, s=200, alpha=0.7, edgecolors="k", linewidths=1.5)

        # Annotate points
        for i, receptor in enumerate(receptors):
            ax.annotate(
                receptor,
                (weights[i], importance[i]),
                fontsize=10,
                xytext=(7, 7),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax.set_xlabel("LASSO Weight (absolute value)", fontsize=12)
        ax.set_ylabel("Sensitivity Score (% MSE increase)", fontsize=12)
        ax.set_title(
            "Sensitivity vs LASSO Weight\n"
            "Are high-weight receptors always important?",
            fontsize=14,
        )
        ax.axhline(0.5, color="red", linestyle="--", alpha=0.3, label="50% critical threshold")
        ax.axhline(0.1, color="orange", linestyle="--", alpha=0.3, label="10% important threshold")
        ax.grid(alpha=0.3)
        ax.legend()

        plt.tight_layout()

        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved sensitivity vs weight scatter to {save_path}")
        plt.close()

    def plot_priority_heatmap(
        self,
        priority_df: pd.DataFrame,
        save_to: str,
        dpi: int = 300,
    ):
        """
        Create heatmap of priority matrix.

        Args:
            priority_df: Priority matrix
            save_to: Path to save figure
            dpi: Resolution
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create matrix for heatmap
        receptors = priority_df["receptor"].values
        metrics = ["importance_score", "lasso_weight", "appears_in_conditions"]

        matrix = priority_df[metrics].values.T

        # Normalize each row to 0-1 for visualization
        matrix_norm = np.zeros_like(matrix, dtype=float)
        for i in range(matrix.shape[0]):
            row = matrix[i]
            if row.max() > row.min():
                matrix_norm[i] = (row - row.min()) / (row.max() - row.min())
            else:
                matrix_norm[i] = 0.5

        im = ax.imshow(matrix_norm, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

        ax.set_xticks(range(len(receptors)))
        ax.set_yticks(range(len(metrics)))
        ax.set_xticklabels(receptors, rotation=45, ha="right")
        ax.set_yticklabels(["Importance Score", "LASSO Weight", "Multi-Condition"])

        # Add text annotations
        for i in range(len(metrics)):
            for j in range(len(receptors)):
                text = ax.text(
                    j, i, f"{matrix[i, j]:.2f}",
                    ha="center", va="center", color="black", fontsize=10
                )

        ax.set_title("Receptor Priority Matrix\n(Normalized for visualization)", fontsize=14)
        plt.colorbar(im, ax=ax, label="Normalized Score")
        plt.tight_layout()

        save_path = Path(save_to)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        logger.info(f"Saved priority heatmap to {save_path}")
        plt.close()

    def generate_report(
        self,
        condition_name: str,
        sensitivity_df: pd.DataFrame,
        priority_df: pd.DataFrame,
        baseline_mse: float,
        baseline_r2: float,
        save_to: str,
    ):
        """
        Generate comprehensive text report.

        Args:
            condition_name: Condition name
            sensitivity_df: Sensitivity results
            priority_df: Priority matrix
            baseline_mse: Baseline MSE
            baseline_r2: Baseline R²
            save_to: Path to save report
        """
        lines = [
            "=" * 80,
            "RECEPTOR SENSITIVITY ANALYSIS REPORT",
            "=" * 80,
            f"Condition: {condition_name}",
            f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "=" * 80,
            "EXECUTIVE SUMMARY",
            "=" * 80,
            "",
            "This sensitivity analysis identifies which olfactory receptors are CRITICAL",
            "vs REDUNDANT for predicting learned behavioral responses (PER).",
            "",
            "Method:",
            "  1. Fit baseline LASSO model with all sparse receptors",
            "  2. Drop each receptor one-by-one and re-fit model",
            "  3. Measure MSE increase (importance score)",
            "  4. Rank receptors by criticality",
            "",
            "Interpretation:",
            "  - Importance > 50%: CRITICAL (test with optogenetics FIRST)",
            "  - Importance 10-50%: IMPORTANT (secondary experimental target)",
            "  - Importance < 10%: REDUNDANT (can skip in experiments)",
            "",
            "=" * 80,
            "BASELINE MODEL PERFORMANCE",
            "=" * 80,
            f"  Baseline MSE: {baseline_mse:.6f}",
            f"  Baseline R²: {baseline_r2:.6f}",
            f"  Receptors used: {len(sensitivity_df)}",
            "",
            "=" * 80,
            "SENSITIVITY ANALYSIS RESULTS",
            "=" * 80,
            "",
        ]

        # Add sensitivity table
        lines.append(f"{'Receptor':<15} | {'Importance':<12} | {'MSE Change':<12} | {'Criticality':<12}")
        lines.append("-" * 80)

        for _, row in sensitivity_df.iterrows():
            lines.append(
                f"{row['receptor']:<15} | "
                f"{row['percent_increase']:>10.1f}% | "
                f"{row['mse_without']:>10.6f} | "
                f"{row['criticality']:<12}"
            )

        lines.append("")
        lines.append("=" * 80)
        lines.append("PRIORITY MATRIX FOR EXPERIMENTAL VALIDATION")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"{'Receptor':<12} | {'Importance':<12} | {'LASSO Wt':<10} | {'Multi-Cond':<10} | {'Score':<8} | {'Recommendation':<20}")
        lines.append("-" * 80)

        for _, row in priority_df.iterrows():
            lines.append(
                f"{row['receptor']:<12} | "
                f"{row['importance_score']:>10.2f} | "
                f"{row['lasso_weight']:>10.4f} | "
                f"{row['appears_in_conditions']:>10d} | "
                f"{row['composite_score']:>8.2f} | "
                f"{row['recommendation']:<20}"
            )

        lines.append("")
        lines.append("=" * 80)
        lines.append("INTERPRETATION & RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")

        # Find top receptor
        top_receptor = sensitivity_df.iloc[0]
        lines.append(f"1. MOST CRITICAL RECEPTOR: {top_receptor['receptor']}")
        lines.append(f"   - Importance: {top_receptor['percent_increase']:.1f}% MSE increase when removed")
        lines.append(f"   - Impact: {top_receptor['criticality']}")
        lines.append(f"   - Recommendation: This receptor should be tested FIRST with optogenetics")
        lines.append("")

        # Count criticality levels
        critical_count = len(sensitivity_df[sensitivity_df["criticality"] == "CRITICAL"])
        important_count = len(sensitivity_df[sensitivity_df["criticality"] == "IMPORTANT"])
        redundant_count = len(sensitivity_df[sensitivity_df["criticality"] == "REDUNDANT"])

        lines.append(f"2. RECEPTOR BREAKDOWN:")
        lines.append(f"   - CRITICAL: {critical_count} receptor(s)")
        lines.append(f"   - IMPORTANT: {important_count} receptor(s)")
        lines.append(f"   - REDUNDANT: {redundant_count} receptor(s)")
        lines.append("")

        lines.append("3. EXPERIMENTAL VALIDATION PRIORITY:")
        for i, (_, row) in enumerate(priority_df.head(3).iterrows(), 1):
            lines.append(f"   {i}. {row['receptor']} (composite score: {row['composite_score']:.2f})")
        lines.append("")

        lines.append("4. CAVEATS & LIMITATIONS:")
        lines.append("   - Small sample size (n=7 odorants) means estimates may be unstable")
        lines.append("   - Correlation ≠ causation: Silencing experiments are needed")
        lines.append("   - Some receptors may show collinearity (shared information)")
        lines.append("   - Model assumes linear combination of receptor activations")
        lines.append("")

        lines.append("5. NEXT STEPS:")
        lines.append("   - Design optogenetic silencing experiments for top 2-3 receptors")
        lines.append("   - Map identified receptors to FlyWire connectome")
        lines.append("   - Compare with known receptor-behavior literature")
        lines.append("   - Validate predictions with independent behavioral assays")
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

        logger.info(f"Saved sensitivity analysis report to {save_path}")

        return report_text

    def run_analysis(
        self,
        condition_name: str = "opto_hex",
        output_dir: str = None,
    ):
        """
        Run complete sensitivity analysis workflow.

        Args:
            condition_name: Condition to analyze (default: "opto_hex")
            output_dir: Output directory (default: behavioral_prediction_results)
        """
        if output_dir is None:
            output_dir = self.results_dir
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("RECEPTOR SENSITIVITY ANALYSIS")
        logger.info("=" * 80)
        logger.info(f"Condition: {condition_name}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("")

        # Step 1: Load model results
        logger.info("Step 1: Loading LASSO model results...")
        model_results = self.load_model_results(condition_name)

        sparse_receptors = list(model_results["lasso_weights"].keys())
        lambda_alpha = model_results["lambda_value"]
        trained_odorant = model_results["trained_odorant"]

        # Step 2: Prepare feature matrix
        logger.info("\nStep 2: Preparing feature matrix...")
        X, y, test_odorants, receptor_names = self.prepare_feature_matrix(
            condition_name=condition_name,
            trained_odorant=trained_odorant,
        )

        # Step 3: Fit baseline model
        logger.info("\nStep 3: Fitting baseline model...")
        receptor_name_to_idx = {name: i for i, name in enumerate(receptor_names)}
        sparse_indices = [receptor_name_to_idx[r] for r in sparse_receptors]

        baseline_mse, baseline_r2, y_pred = self.fit_baseline_model(
            X=X,
            y=y,
            receptor_indices=sparse_indices,
            lambda_alpha=lambda_alpha,
        )

        # Step 4: Sensitivity loop
        logger.info("\nStep 4: Running sensitivity analysis...")
        sensitivity_df = self.sensitivity_loop(
            X=X,
            y=y,
            receptor_names=receptor_names,
            sparse_receptors=sparse_receptors,
            lambda_alpha=lambda_alpha,
            baseline_mse=baseline_mse,
        )

        # Step 5: Create priority matrix
        logger.info("\nStep 5: Creating priority matrix...")
        priority_df = self.create_priority_matrix(
            sensitivity_df=sensitivity_df,
            lasso_weights=model_results["lasso_weights"],
            condition_name=condition_name,
        )

        # Step 6: Generate visualizations
        logger.info("\nStep 6: Generating visualizations...")
        self.plot_importance_bar(
            sensitivity_df=sensitivity_df,
            save_to=str(output_dir / f"{condition_name}_sensitivity_importance.png"),
        )

        self.plot_sensitivity_vs_weight(
            sensitivity_df=sensitivity_df,
            lasso_weights=model_results["lasso_weights"],
            save_to=str(output_dir / f"{condition_name}_sensitivity_vs_weight.png"),
        )

        self.plot_priority_heatmap(
            priority_df=priority_df,
            save_to=str(output_dir / f"{condition_name}_priority_heatmap.png"),
        )

        # Step 7: Export results
        logger.info("\nStep 7: Exporting results...")
        sensitivity_df.to_csv(
            output_dir / f"{condition_name}_sensitivity_results.csv", index=False
        )
        priority_df.to_csv(
            output_dir / f"{condition_name}_priority_matrix.csv", index=False
        )

        # Step 8: Generate report
        logger.info("\nStep 8: Generating report...")
        report_text = self.generate_report(
            condition_name=condition_name,
            sensitivity_df=sensitivity_df,
            priority_df=priority_df,
            baseline_mse=baseline_mse,
            baseline_r2=baseline_r2,
            save_to=str(output_dir / f"{condition_name}_sensitivity_report.txt"),
        )

        logger.info("")
        logger.info("=" * 80)
        logger.info("ANALYSIS COMPLETE!")
        logger.info("=" * 80)
        logger.info(f"\nGenerated files in {output_dir}:")
        logger.info(f"  - {condition_name}_sensitivity_results.csv")
        logger.info(f"  - {condition_name}_priority_matrix.csv")
        logger.info(f"  - {condition_name}_sensitivity_importance.png")
        logger.info(f"  - {condition_name}_sensitivity_vs_weight.png")
        logger.info(f"  - {condition_name}_priority_heatmap.png")
        logger.info(f"  - {condition_name}_sensitivity_report.txt")
        logger.info("")

        # Print summary
        print("\n" + "=" * 80)
        print("QUICK SUMMARY")
        print("=" * 80)
        print(f"\nTop 3 Receptors to Test:")
        for i, (_, row) in enumerate(priority_df.head(3).iterrows(), 1):
            print(f"  {i}. {row['receptor']:<12} - {row['recommendation']}")
        print("\n" + "=" * 80)


def main():
    """Run sensitivity analysis for opto_hex condition."""

    # Paths
    DOORCACHE_PATH = "door_cache"
    BEHAVIOR_CSV_PATH = (
        "/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions(Strictest)/"
        "reaction_rates_summary_unordered.csv"
    )
    OUTPUT_DIR = "behavioral_prediction_results"

    # Initialize analyzer
    analyzer = ReceptorSensitivityAnalyzer(
        doorcache_path=DOORCACHE_PATH,
        behavior_csv_path=BEHAVIOR_CSV_PATH,
        results_dir=OUTPUT_DIR,
    )

    # Run analysis for opto_hex
    analyzer.run_analysis(
        condition_name="opto_hex",
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
