"""
Diagnostic Analysis for Negative Importance Scores
===================================================

This script investigates why sensitivity analysis shows negative importance scores
and identifies the optimal sparse receptor circuit.

Diagnostic Tests:
1. Lambda sweep: Test LASSO with different regularization strengths
2. Receptor correlation analysis: Check for redundancy
3. Simpler model evaluation: Test 1, 2, 3 receptor models
4. Cross-validation comparison: Compare model performance

Example usage:
    python examples/receptor_sensitivity_diagnostics.py
"""

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")

from door_toolkit.pathways import LassoBehavioralPredictor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SensitivityDiagnostics:
    """Diagnostic tools for understanding negative importance scores."""

    def __init__(
        self,
        doorcache_path: str,
        behavior_csv_path: str,
    ):
        """Initialize diagnostics."""
        self.predictor = LassoBehavioralPredictor(
            doorcache_path=doorcache_path,
            behavior_csv_path=behavior_csv_path,
            scale_features=True,
            scale_targets=False,
        )

        logger.info("Initialized SensitivityDiagnostics")

    def lambda_sweep_analysis(
        self,
        condition_name: str,
        lambda_values: list = None,
        output_dir: str = "behavioral_prediction_results",
    ):
        """
        Test LASSO with different lambda values to find optimal sparsity.

        Args:
            condition_name: Condition name
            lambda_values: List of lambda values to test
            output_dir: Output directory
        """
        if lambda_values is None:
            lambda_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("LAMBDA SWEEP ANALYSIS")
        logger.info("=" * 80)

        results = []

        for lambda_val in lambda_values:
            logger.info(f"\nTesting λ = {lambda_val:.3f}")

            # Fit model
            model_result = self.predictor.fit_behavior(
                condition_name=condition_name,
                lambda_range=[lambda_val],
                prediction_mode="test_odorant",
            )

            results.append({
                "lambda": lambda_val,
                "n_receptors": model_result.n_receptors_selected,
                "cv_mse": model_result.cv_mse,
                "cv_r2": model_result.cv_r2_score,
                "top_receptor": (
                    model_result.get_top_receptors(1)[0][0]
                    if model_result.n_receptors_selected > 0
                    else "None"
                ),
                "top_weight": (
                    model_result.get_top_receptors(1)[0][1]
                    if model_result.n_receptors_selected > 0
                    else 0.0
                ),
            })

            logger.info(f"  Receptors selected: {model_result.n_receptors_selected}")
            logger.info(f"  CV MSE: {model_result.cv_mse:.6f}")

        # Create DataFrame
        df = pd.DataFrame(results)
        df.to_csv(output_path / f"{condition_name}_lambda_sweep.csv", index=False)

        # Plot results
        self._plot_lambda_sweep(df, output_path / f"{condition_name}_lambda_sweep.png")

        # Find optimal lambda
        optimal_idx = df["cv_mse"].idxmin()
        optimal_lambda = df.loc[optimal_idx, "lambda"]
        optimal_n_receptors = df.loc[optimal_idx, "n_receptors"]

        logger.info("\n" + "=" * 80)
        logger.info("OPTIMAL LAMBDA")
        logger.info("=" * 80)
        logger.info(f"  λ = {optimal_lambda:.3f}")
        logger.info(f"  Receptors selected: {optimal_n_receptors}")
        logger.info(f"  CV MSE: {df.loc[optimal_idx, 'cv_mse']:.6f}")
        logger.info("")

        return df

    def _plot_lambda_sweep(self, df: pd.DataFrame, save_to: str):
        """Plot lambda sweep results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: MSE vs Lambda
        ax1 = axes[0]
        ax1.plot(df["lambda"], df["cv_mse"], "o-", linewidth=2, markersize=8)
        ax1.set_xlabel("Lambda (α)", fontsize=12)
        ax1.set_ylabel("Cross-Validated MSE", fontsize=12)
        ax1.set_title("LASSO Regularization Strength vs Prediction Error", fontsize=13)
        ax1.set_xscale("log")
        ax1.grid(alpha=0.3)

        # Mark optimal
        optimal_idx = df["cv_mse"].idxmin()
        ax1.axvline(
            df.loc[optimal_idx, "lambda"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Optimal λ = {df.loc[optimal_idx, 'lambda']:.3f}",
        )
        ax1.legend()

        # Plot 2: Number of receptors vs Lambda
        ax2 = axes[1]
        ax2.plot(df["lambda"], df["n_receptors"], "o-", linewidth=2, markersize=8, color="orange")
        ax2.set_xlabel("Lambda (α)", fontsize=12)
        ax2.set_ylabel("Number of Receptors Selected", fontsize=12)
        ax2.set_title("Model Sparsity vs Regularization Strength", fontsize=13)
        ax2.set_xscale("log")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
        logger.info(f"Saved lambda sweep plot to {save_to}")
        plt.close()

    def receptor_correlation_analysis(
        self,
        condition_name: str,
        receptors_to_check: list,
        output_dir: str = "behavioral_prediction_results",
    ):
        """
        Analyze correlations between selected receptors.

        Args:
            condition_name: Condition name
            receptors_to_check: List of receptor names
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("RECEPTOR CORRELATION ANALYSIS")
        logger.info("=" * 80)

        # Get feature matrix
        model_result = self.predictor.fit_behavior(
            condition_name=condition_name,
            lambda_range=[0.1],
            prediction_mode="test_odorant",
        )

        X = model_result.feature_matrix
        receptor_names = model_result.receptor_names

        # Get indices for receptors to check
        receptor_indices = [
            receptor_names.index(r) for r in receptors_to_check
            if r in receptor_names
        ]

        # Extract features for these receptors
        X_subset = X[:, receptor_indices]

        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X_subset.T)

        # Create DataFrame
        corr_df = pd.DataFrame(
            corr_matrix,
            index=receptors_to_check,
            columns=receptors_to_check,
        )

        # Save
        corr_df.to_csv(output_path / f"{condition_name}_receptor_correlations.csv")

        # Plot heatmap
        self._plot_correlation_heatmap(
            corr_df,
            output_path / f"{condition_name}_receptor_correlations.png",
        )

        # Report high correlations
        logger.info("\nPairwise Correlations:")
        for i, r1 in enumerate(receptors_to_check):
            for j, r2 in enumerate(receptors_to_check):
                if i < j:
                    corr_val = corr_matrix[i, j]
                    logger.info(f"  {r1} vs {r2}: r = {corr_val:.3f}")
                    if abs(corr_val) > 0.7:
                        logger.warning(f"    ⚠️ HIGH CORRELATION detected!")

        return corr_df

    def _plot_correlation_heatmap(self, corr_df: pd.DataFrame, save_to: str):
        """Plot correlation heatmap."""
        fig, ax = plt.subplots(figsize=(8, 6))

        im = ax.imshow(corr_df.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")

        # Set ticks
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df.index)))
        ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
        ax.set_yticklabels(corr_df.index)

        # Add text annotations
        for i in range(len(corr_df.index)):
            for j in range(len(corr_df.columns)):
                text = ax.text(
                    j, i, f"{corr_df.values[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if abs(corr_df.values[i, j]) > 0.5 else "black",
                    fontsize=11,
                )

        ax.set_title(
            "Receptor Response Correlations Across Test Odorants\n"
            "(High correlation → Redundancy)",
            fontsize=13,
        )
        plt.colorbar(im, ax=ax, label="Pearson Correlation")
        plt.tight_layout()
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
        logger.info(f"Saved correlation heatmap to {save_to}")
        plt.close()

    def simple_model_comparison(
        self,
        condition_name: str,
        receptors_ranked: list,
        output_dir: str = "behavioral_prediction_results",
    ):
        """
        Test simpler models with 1, 2, 3 receptors.

        Args:
            condition_name: Condition name
            receptors_ranked: List of receptors in priority order
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("SIMPLE MODEL COMPARISON")
        logger.info("=" * 80)

        # Get feature matrix
        model_result = self.predictor.fit_behavior(
            condition_name=condition_name,
            lambda_range=[0.1],
            prediction_mode="test_odorant",
        )

        X = model_result.feature_matrix
        y = model_result.actual_per
        receptor_names = model_result.receptor_names

        results = []

        # Test models with increasing complexity
        for n_receptors in range(1, min(len(receptors_ranked) + 1, 5)):
            selected_receptors = receptors_ranked[:n_receptors]

            logger.info(f"\nTesting {n_receptors}-receptor model: {selected_receptors}")

            # Get indices
            indices = [
                receptor_names.index(r) for r in selected_receptors
                if r in receptor_names
            ]

            X_subset = X[:, indices]

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_subset)

            # Fit LASSO with LOO-CV
            lasso = Lasso(alpha=0.1, max_iter=10000, random_state=42)

            loo = LeaveOneOut()
            cv_scores_mse = cross_val_score(
                lasso, X_scaled, y, cv=loo, scoring="neg_mean_squared_error"
            )
            cv_mse = float(-np.mean(cv_scores_mse))

            # Fit on full data
            lasso.fit(X_scaled, y)

            logger.info(f"  CV MSE: {cv_mse:.6f}")
            logger.info(f"  Coefficients: {lasso.coef_}")

            results.append({
                "n_receptors": n_receptors,
                "receptors": ", ".join(selected_receptors),
                "cv_mse": cv_mse,
            })

        # Create DataFrame
        df = pd.DataFrame(results)
        df.to_csv(output_path / f"{condition_name}_simple_model_comparison.csv", index=False)

        # Plot comparison
        self._plot_simple_model_comparison(
            df,
            output_path / f"{condition_name}_simple_model_comparison.png",
        )

        # Find best model
        best_idx = df["cv_mse"].idxmin()
        logger.info("\n" + "=" * 80)
        logger.info("BEST SIMPLE MODEL")
        logger.info("=" * 80)
        logger.info(f"  Number of receptors: {df.loc[best_idx, 'n_receptors']}")
        logger.info(f"  Receptors: {df.loc[best_idx, 'receptors']}")
        logger.info(f"  CV MSE: {df.loc[best_idx, 'cv_mse']:.6f}")
        logger.info("")

        return df

    def _plot_simple_model_comparison(self, df: pd.DataFrame, save_to: str):
        """Plot simple model comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(df["n_receptors"], df["cv_mse"], "o-", linewidth=2, markersize=10)

        ax.set_xlabel("Number of Receptors", fontsize=12)
        ax.set_ylabel("Cross-Validated MSE", fontsize=12)
        ax.set_title(
            "Model Complexity vs Prediction Error\n"
            "Lower MSE = Better Prediction",
            fontsize=13,
        )
        ax.grid(alpha=0.3)

        # Mark best model
        best_idx = df["cv_mse"].idxmin()
        ax.axvline(
            df.loc[best_idx, "n_receptors"],
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Best: {df.loc[best_idx, 'n_receptors']} receptor(s)",
        )

        # Annotate points with receptor names
        for i, row in df.iterrows():
            ax.annotate(
                row["receptors"],
                (row["n_receptors"], row["cv_mse"]),
                fontsize=9,
                xytext=(10, 5),
                textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7),
            )

        ax.legend()
        plt.tight_layout()
        plt.savefig(save_to, dpi=300, bbox_inches="tight")
        logger.info(f"Saved simple model comparison to {save_to}")
        plt.close()

    def run_full_diagnostics(
        self,
        condition_name: str = "opto_hex",
        sparse_receptors: list = None,
        output_dir: str = "behavioral_prediction_results",
    ):
        """Run all diagnostic tests."""

        if sparse_receptors is None:
            # Default from opto_hex LASSO results
            sparse_receptors = ["Or22b", "Or49a", "Or23a", "Or13a"]

        logger.info("=" * 80)
        logger.info("COMPREHENSIVE SENSITIVITY DIAGNOSTICS")
        logger.info("=" * 80)
        logger.info(f"Condition: {condition_name}")
        logger.info(f"Receptors to analyze: {sparse_receptors}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("")

        # Test 1: Lambda sweep
        logger.info("\n[1/3] Running lambda sweep analysis...")
        lambda_df = self.lambda_sweep_analysis(
            condition_name=condition_name,
            output_dir=output_dir,
        )

        # Test 2: Receptor correlations
        logger.info("\n[2/3] Running receptor correlation analysis...")
        corr_df = self.receptor_correlation_analysis(
            condition_name=condition_name,
            receptors_to_check=sparse_receptors,
            output_dir=output_dir,
        )

        # Test 3: Simple model comparison
        logger.info("\n[3/3] Running simple model comparison...")
        comparison_df = self.simple_model_comparison(
            condition_name=condition_name,
            receptors_ranked=sparse_receptors,  # Ranked by LASSO weight
            output_dir=output_dir,
        )

        # Generate summary report
        logger.info("\nGenerating diagnostic summary report...")
        self._generate_diagnostic_report(
            condition_name=condition_name,
            lambda_df=lambda_df,
            corr_df=corr_df,
            comparison_df=comparison_df,
            output_dir=output_dir,
        )

        logger.info("\n" + "=" * 80)
        logger.info("DIAGNOSTICS COMPLETE!")
        logger.info("=" * 80)

    def _generate_diagnostic_report(
        self,
        condition_name: str,
        lambda_df: pd.DataFrame,
        corr_df: pd.DataFrame,
        comparison_df: pd.DataFrame,
        output_dir: str,
    ):
        """Generate diagnostic summary report."""
        lines = [
            "=" * 80,
            "SENSITIVITY DIAGNOSTICS SUMMARY REPORT",
            "=" * 80,
            f"Condition: {condition_name}",
            f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "This report investigates why sensitivity analysis showed negative importance",
            "scores and provides recommendations for refining the sparse receptor circuit.",
            "",
            "=" * 80,
            "TEST 1: LAMBDA SWEEP ANALYSIS",
            "=" * 80,
            "",
            "Question: Is λ=0.1 optimal, or should we use stronger regularization?",
            "",
        ]

        # Lambda sweep summary
        optimal_idx = lambda_df["cv_mse"].idxmin()
        lines.append(f"Optimal λ: {lambda_df.loc[optimal_idx, 'lambda']:.3f}")
        lines.append(f"Receptors at optimal λ: {lambda_df.loc[optimal_idx, 'n_receptors']}")
        lines.append(f"MSE at optimal λ: {lambda_df.loc[optimal_idx, 'cv_mse']:.6f}")
        lines.append("")

        if lambda_df.loc[optimal_idx, "lambda"] > 0.1:
            lines.append("✅ FINDING: Stronger regularization (higher λ) improves prediction")
            lines.append("   → Current λ=0.1 is TOO WEAK, model is overfitted")
            lines.append(f"   → Recommended λ = {lambda_df.loc[optimal_idx, 'lambda']:.3f}")
        else:
            lines.append("⚠️ FINDING: Current λ=0.1 is already optimal")
            lines.append("   → Overfitting is not primarily due to λ choice")

        lines.append("")
        lines.append("=" * 80)
        lines.append("TEST 2: RECEPTOR CORRELATION ANALYSIS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Question: Are receptors redundant (highly correlated)?")
        lines.append("")

        # Check for high correlations
        high_corr_found = False
        for i in range(len(corr_df.index)):
            for j in range(i + 1, len(corr_df.columns)):
                r1 = corr_df.index[i]
                r2 = corr_df.columns[j]
                corr_val = corr_df.values[i, j]
                if abs(corr_val) > 0.7:
                    lines.append(f"⚠️ HIGH CORRELATION: {r1} vs {r2} (r = {corr_val:.3f})")
                    high_corr_found = True

        if high_corr_found:
            lines.append("")
            lines.append("✅ FINDING: Receptors show redundancy (r > 0.7)")
            lines.append("   → This explains negative importance scores")
            lines.append("   → Removing one receptor doesn't hurt (other picks up slack)")
        else:
            lines.append("✅ FINDING: No high correlations detected (all r < 0.7)")
            lines.append("   → Redundancy is NOT the primary cause")

        lines.append("")
        lines.append("=" * 80)
        lines.append("TEST 3: SIMPLE MODEL COMPARISON")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Question: Would a simpler model (1-2 receptors) perform better?")
        lines.append("")

        # Find best simple model
        best_idx = comparison_df["cv_mse"].idxmin()
        lines.append(f"Best model: {comparison_df.loc[best_idx, 'n_receptors']} receptor(s)")
        lines.append(f"Receptors: {comparison_df.loc[best_idx, 'receptors']}")
        lines.append(f"MSE: {comparison_df.loc[best_idx, 'cv_mse']:.6f}")
        lines.append("")

        if comparison_df.loc[best_idx, "n_receptors"] < 4:
            lines.append("✅ FINDING: Simpler model outperforms 4-receptor model")
            lines.append("   → Current LASSO selected too many receptors")
            lines.append(f"   → Use {comparison_df.loc[best_idx, 'receptors']} for experiments")
        else:
            lines.append("⚠️ FINDING: 4-receptor model is already optimal")

        lines.append("")
        lines.append("=" * 80)
        lines.append("OVERALL CONCLUSIONS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Why are importance scores negative?")

        # Synthesize findings
        if lambda_df.loc[optimal_idx, "lambda"] > 0.1:
            lines.append("1. ✅ Lambda too low (overfitting)")
        if high_corr_found:
            lines.append("2. ✅ Receptor redundancy detected")
        if comparison_df.loc[best_idx, "n_receptors"] < 4:
            lines.append("3. ✅ Model too complex (simpler is better)")

        lines.append("")
        lines.append("=" * 80)
        lines.append("REFINED EXPERIMENTAL RECOMMENDATIONS")
        lines.append("=" * 80)
        lines.append("")
        lines.append("Based on diagnostic tests:")
        lines.append(f"1. Use λ = {lambda_df.loc[optimal_idx, 'lambda']:.3f} (not 0.1)")
        lines.append(f"2. Test {comparison_df.loc[best_idx, 'receptors']} circuit")
        lines.append("3. Design combination silencing experiments (not single receptors)")
        lines.append("4. Collect more behavioral data (current n=7 is too small)")
        lines.append("")
        lines.append("=" * 80)

        # Save report
        report_path = Path(output_dir) / f"{condition_name}_diagnostic_report.txt"
        with open(report_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Saved diagnostic report to {report_path}")


def main():
    """Run full diagnostic analysis."""

    DOORCACHE_PATH = "door_cache"
    BEHAVIOR_CSV_PATH = (
        "/home/ramanlab/Documents/cole/Results/Opto/Reaction_Predictions(Strictest)/"
        "reaction_rates_summary_unordered.csv"
    )
    OUTPUT_DIR = "behavioral_prediction_results"

    diagnostics = SensitivityDiagnostics(
        doorcache_path=DOORCACHE_PATH,
        behavior_csv_path=BEHAVIOR_CSV_PATH,
    )

    # Run full diagnostics
    diagnostics.run_full_diagnostics(
        condition_name="opto_hex",
        sparse_receptors=["Or22b", "Or49a", "Or23a", "Or13a"],  # From LASSO
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
