"""
Behavioral Prediction Module
=============================

Predict behavioral responses from receptor activation patterns.

This module provides two approaches:
1. Heuristic prediction based on known receptor-behavior associations
2. LASSO regression from optogenetic behavioral data to identify sparse receptor circuits

Classes:
    BehavioralPredictor: Heuristic-based prediction using literature weights
    LassoBehavioralPredictor: LASSO regression-based prediction from behavioral data
    BehaviorModelResults: Container for LASSO model results with visualization
"""

import json
import logging
import warnings
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")  # Non-interactive backend for headless environments

# Suppress sklearn warnings for R² with small samples (expected with LOO-CV)
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from door_toolkit.encoder import DoOREncoder
from door_toolkit.utils import load_response_matrix

logger = logging.getLogger(__name__)


# ============================================================================
# Heuristic Behavioral Predictor (Original)
# ============================================================================


@dataclass
class BehaviorPrediction:
    """
    Predicted behavioral response to an odorant.

    Attributes:
        odorant_name: Name of the odorant
        predicted_valence: Predicted valence (attractive/aversive/neutral)
        confidence: Confidence score (0-1)
        receptor_pattern: Activation pattern of receptors
        key_contributors: Top contributing receptors
    """

    odorant_name: str
    predicted_valence: str
    confidence: float
    receptor_pattern: Dict[str, float]
    key_contributors: List[Tuple[str, float]]

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "odorant_name": self.odorant_name,
            "predicted_valence": self.predicted_valence,
            "confidence": self.confidence,
            "receptor_pattern": self.receptor_pattern,
            "key_contributors": [
                {"receptor": r, "contribution": c} for r, c in self.key_contributors
            ],
        }


class BehavioralPredictor:
    """
    Predict behavioral responses from receptor activation patterns.

    This class uses heuristic rules based on known receptor-behavior
    relationships to predict behavioral outcomes.

    Attributes:
        encoder: DoOREncoder instance
        response_matrix: DoOR response matrix

    Example:
        >>> predictor = BehavioralPredictor("door_cache")
        >>> prediction = predictor.predict_behavior("1-hexanol")
        >>> print(f"Valence: {prediction.predicted_valence}")
        >>> print(f"Confidence: {prediction.confidence:.2%}")
    """

    # Known receptor-behavior associations from literature
    ATTRACTIVE_RECEPTORS = {
        "Or42b": 0.9,  # Fruit esters - highly attractive
        "Or47b": 0.9,  # Hexanol - feeding attractive
        "Or59b": 0.7,  # Citrus - attractive
        "Or22a": 0.7,  # Fruit volatiles
        "Or42a": 0.6,  # Esters
    }

    AVERSIVE_RECEPTORS = {
        "Or92a": 0.9,  # Geosmin - highly aversive
        "Or7a": 0.7,  # CO2 - aversive
        "Or56a": 0.6,  # Fatty acids - aversive
        "Or69a": 0.6,  # Aversive odorants
    }

    FEEDING_RECEPTORS = {
        "Or47b": 0.9,  # Hexanol - feeding
        "Or42b": 0.7,  # Fruit - feeding
        "Or59b": 0.6,  # Sweet fruit
    }

    def __init__(self, door_cache_path: str):
        """
        Initialize behavioral predictor.

        Args:
            door_cache_path: Path to DoOR cache directory

        Raises:
            FileNotFoundError: If cache not found
        """
        self.door_cache_path = Path(door_cache_path)
        if not self.door_cache_path.exists():
            raise FileNotFoundError(f"DoOR cache not found: {self.door_cache_path}")

        self.encoder = DoOREncoder(str(self.door_cache_path), use_torch=False)
        self.response_matrix = load_response_matrix(str(self.door_cache_path))

        logger.info("Initialized BehavioralPredictor")

    def predict_behavior(self, odorant: str, threshold: float = 0.3) -> BehaviorPrediction:
        """
        Predict behavioral response to an odorant.

        Args:
            odorant: Odorant name
            threshold: Minimum receptor activation threshold

        Returns:
            BehaviorPrediction with valence and confidence

        Example:
            >>> predictor = BehavioralPredictor("door_cache")
            >>> pred = predictor.predict_behavior("ethyl butyrate")
            >>> print(f"{pred.odorant_name}: {pred.predicted_valence}")
        """
        logger.debug(f"Predicting behavior for {odorant}")

        # Encode odorant
        try:
            response_vector = self.encoder.encode(odorant)
        except Exception as e:
            raise ValueError(f"Could not encode odorant '{odorant}': {e}")

        # Build receptor pattern
        receptor_pattern = {}
        for i, receptor in enumerate(self.encoder.receptor_names):
            response = float(response_vector[i])
            if not np.isnan(response) and abs(response) >= threshold:
                receptor_pattern[receptor] = response

        # Calculate valence scores
        attractive_score = 0.0
        aversive_score = 0.0
        feeding_score = 0.0

        for receptor, response in receptor_pattern.items():
            # Attractive
            if receptor in self.ATTRACTIVE_RECEPTORS:
                weight = self.ATTRACTIVE_RECEPTORS[receptor]
                attractive_score += abs(response) * weight

            # Aversive
            if receptor in self.AVERSIVE_RECEPTORS:
                weight = self.AVERSIVE_RECEPTORS[receptor]
                aversive_score += abs(response) * weight

            # Feeding
            if receptor in self.FEEDING_RECEPTORS:
                weight = self.FEEDING_RECEPTORS[receptor]
                feeding_score += abs(response) * weight

        # Determine valence
        total_score = attractive_score + aversive_score + feeding_score

        if total_score == 0:
            predicted_valence = "neutral"
            confidence = 0.3
        elif attractive_score > aversive_score:
            predicted_valence = "attractive"
            confidence = min(attractive_score / (total_score + 1e-6), 1.0)
        elif aversive_score > attractive_score:
            predicted_valence = "aversive"
            confidence = min(aversive_score / (total_score + 1e-6), 1.0)
        else:
            predicted_valence = "neutral"
            confidence = 0.5

        # Add feeding annotation if high feeding score
        if feeding_score > 0.5:
            predicted_valence = f"{predicted_valence} (feeding)"

        # Find key contributors
        key_contributors = sorted(receptor_pattern.items(), key=lambda x: abs(x[1]), reverse=True)[
            :5
        ]

        prediction = BehaviorPrediction(
            odorant_name=odorant,
            predicted_valence=predicted_valence,
            confidence=float(confidence),
            receptor_pattern=receptor_pattern,
            key_contributors=key_contributors,
        )

        logger.info(f"Predicted {odorant}: {predicted_valence} " f"(confidence: {confidence:.2%})")

        return prediction

    def predict_batch(
        self, odorants: List[str], threshold: float = 0.3
    ) -> List[BehaviorPrediction]:
        """
        Predict behavior for multiple odorants.

        Args:
            odorants: List of odorant names
            threshold: Minimum receptor activation threshold

        Returns:
            List of BehaviorPrediction objects

        Example:
            >>> predictor = BehavioralPredictor("door_cache")
            >>> odorants = ["hexanol", "geosmin", "ethyl butyrate"]
            >>> predictions = predictor.predict_batch(odorants)
            >>> for pred in predictions:
            ...     print(f"{pred.odorant_name}: {pred.predicted_valence}")
        """
        predictions = []
        for odorant in odorants:
            try:
                prediction = self.predict_behavior(odorant, threshold)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"Could not predict behavior for {odorant}: {e}")
                continue

        return predictions


# ============================================================================
# LASSO Regression-Based Behavioral Predictor
# ============================================================================


@dataclass
class BehaviorModelResults:
    """
    Results from LASSO behavioral prediction model.

    Attributes:
        condition_name: Name of the optogenetic condition (e.g., "opto_hex")
        trained_odorant: Odorant used for training
        trained_odorant_door: DoOR name for trained odorant
        lasso_weights: Dictionary of {receptor: weight} for non-zero coefficients
        cv_r2_score: Cross-validated R² score
        cv_mse: Cross-validated mean squared error
        lambda_value: Selected LASSO regularization parameter
        n_receptors_selected: Number of non-zero receptors
        test_odorants: List of test odorants used
        actual_per: Actual PER values
        predicted_per: Predicted PER values
        receptor_coverage: Number of receptors with data for trained odorant
    """

    condition_name: str
    trained_odorant: str
    trained_odorant_door: str
    lasso_weights: Dict[str, float]
    cv_r2_score: float
    cv_mse: float
    lambda_value: float
    n_receptors_selected: int
    test_odorants: List[str]
    actual_per: np.ndarray
    predicted_per: np.ndarray
    receptor_coverage: int
    feature_matrix: Optional[np.ndarray] = None
    receptor_names: List[str] = field(default_factory=list)

    def get_top_receptors(self, n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N receptors by absolute LASSO weight.

        Args:
            n: Number of top receptors to return

        Returns:
            List of (receptor_name, weight) tuples sorted by absolute weight
        """
        sorted_weights = sorted(
            self.lasso_weights.items(), key=lambda x: abs(x[1]), reverse=True
        )
        return sorted_weights[:n]

    def plot_predictions(
        self, save_to: Optional[str] = None, dpi: int = 300
    ) -> Optional[str]:
        """
        Create scatter plot of predicted vs actual PER.

        Args:
            save_to: Path to save figure (PNG/PDF/SVG)
            dpi: Resolution for raster formats

        Returns:
            Path where figure was saved, or None if displayed
        """
        fig, ax = plt.subplots(figsize=(8, 6))

        # Scatter plot
        ax.scatter(self.actual_per, self.predicted_per, alpha=0.7, s=100, edgecolors="k")

        # Identity line
        min_val = min(self.actual_per.min(), self.predicted_per.min())
        max_val = max(self.actual_per.max(), self.predicted_per.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="Perfect prediction")

        # Labels and formatting
        ax.set_xlabel("Actual PER", fontsize=12)
        ax.set_ylabel("Predicted PER", fontsize=12)
        ax.set_title(
            f"{self.condition_name} - Predicted vs Actual PER\n"
            f"R² = {self.cv_r2_score:.3f}, MSE = {self.cv_mse:.4f}",
            fontsize=14,
        )
        ax.legend()
        ax.grid(alpha=0.3)

        # Annotate points with odorant names
        for i, odorant in enumerate(self.test_odorants):
            ax.annotate(
                odorant,
                (self.actual_per[i], self.predicted_per[i]),
                fontsize=8,
                alpha=0.7,
                xytext=(5, 5),
                textcoords="offset points",
            )

        plt.tight_layout()

        if save_to:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved prediction plot to {save_path}")
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return None

    def plot_receptors(
        self, n_top: int = 15, save_to: Optional[str] = None, dpi: int = 300
    ) -> Optional[str]:
        """
        Create bar plot of top receptor weights.

        Args:
            n_top: Number of top receptors to display
            save_to: Path to save figure
            dpi: Resolution for raster formats

        Returns:
            Path where figure was saved, or None if displayed
        """
        top_receptors = self.get_top_receptors(n_top)

        if not top_receptors:
            logger.warning("No receptors with non-zero weights")
            return None

        receptors, weights = zip(*top_receptors)

        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ["red" if w < 0 else "blue" for w in weights]
        ax.barh(range(len(receptors)), weights, color=colors, alpha=0.7, edgecolor="k")

        ax.set_yticks(range(len(receptors)))
        ax.set_yticklabels(receptors)
        ax.set_xlabel("LASSO Weight", fontsize=12)
        ax.set_ylabel("Receptor", fontsize=12)
        ax.set_title(
            f"{self.condition_name} - Top {n_top} Receptor Importance\n"
            f"({self.n_receptors_selected} total receptors selected, λ = {self.lambda_value:.4f})",
            fontsize=14,
        )
        ax.axvline(0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()

        if save_to:
            save_path = Path(save_to)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info(f"Saved receptor importance plot to {save_path}")
            plt.close()
            return str(save_path)
        else:
            plt.show()
            return None

    def export_csv(self, output_path: str):
        """
        Export results to CSV.

        Args:
            output_path: Path to save CSV file
        """
        rows = []
        for receptor, weight in sorted(
            self.lasso_weights.items(), key=lambda x: abs(x[1]), reverse=True
        ):
            rows.append(
                {
                    "condition": self.condition_name,
                    "trained_odorant": self.trained_odorant,
                    "receptor": receptor,
                    "lasso_weight": weight,
                    "abs_weight": abs(weight),
                }
            )

        df = pd.DataFrame(rows)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported results to {output_path}")

    def export_json(self, output_path: str):
        """
        Export model metadata and results to JSON.

        Args:
            output_path: Path to save JSON file
        """
        data = {
            "condition_name": self.condition_name,
            "trained_odorant": self.trained_odorant,
            "trained_odorant_door": self.trained_odorant_door,
            "lambda_value": float(self.lambda_value),
            "cv_r2_score": float(self.cv_r2_score),
            "cv_mse": float(self.cv_mse),
            "n_receptors_selected": int(self.n_receptors_selected),
            "receptor_coverage": int(self.receptor_coverage),
            "lasso_weights": {k: float(v) for k, v in self.lasso_weights.items()},
            "top_10_receptors": [
                {"receptor": r, "weight": float(w)} for r, w in self.get_top_receptors(10)
            ],
            "test_odorants": self.test_odorants,
            "actual_per": self.actual_per.tolist(),
            "predicted_per": self.predicted_per.tolist(),
            "prediction_error": (self.actual_per - self.predicted_per).tolist(),
        }

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported model to {output_path}")

    def summary(self) -> str:
        """
        Generate text summary of results.

        Returns:
            Formatted summary string
        """
        lines = [
            f"=" * 80,
            f"LASSO Behavioral Prediction Results: {self.condition_name}",
            f"=" * 80,
            f"Trained Odorant: {self.trained_odorant} (DoOR: {self.trained_odorant_door})",
            f"Receptor Coverage: {self.receptor_coverage} receptors with data",
            f"",
            f"Model Performance:",
            f"  Cross-validated R²: {self.cv_r2_score:.4f}",
            f"  Cross-validated MSE: {self.cv_mse:.4f}",
            f"  Lambda (α): {self.lambda_value:.6f}",
            f"",
            f"Sparse Receptor Circuit:",
            f"  Total receptors selected: {self.n_receptors_selected}",
            f"",
            f"Top 10 Receptors:",
        ]

        for i, (receptor, weight) in enumerate(self.get_top_receptors(10), 1):
            lines.append(f"  {i:2d}. {receptor:12s}  weight = {weight:+.4f}")

        lines.append("")
        lines.append(f"Test Odorants ({len(self.test_odorants)}):")
        for i, odorant in enumerate(self.test_odorants):
            lines.append(
                f"  {odorant:20s}  Actual = {self.actual_per[i]:.4f}  "
                f"Predicted = {self.predicted_per[i]:.4f}  "
                f"Error = {self.actual_per[i] - self.predicted_per[i]:+.4f}"
            )

        lines.append("=" * 80)

        return "\n".join(lines)


class LassoBehavioralPredictor:
    """
    LASSO regression-based behavioral prediction from optogenetic data.

    This class fits sparse linear models (LASSO) to identify minimal receptor
    circuits that predict behavioral responses (PER) in optogenetic manipulation
    experiments.

    Attributes:
        doorcache_path: Path to DoOR cache directory
        behavior_csv_path: Path to reaction_rates_summary_unordered.csv
        encoder: DoOREncoder instance
        behavioral_data: Loaded behavioral DataFrame
        scale_features: Whether to standardize receptor features
        scale_targets: Whether to standardize PER targets

    Example:
        >>> predictor = LassoBehavioralPredictor(
        ...     doorcache_path="door_cache",
        ...     behavior_csv_path="reaction_rates.csv"
        ... )
        >>> results = predictor.fit_behavior("opto_hex", "Hexanol")
        >>> results.plot_predictions(save_to="opto_hex_predictions.png")
        >>> print(results.summary())
    """

    # Odorant name mapping: CSV name → DoOR name
    ODORANT_NAME_MAPPING = {
        "3octonol": ["3-octanol", "oct-3-ol", "octan-3-ol"],
        "air": None,  # Control stimulus, not in DoOR
        "applecidervinegar": ["acetic acid", "ethanoic acid"],
        "benzaldehyde": ["benzaldehyde"],
        "citral": ["citral", "geranial"],
        "ethylbutyrate": ["ethyl butyrate", "ethyl butanoate"],
        "ethylbutyrate(6training)": ["ethyl butyrate", "ethyl butanoate"],
        "hexanol": ["1-hexanol", "hexan-1-ol"],
        "linalool": ["linalool"],
    }

    # Condition name → trained odorant mapping
    CONDITION_ODORANT_MAPPING = {
        "opto_hex": "Hexanol",
        "hex_control": "Hexanol",
        "opto_EB": "Ethyl_Butyrate",
        "opto_EB_6_training": "Ethyl_Butyrate_(6-Training)",
        "EB_control": "Ethyl_Butyrate",
        "opto_benz_1": "Benzaldehyde",
        "Benz_control": "Benzaldehyde",
        "opto_ACV": "Apple_Cider_Vinegar",
        "opto_3-oct": "3-Octonol",
        "opto_AIR": "AIR",
        "opto_3oct": "3-Octonol",  # Alternative naming
    }

    def __init__(
        self,
        doorcache_path: str,
        behavior_csv_path: str,
        scale_features: bool = True,
        scale_targets: bool = False,
    ):
        """
        Initialize LASSO behavioral predictor.

        Args:
            doorcache_path: Path to DoOR cache directory
            behavior_csv_path: Path to reaction_rates_summary_unordered.csv
            scale_features: Standardize receptor features before fitting
            scale_targets: Standardize PER targets before fitting

        Raises:
            FileNotFoundError: If paths don't exist
        """
        self.doorcache_path = Path(doorcache_path)
        self.behavior_csv_path = Path(behavior_csv_path)
        self.scale_features = scale_features
        self.scale_targets = scale_targets

        if not self.doorcache_path.exists():
            raise FileNotFoundError(f"DoOR cache not found: {self.doorcache_path}")
        if not self.behavior_csv_path.exists():
            raise FileNotFoundError(f"Behavioral CSV not found: {self.behavior_csv_path}")

        # Initialize encoder
        self.encoder = DoOREncoder(str(self.doorcache_path), use_torch=False)

        # Load behavioral data
        self.behavioral_data = pd.read_csv(self.behavior_csv_path, index_col=0)

        # Validate behavioral data is not empty
        if self.behavioral_data.empty or self.behavioral_data.shape[0] == 0:
            raise ValueError(
                f"Behavioral CSV is empty: {self.behavior_csv_path}. "
                "Expected at least one condition row."
            )

        logger.info(
            f"Loaded behavioral data: {self.behavioral_data.shape[0]} conditions, "
            f"{self.behavioral_data.shape[1]} odorants"
        )

        # Build reverse odorant mapping for fuzzy matching
        self._build_odorant_mapping()

    def _build_odorant_mapping(self):
        """Build case-insensitive odorant name mapping."""
        self.odorant_to_door = {}

        for csv_name, door_candidates in self.ODORANT_NAME_MAPPING.items():
            if door_candidates is None:
                self.odorant_to_door[csv_name] = None
                continue

            # Try each candidate
            for candidate in door_candidates:
                try:
                    # Test if odorant exists in DoOR
                    _ = self.encoder.encode(candidate)
                    self.odorant_to_door[csv_name] = candidate
                    logger.debug(f"Mapped '{csv_name}' → '{candidate}'")
                    break
                except KeyError:
                    continue

            # If no match found, try fuzzy matching
            if csv_name not in self.odorant_to_door:
                matches = get_close_matches(csv_name, self.encoder.odorant_names, n=1, cutoff=0.6)
                if matches:
                    self.odorant_to_door[csv_name] = matches[0]
                    logger.warning(f"Fuzzy matched '{csv_name}' → '{matches[0]}'")
                else:
                    self.odorant_to_door[csv_name] = None
                    logger.warning(f"Could not find DoOR match for '{csv_name}'")

    def match_odorant_name(self, csv_odorant_name: str) -> Optional[str]:
        """
        Match CSV odorant name to DoOR database name.

        Args:
            csv_odorant_name: Odorant name from behavioral CSV

        Returns:
            DoOR odorant name, or None if not found
        """
        csv_name_clean = csv_odorant_name.lower().replace("_", "").replace(" ", "").replace("-", "")
        return self.odorant_to_door.get(csv_name_clean)

    def get_receptor_profile(
        self, odorant_name: str, fill_missing: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Get receptor response profile for an odorant.

        Args:
            odorant_name: Odorant name (CSV format)
            fill_missing: Value for missing receptor responses

        Returns:
            Tuple of (response_vector, n_receptors_with_data)
        """
        door_name = self.match_odorant_name(odorant_name)

        if door_name is None:
            logger.warning(f"Odorant '{odorant_name}' not in DoOR, using zeros")
            profile = np.zeros(self.encoder.n_channels)
            coverage = 0
        else:
            profile = self.encoder.encode(door_name, fill_missing=fill_missing)
            if isinstance(profile, np.ndarray):
                profile = profile.astype(np.float64)
            else:
                profile = np.array(profile, dtype=np.float64)
            coverage = int(np.sum(~np.isnan(profile)))

        return profile, coverage

    def fit_behavior(
        self,
        condition_name: str,
        trained_odorant: Optional[str] = None,
        lambda_range: Optional[List[float]] = None,
        cv_folds: int = 5,
        prediction_mode: str = "test_odorant",
    ) -> BehaviorModelResults:
        """
        Fit LASSO model to predict PER from receptor profiles.

        Args:
            condition_name: Optogenetic condition (e.g., "opto_hex")
            trained_odorant: Odorant used for training (auto-detected if None)
            lambda_range: List of lambda values for cross-validation
            cv_folds: Number of cross-validation folds
            prediction_mode: Feature extraction mode:
                - "test_odorant": Use test odorant receptor profiles (default)
                - "trained_odorant": Use trained odorant receptor profile
                - "interaction": Use element-wise product of trained × test

        Returns:
            BehaviorModelResults object with fitted model and metrics

        Example:
            >>> predictor = LassoBehavioralPredictor("door_cache", "behavior.csv")
            >>> results = predictor.fit_behavior("opto_hex")
            >>> print(results.summary())
        """
        logger.info(f"Fitting LASSO model for condition: {condition_name}")

        # Get behavioral responses for this condition
        if condition_name not in self.behavioral_data.index:
            raise ValueError(f"Condition '{condition_name}' not found in behavioral data")

        per_responses = self.behavioral_data.loc[condition_name]

        # Filter out NaN and untested odorants
        valid_odorants = per_responses.dropna()
        if len(valid_odorants) == 0:
            raise ValueError(f"No valid PER data for condition '{condition_name}'")

        logger.info(f"Found {len(valid_odorants)} valid test odorants")

        # Auto-detect trained odorant (best-effort for all prediction modes)
        # Decision: Attempt auto-detection for all modes to populate results.trained_odorant
        # Evidence: test_fit_behavior_auto_detect_odorant expects trained_odorant even in test_odorant mode
        # Implementation: Try CONDITION_ODORANT_MAPPING, only raise if needed and missing
        if trained_odorant is None:
            trained_odorant_guess = self.CONDITION_ODORANT_MAPPING.get(condition_name)
            if trained_odorant_guess is not None:
                trained_odorant = trained_odorant_guess
                logger.info(f"Auto-detected trained odorant: {trained_odorant}")
            elif prediction_mode in ["trained_odorant", "interaction"]:
                # These modes require trained_odorant for feature extraction
                raise ValueError(
                    f"Could not auto-detect trained odorant for '{condition_name}'. "
                    f"Please specify manually or use prediction_mode='test_odorant'."
                )
            # For test_odorant mode: continue with trained_odorant=None (not needed for features)

        # Extract features based on prediction mode
        if prediction_mode == "test_odorant":
            X, test_odorants, y = self._extract_test_odorant_features(valid_odorants)
        elif prediction_mode == "trained_odorant":
            X, test_odorants, y = self._extract_trained_odorant_features(
                trained_odorant, valid_odorants
            )
        elif prediction_mode == "interaction":
            X, test_odorants, y = self._extract_interaction_features(
                trained_odorant, valid_odorants
            )
        else:
            raise ValueError(f"Unknown prediction_mode: {prediction_mode}")

        if X.shape[0] < 3:
            raise ValueError(
                f"Insufficient data: only {X.shape[0]} samples. Need at least 3 for CV."
            )

        # Warn about small sample sizes
        if X.shape[0] < 10:
            logger.warning(
                f"Small sample size ({X.shape[0]} samples). Results may be unreliable. "
                f"Consider collecting more behavioral data for robust predictions."
            )

        # Get trained odorant DoOR name and coverage
        if trained_odorant:
            trained_door_name = self.match_odorant_name(trained_odorant)
            if trained_door_name:
                _, trained_coverage = self.get_receptor_profile(trained_odorant)
            else:
                trained_coverage = 0
        else:
            trained_door_name = None
            trained_coverage = 0

        # Scale features and targets if requested
        X_scaled = X.copy()
        y_scaled = y.copy()

        if self.scale_features:
            scaler_X = StandardScaler()
            X_scaled = scaler_X.fit_transform(X)

        if self.scale_targets:
            scaler_y = StandardScaler()
            y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # Set default lambda range if not provided
        if lambda_range is None:
            lambda_range = np.logspace(-4, 0, 50)

        # Adjust CV folds for small samples
        # Use Leave-One-Out CV for very small samples (< 10)
        # This ensures each fold has at least n-1 samples
        if X_scaled.shape[0] < 10:
            cv_folds_adjusted = X_scaled.shape[0]  # LOOCV
            logger.info(
                f"Using Leave-One-Out CV ({cv_folds_adjusted} folds) due to small sample size"
            )
        else:
            cv_folds_adjusted = min(cv_folds, X_scaled.shape[0])

        # Fit LASSO with cross-validation
        lasso_cv = LassoCV(
            alphas=lambda_range, cv=cv_folds_adjusted, max_iter=10000, random_state=42
        )
        lasso_cv.fit(X_scaled, y_scaled)

        # Get best lambda and refit
        best_lambda = lasso_cv.alpha_
        logger.info(f"Selected lambda: {best_lambda:.6f}")

        lasso = Lasso(alpha=best_lambda, max_iter=10000, random_state=42)
        lasso.fit(X_scaled, y_scaled)

        # Predict
        y_pred_scaled = lasso.predict(X_scaled)

        # Unscale predictions if needed
        if self.scale_targets:
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
        else:
            y_pred = y_pred_scaled

        # Extract non-zero coefficients
        lasso_weights = {}
        for i, coef in enumerate(lasso.coef_):
            if abs(coef) > 1e-6:
                lasso_weights[self.encoder.receptor_names[i]] = float(coef)

        # Compute cross-validated metrics
        cv_scores = cross_val_score(
            lasso, X_scaled, y_scaled, cv=cv_folds_adjusted, scoring="r2"
        )
        cv_r2 = float(np.mean(cv_scores))

        cv_mse_scores = cross_val_score(
            lasso,
            X_scaled,
            y_scaled,
            cv=cv_folds_adjusted,
            scoring="neg_mean_squared_error",
        )
        cv_mse = float(-np.mean(cv_mse_scores))

        logger.info(f"Model performance: R² = {cv_r2:.4f}, MSE = {cv_mse:.4f}")
        logger.info(f"Selected {len(lasso_weights)} receptors with non-zero weights")

        # Create results object
        results = BehaviorModelResults(
            condition_name=condition_name,
            trained_odorant=trained_odorant,
            trained_odorant_door=trained_door_name or "NOT_FOUND",
            lasso_weights=lasso_weights,
            cv_r2_score=cv_r2,
            cv_mse=cv_mse,
            lambda_value=best_lambda,
            n_receptors_selected=len(lasso_weights),
            test_odorants=test_odorants,
            actual_per=y,
            predicted_per=y_pred,
            receptor_coverage=trained_coverage,
            feature_matrix=X,
            receptor_names=self.encoder.receptor_names,
        )

        return results

    def _extract_test_odorant_features(
        self, valid_odorants: pd.Series
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Extract features from test odorant receptor profiles."""
        X_list = []
        test_odorants = []
        y_list = []

        for odorant_name, per_value in valid_odorants.items():
            profile, coverage = self.get_receptor_profile(odorant_name)
            if coverage > 0:  # Only use odorants with some DoOR data
                X_list.append(profile)
                test_odorants.append(odorant_name)
                y_list.append(per_value)
            else:
                logger.warning(f"Skipping '{odorant_name}' - no DoOR data")

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)

        logger.info(f"Feature matrix shape: {X.shape}")
        return X, test_odorants, y

    def _extract_trained_odorant_features(
        self, trained_odorant: str, valid_odorants: pd.Series
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Extract features from trained odorant receptor profile (replicated)."""
        trained_profile, coverage = self.get_receptor_profile(trained_odorant)

        if coverage == 0:
            raise ValueError(f"Trained odorant '{trained_odorant}' has no DoOR data")

        # Replicate trained profile for each test odorant
        X_list = []
        test_odorants = []
        y_list = []

        for odorant_name, per_value in valid_odorants.items():
            X_list.append(trained_profile)
            test_odorants.append(odorant_name)
            y_list.append(per_value)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)

        logger.info(f"Using trained odorant profile replicated {len(y_list)} times")
        return X, test_odorants, y

    def _extract_interaction_features(
        self, trained_odorant: str, valid_odorants: pd.Series
    ) -> Tuple[np.ndarray, List[str], np.ndarray]:
        """Extract features from element-wise product of trained × test profiles."""
        trained_profile, trained_coverage = self.get_receptor_profile(trained_odorant)

        if trained_coverage == 0:
            raise ValueError(f"Trained odorant '{trained_odorant}' has no DoOR data")

        X_list = []
        test_odorants = []
        y_list = []

        for odorant_name, per_value in valid_odorants.items():
            test_profile, test_coverage = self.get_receptor_profile(odorant_name)
            if test_coverage > 0:
                interaction = trained_profile * test_profile
                X_list.append(interaction)
                test_odorants.append(odorant_name)
                y_list.append(per_value)

        X = np.array(X_list, dtype=np.float64)
        y = np.array(y_list, dtype=np.float64)

        logger.info(f"Using interaction features (trained × test)")
        return X, test_odorants, y

    def compare_conditions(
        self,
        conditions: List[str],
        lambda_range: Optional[List[float]] = None,
        plot: bool = True,
        save_dir: Optional[str] = None,
    ) -> Dict[str, BehaviorModelResults]:
        """
        Compare LASSO models across multiple optogenetic conditions.

        Args:
            conditions: List of condition names
            lambda_range: Lambda values for LASSO CV
            plot: Generate comparison plots
            save_dir: Directory to save plots

        Returns:
            Dictionary mapping condition name to BehaviorModelResults

        Example:
            >>> predictor = LassoBehavioralPredictor("door_cache", "behavior.csv")
            >>> comparison = predictor.compare_conditions(
            ...     ["opto_hex", "opto_EB", "opto_benz_1"]
            ... )
        """
        results = {}

        for condition in conditions:
            try:
                result = self.fit_behavior(condition, lambda_range=lambda_range)
                results[condition] = result
                logger.info(f"✓ {condition}: R² = {result.cv_r2_score:.3f}")

                if plot and save_dir:
                    save_path = Path(save_dir)
                    result.plot_predictions(save_to=str(save_path / f"{condition}_predictions.png"))
                    result.plot_receptors(save_to=str(save_path / f"{condition}_receptors.png"))

            except Exception as e:
                logger.error(f"✗ {condition}: {e}")
                continue

        if plot:
            self._plot_comparison(results, save_dir)

        return results

    def _plot_comparison(
        self, results: Dict[str, BehaviorModelResults], save_dir: Optional[str] = None
    ):
        """Create comparison plots across conditions."""
        if not results:
            return

        # Plot 1: R² comparison
        fig, ax = plt.subplots(figsize=(10, 6))
        conditions = list(results.keys())
        r2_scores = [results[c].cv_r2_score for c in conditions]

        ax.bar(range(len(conditions)), r2_scores, alpha=0.7, edgecolor="k")
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_ylabel("Cross-validated R²", fontsize=12)
        ax.set_title("Model Performance Comparison", fontsize=14)
        ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
        ax.grid(alpha=0.3, axis="y")

        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / "comparison_r2.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved comparison plot to {save_path}")
            plt.close()
        else:
            plt.show()

        # Plot 2: Receptor overlap heatmap
        self._plot_receptor_overlap(results, save_dir)

    def _plot_receptor_overlap(
        self, results: Dict[str, BehaviorModelResults], save_dir: Optional[str] = None
    ):
        """Plot heatmap of receptor overlap between conditions."""
        conditions = list(results.keys())
        n = len(conditions)
        overlap_matrix = np.zeros((n, n))

        for i, cond1 in enumerate(conditions):
            receptors1 = set(results[cond1].lasso_weights.keys())
            for j, cond2 in enumerate(conditions):
                receptors2 = set(results[cond2].lasso_weights.keys())
                if len(receptors1) > 0 and len(receptors2) > 0:
                    overlap = len(receptors1 & receptors2) / len(receptors1 | receptors2)
                    overlap_matrix[i, j] = overlap

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(overlap_matrix, cmap="YlOrRd", vmin=0, vmax=1)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(conditions, rotation=45, ha="right")
        ax.set_yticklabels(conditions)

        # Add text annotations
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, f"{overlap_matrix[i, j]:.2f}", ha="center", va="center")

        ax.set_title("Receptor Overlap (Jaccard Index)", fontsize=14)
        plt.colorbar(im, ax=ax, label="Jaccard Index")
        plt.tight_layout()

        if save_dir:
            save_path = Path(save_dir) / "receptor_overlap.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved receptor overlap heatmap to {save_path}")
            plt.close()
        else:
            plt.show()
