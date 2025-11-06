"""
Behavioral Prediction Module
=============================

Predict behavioral responses from receptor activation patterns.

This module uses DoOR receptor response profiles to predict behavioral outcomes
like attraction, avoidance, and feeding responses.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from door_toolkit.encoder import DoOREncoder
from door_toolkit.utils import load_response_matrix

logger = logging.getLogger(__name__)


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

    def compare_odorants(self, odorant1: str, odorant2: str) -> Dict[str, any]:
        """
        Compare predicted behaviors between two odorants.

        Args:
            odorant1: First odorant name
            odorant2: Second odorant name

        Returns:
            Dictionary with comparison results

        Example:
            >>> predictor = BehavioralPredictor("door_cache")
            >>> comparison = predictor.compare_odorants("hexanol", "geosmin")
            >>> print(comparison["similarity"])
        """
        pred1 = self.predict_behavior(odorant1)
        pred2 = self.predict_behavior(odorant2)

        # Calculate pattern similarity (cosine similarity)
        all_receptors = set(pred1.receptor_pattern.keys()) | set(pred2.receptor_pattern.keys())

        vec1 = np.array([pred1.receptor_pattern.get(r, 0.0) for r in all_receptors])
        vec2 = np.array([pred2.receptor_pattern.get(r, 0.0) for r in all_receptors])

        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 > 0 and norm2 > 0:
            similarity = dot_product / (norm1 * norm2)
        else:
            similarity = 0.0

        comparison = {
            "odorant1": odorant1,
            "odorant2": odorant2,
            "valence1": pred1.predicted_valence,
            "valence2": pred2.predicted_valence,
            "valence_match": pred1.predicted_valence == pred2.predicted_valence,
            "similarity": float(similarity),
            "confidence1": pred1.confidence,
            "confidence2": pred2.confidence,
        }

        return comparison

    def find_similar_odorants(
        self, target_odorant: str, n: int = 5, min_similarity: float = 0.3
    ) -> List[Tuple[str, float]]:
        """
        Find odorants with similar predicted behavioral profiles.

        Args:
            target_odorant: Target odorant to compare against
            n: Number of similar odorants to return
            min_similarity: Minimum similarity threshold

        Returns:
            List of (odorant_name, similarity) tuples

        Example:
            >>> predictor = BehavioralPredictor("door_cache")
            >>> similar = predictor.find_similar_odorants("hexanol", n=5)
            >>> for odor, sim in similar:
            ...     print(f"{odor}: {sim:.3f}")
        """
        logger.info(f"Finding odorants similar to {target_odorant}")

        target_pred = self.predict_behavior(target_odorant)
        target_vec = target_pred.receptor_pattern

        similarities = []
        all_odorants = self.encoder.odorant_names

        for odorant in all_odorants:
            if odorant.lower() == target_odorant.lower():
                continue

            try:
                comparison = self.compare_odorants(target_odorant, odorant)
                similarity = comparison["similarity"]

                if similarity >= min_similarity:
                    similarities.append((odorant, similarity))

            except Exception:
                continue

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:n]

    def create_prediction_report(
        self, odorants: List[str], output_path: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create comprehensive prediction report for multiple odorants.

        Args:
            odorants: List of odorant names
            output_path: Optional path to save CSV report

        Returns:
            DataFrame with prediction results

        Example:
            >>> predictor = BehavioralPredictor("door_cache")
            >>> odorants = ["hexanol", "geosmin", "ethyl butyrate"]
            >>> report = predictor.create_prediction_report(odorants)
            >>> print(report)
        """
        predictions = self.predict_batch(odorants)

        rows = []
        for pred in predictions:
            row = {
                "odorant": pred.odorant_name,
                "predicted_valence": pred.predicted_valence,
                "confidence": pred.confidence,
                "n_active_receptors": len(pred.receptor_pattern),
            }

            # Add top 3 contributors
            for i, (receptor, contribution) in enumerate(pred.key_contributors[:3], 1):
                row[f"top_receptor_{i}"] = receptor
                row[f"contribution_{i}"] = contribution

            rows.append(row)

        df = pd.DataFrame(rows)

        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Saved prediction report to {output_path}")

        return df

    def validate_known_behaviors(self, known_behaviors: Dict[str, str]) -> Dict[str, float]:
        """
        Validate predictor against known behavioral data.

        Args:
            known_behaviors: Dictionary mapping odorant to known valence

        Returns:
            Dictionary with validation metrics

        Example:
            >>> predictor = BehavioralPredictor("door_cache")
            >>> known = {"hexanol": "attractive", "geosmin": "aversive"}
            >>> metrics = predictor.validate_known_behaviors(known)
            >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
        """
        correct = 0
        total = 0

        for odorant, true_valence in known_behaviors.items():
            try:
                pred = self.predict_behavior(odorant)
                predicted_valence = pred.predicted_valence.split("(")[0].strip()

                if predicted_valence.lower() == true_valence.lower():
                    correct += 1
                total += 1

            except Exception as e:
                logger.warning(f"Could not validate {odorant}: {e}")
                continue

        accuracy = correct / total if total > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "error_rate": 1.0 - accuracy,
        }

        logger.info(f"Validation: {correct}/{total} correct ({accuracy:.2%} accuracy)")

        return metrics
