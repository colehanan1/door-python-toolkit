"""Tests for pathway analysis modules."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from door_toolkit.pathways.analyzer import PathwayAnalyzer, PathwayResult
from door_toolkit.pathways.blocking_experiments import (
    BlockingExperimentGenerator,
    ExperimentProtocol,
    ExperimentStep,
)
from door_toolkit.pathways.behavioral_prediction import (
    BehavioralPredictor,
    BehaviorPrediction,
)


@pytest.fixture
def mock_pathway_result():
    """Create mock pathway result."""
    return PathwayResult(
        pathway_name="Test Pathway",
        source_receptors=["Or42b", "Or47b"],
        target_behavior="attraction",
        strength=0.75,
        receptor_contributions={"Or42b": 0.6, "Or47b": 0.4},
        metadata={"test": "value"},
    )


class TestPathwayResult:
    """Tests for PathwayResult dataclass."""

    def test_initialization(self, mock_pathway_result):
        """Test pathway result initialization."""
        assert mock_pathway_result.pathway_name == "Test Pathway"
        assert len(mock_pathway_result.source_receptors) == 2
        assert mock_pathway_result.strength == 0.75

    def test_to_dict(self, mock_pathway_result):
        """Test dictionary serialization."""
        result_dict = mock_pathway_result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["pathway_name"] == "Test Pathway"
        assert result_dict["strength"] == 0.75

    def test_get_top_receptors(self, mock_pathway_result):
        """Test getting top receptors."""
        top = mock_pathway_result.get_top_receptors(n=2)

        assert len(top) == 2
        assert top[0][0] == "Or42b"  # Highest contributor
        assert top[0][1] == 0.6


class TestPathwayAnalyzer:
    """Tests for PathwayAnalyzer class."""

    @pytest.fixture
    def mock_analyzer(self, mock_door_cache):
        """Create mock analyzer."""
        return PathwayAnalyzer(str(mock_door_cache))

    def test_initialization(self, mock_analyzer):
        """Test analyzer initialization."""
        assert mock_analyzer.encoder is not None
        assert mock_analyzer.response_matrix is not None

    def test_known_pathways(self, mock_analyzer):
        """Test known pathway configurations."""
        assert "or47b_feeding" in PathwayAnalyzer.KNOWN_PATHWAYS
        assert "or42b" in PathwayAnalyzer.KNOWN_PATHWAYS

        pathway_config = PathwayAnalyzer.KNOWN_PATHWAYS["or47b_feeding"]
        assert "receptors" in pathway_config
        assert "Or47b" in pathway_config["receptors"]

    def test_trace_or47b_pathway(self, mock_analyzer):
        """Test Or47b pathway tracing."""
        pathway = mock_analyzer.trace_or47b_feeding_pathway()

        assert isinstance(pathway, PathwayResult)
        assert pathway.pathway_name is not None
        assert pathway.target_behavior is not None
        assert isinstance(pathway.strength, float)

    def test_trace_or42b_pathway(self, mock_analyzer):
        """Test Or42b pathway tracing."""
        pathway = mock_analyzer.trace_or42b_pathway()

        assert isinstance(pathway, PathwayResult)
        assert "Or42b" in pathway.pathway_name or "Or42b" in pathway.source_receptors

    def test_trace_custom_pathway(self, mock_analyzer):
        """Test custom pathway tracing."""
        pathway = mock_analyzer.trace_custom_pathway(
            receptors=["Or42b"],
            odorants=["ethyl butyrate"],
            behavior="attraction",
        )

        assert isinstance(pathway, PathwayResult)
        assert pathway.target_behavior == "attraction"
        assert "Or42b" in pathway.source_receptors


class TestBlockingExperimentGenerator:
    """Tests for BlockingExperimentGenerator class."""

    @pytest.fixture
    def mock_generator(self, mock_door_cache):
        """Create mock generator."""
        return BlockingExperimentGenerator(str(mock_door_cache))

    def test_initialization(self, mock_generator):
        """Test generator initialization."""
        assert mock_generator.analyzer is not None

    def test_generate_experiment_1(self, mock_generator):
        """Test experiment 1 protocol generation."""
        protocol = mock_generator.generate_experiment_1_protocol()

        assert isinstance(protocol, ExperimentProtocol)
        assert protocol.experiment_id == "PGCN-EXP-001"
        assert len(protocol.steps) > 0
        assert len(protocol.controls) > 0

    def test_generate_experiment_2(self, mock_generator):
        """Test experiment 2 protocol generation."""
        protocol = mock_generator.generate_experiment_2_protocol()

        assert isinstance(protocol, ExperimentProtocol)
        assert protocol.experiment_id == "PGCN-EXP-002"
        assert "microsurgery" in protocol.experiment_name.lower()

    def test_generate_experiment_3(self, mock_generator):
        """Test experiment 3 protocol generation."""
        protocol = mock_generator.generate_experiment_3_protocol()

        assert isinstance(protocol, ExperimentProtocol)
        assert protocol.experiment_id == "PGCN-EXP-003"
        assert "synaptic" in protocol.experiment_name.lower()

    def test_generate_experiment_6(self, mock_generator):
        """Test experiment 6 protocol generation."""
        protocol = mock_generator.generate_experiment_6_protocol()

        assert isinstance(protocol, ExperimentProtocol)
        assert protocol.experiment_id == "PGCN-EXP-006"

    def test_protocol_export_json(self, mock_generator):
        """Test protocol JSON export."""
        protocol = mock_generator.generate_experiment_1_protocol()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
            temp_path = f.name

        try:
            protocol.export_json(temp_path)
            assert Path(temp_path).exists()
            assert Path(temp_path).stat().st_size > 0
        finally:
            Path(temp_path).unlink()

    def test_protocol_export_markdown(self, mock_generator):
        """Test protocol Markdown export."""
        protocol = mock_generator.generate_experiment_1_protocol()

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
            temp_path = f.name

        try:
            protocol.export_markdown(temp_path)
            assert Path(temp_path).exists()
            content = Path(temp_path).read_text()
            assert "# Experiment Protocol" in content
        finally:
            Path(temp_path).unlink()


class TestBehavioralPredictor:
    """Tests for BehavioralPredictor class."""

    @pytest.fixture
    def mock_predictor(self, mock_door_cache):
        """Create mock predictor."""
        return BehavioralPredictor(str(mock_door_cache))

    def test_initialization(self, mock_predictor):
        """Test predictor initialization."""
        assert mock_predictor.encoder is not None
        assert mock_predictor.response_matrix is not None

    def test_attractive_receptors(self, mock_predictor):
        """Test attractive receptor configuration."""
        assert "Or42b" in BehavioralPredictor.ATTRACTIVE_RECEPTORS
        assert "Or47b" in BehavioralPredictor.ATTRACTIVE_RECEPTORS

    def test_aversive_receptors(self, mock_predictor):
        """Test aversive receptor configuration."""
        assert "Or92a" in BehavioralPredictor.AVERSIVE_RECEPTORS

    def test_predict_behavior(self, mock_predictor):
        """Test behavior prediction."""
        # Get first available odorant
        odorant = mock_predictor.encoder.odorant_names[0]
        prediction = mock_predictor.predict_behavior(odorant)

        assert isinstance(prediction, BehaviorPrediction)
        assert prediction.odorant_name == odorant
        assert prediction.predicted_valence in ["attractive", "aversive", "neutral"] or "feeding" in prediction.predicted_valence
        assert 0 <= prediction.confidence <= 1

    def test_prediction_to_dict(self, mock_predictor):
        """Test prediction serialization."""
        odorant = mock_predictor.encoder.odorant_names[0]
        prediction = mock_predictor.predict_behavior(odorant)

        pred_dict = prediction.to_dict()
        assert isinstance(pred_dict, dict)
        assert "odorant_name" in pred_dict
        assert "predicted_valence" in pred_dict


def test_experiment_step():
    """Test ExperimentStep dataclass."""
    step = ExperimentStep(
        step_number=1,
        action="Test action",
        target="Test target",
        method="Test method",
        parameters={"param1": "value1"},
        expected_outcome="Test outcome",
        measurement="Test measurement",
    )

    assert step.step_number == 1
    assert step.action == "Test action"

    step_dict = step.to_dict()
    assert isinstance(step_dict, dict)
    assert step_dict["step_number"] == 1
