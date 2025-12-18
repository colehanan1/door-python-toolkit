"""
Tests for StaticDoORRecurrentCircuit model

Validation requirements:
1. Forward pass shape correctness
2. Backward pass correctness
3. Trainable parameters match constraint tier selection
4. Connectivity matrices are buffers (no gradients)
5. KC sparsity is correctly enforced
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path

from door_toolkit.neural.static_door_recurrent_circuit import StaticDoORRecurrentCircuit


@pytest.fixture
def test_data_dir():
    """Return path to test data directory."""
    return Path('data/pgcn_features')


@pytest.fixture
def mock_data_dir():
    """Create temporary directory with mock data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create mock feature schema
        feature_schema = {
            'feature_dim': 237,
            'feature_groups': {
                'test_door_profile': list(range(78, 156))
            }
        }
        with open(tmpdir / 'feature_metadata.json', 'w') as f:
            json.dump(feature_schema, f)

        # Create mock connectivity metadata
        connectivity_metadata = {
            'n_receptors': 78,
            'n_pns': 841,
            'n_kcs': 608,
        }
        with open(tmpdir / 'connectivity_metadata.json', 'w') as f:
            json.dump(connectivity_metadata, f)

        # Create mock connectivity matrices
        orn_pn_W = torch.randn(78, 841) * 0.1
        pn_kc_W = torch.randn(841, 608) * 0.1
        torch.save(orn_pn_W, tmpdir / 'orn_pn_connectivity.pt')
        torch.save(pn_kc_W, tmpdir / 'pn_kc_connectivity.pt')

        yield tmpdir


def test_model_initialization(mock_data_dir):
    """Test model can be initialized with correct shapes."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )

    assert model.n_orns == 78
    assert model.n_pns == 841
    assert model.n_kcs == 608
    assert model.orn_pn_W.shape == (78, 841)
    assert model.pn_kc_W.shape == (841, 608)


def test_forward_pass_shapes(mock_data_dir):
    """Test forward pass produces correct output shapes."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )

    batch_size = 5
    features = torch.randn(batch_size, 237)

    logits, pn_hidden, activations = model.forward(features)

    assert logits.shape == (batch_size, 1)
    assert pn_hidden.shape == (batch_size, 841)
    assert activations['orns'].shape == (batch_size, 78)
    assert activations['pns'].shape == (batch_size, 841)
    assert activations['kcs_dense'].shape == (batch_size, 608)
    assert activations['kcs_sparse'].shape == (batch_size, 608)


def test_sequence_forward(mock_data_dir):
    """Test forward_sequence processes sequences correctly."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )

    seq_len = 10
    features_seq = torch.randn(seq_len, 237)

    logits_seq, activations_list = model.forward_sequence(features_seq, reset_hidden=True)

    assert logits_seq.shape == (seq_len, 1)
    assert len(activations_list) == seq_len


def test_backward_pass(mock_data_dir):
    """Test backward pass works without errors."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )

    features = torch.randn(3, 237)
    labels = torch.tensor([0.0, 1.0, 0.0])

    logits, _, _ = model.forward(features)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.squeeze(), labels
    )
    loss.backward()

    # Check that at least some parameters have gradients
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "No parameters received gradients"


def test_constraint_tier_0_gradients(mock_data_dir):
    """Test Tier 0: only readout parameters have gradients."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=0,
    )

    trainable = model.get_trainable_parameters()
    trainable_names = [name for name, _ in trainable]

    # Tier 0: only readout
    assert 'readout.weight' in trainable_names
    assert 'readout.bias' in trainable_names
    assert 'pn_gain' not in trainable_names
    assert 'pn_bias' not in trainable_names
    assert 'kc_gain' not in trainable_names
    assert 'kc_bias' not in trainable_names

    # GRU params should not be trainable
    gru_trainable = [name for name in trainable_names if 'pn_gru' in name]
    assert len(gru_trainable) == 0


def test_constraint_tier_1_gradients(mock_data_dir):
    """Test Tier 1: readout + gains/biases have gradients."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=1,
    )

    trainable = model.get_trainable_parameters()
    trainable_names = [name for name, _ in trainable]

    # Tier 1: readout + gains/biases
    assert 'readout.weight' in trainable_names
    assert 'readout.bias' in trainable_names
    assert 'pn_gain' in trainable_names
    assert 'pn_bias' in trainable_names
    assert 'kc_gain' in trainable_names
    assert 'kc_bias' in trainable_names

    # GRU params should not be trainable yet
    gru_trainable = [name for name in trainable_names if 'pn_gru' in name]
    assert len(gru_trainable) == 0


def test_constraint_tier_2_gradients(mock_data_dir):
    """Test Tier 2: readout + gains/biases + GRU have gradients."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=2,
    )

    trainable = model.get_trainable_parameters()
    trainable_names = [name for name, _ in trainable]

    # Tier 2: readout + gains/biases + GRU
    assert 'readout.weight' in trainable_names
    assert 'readout.bias' in trainable_names
    assert 'pn_gain' in trainable_names
    assert 'pn_bias' in trainable_names
    assert 'kc_gain' in trainable_names
    assert 'kc_bias' in trainable_names

    # GRU params should now be trainable
    gru_trainable = [name for name in trainable_names if 'pn_gru' in name]
    assert len(gru_trainable) > 0


def test_connectivity_buffers_no_grad(mock_data_dir):
    """Test that connectivity matrices are buffers without gradients."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=2,  # Most permissive tier
    )

    # Check that connectivity matrices don't require gradients
    assert not model.orn_pn_W.requires_grad
    assert not model.pn_kc_W.requires_grad

    # Verify they're buffers, not parameters
    param_names = [name for name, _ in model.named_parameters()]
    assert 'orn_pn_W' not in param_names
    assert 'pn_kc_W' not in param_names


def test_kc_sparsity_enforcement(mock_data_dir):
    """Test that KC sparsity is correctly enforced."""
    kc_sparsity_k = 30
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=0,
        kc_sparsity_k=kc_sparsity_k,
    )

    batch_size = 5
    features = torch.randn(batch_size, 237)

    logits, _, activations = model.forward(features)
    kcs_sparse = activations['kcs_sparse']

    # Count active KCs per sample
    n_active = (kcs_sparse > 0).sum(dim=1)

    # All samples should have exactly k active KCs
    assert torch.all(n_active == kc_sparsity_k), \
        f"Expected {kc_sparsity_k} active KCs, got {n_active.tolist()}"


def test_sparsity_stats(mock_data_dir):
    """Test sparsity statistics computation."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=0,
        kc_sparsity_k=30,
    )

    batch_size = 10
    features = torch.randn(batch_size, 237)
    logits, _, activations = model.forward(features)

    stats = model.compute_sparsity_stats(activations['kcs_sparse'])

    assert 'kc_frac_active' in stats
    assert 'kc_n_active_mean' in stats
    assert 0.0 <= stats['kc_frac_active'] <= 1.0
    assert stats['kc_n_active_mean'] > 0


def test_model_config_serialization(mock_data_dir):
    """Test model configuration can be serialized."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=1,
        kc_sparsity_k=25,
    )

    config = model.get_config()

    assert config['n_orns'] == 78
    assert config['n_pns'] == 841
    assert config['n_kcs'] == 608
    assert config['constraint_tier'] == 1
    assert config['kc_sparsity_k'] == 25

    # Should be JSON serializable
    json_str = json.dumps(config)
    assert len(json_str) > 0


def test_hidden_state_persistence(mock_data_dir):
    """Test that hidden state persists across trials in a sequence."""
    model = StaticDoORRecurrentCircuit(
        feature_schema_path=str(mock_data_dir / 'feature_metadata.json'),
        orn_pn_connectivity_path=str(mock_data_dir / 'orn_pn_connectivity.pt'),
        pn_kc_connectivity_path=str(mock_data_dir / 'pn_kc_connectivity.pt'),
        connectivity_metadata_path=str(mock_data_dir / 'connectivity_metadata.json'),
        constraint_tier=2,  # Need recurrence for this test
    )
    model.eval()

    # Create a sequence
    seq_len = 5
    features_seq = torch.randn(seq_len, 237)

    # Process sequence maintaining hidden state
    with torch.no_grad():
        pn_hidden = model.init_hidden(batch_size=1)
        hidden_states = [pn_hidden.clone()]

        for t in range(seq_len):
            features_t = features_seq[t:t+1]
            _, pn_hidden, _ = model.forward(features_t, pn_hidden)
            hidden_states.append(pn_hidden.clone())

    # Hidden state should change over time (unless weights are zero)
    # Check that at least some hidden states are different
    diffs = [torch.norm(hidden_states[i+1] - hidden_states[i]).item()
             for i in range(len(hidden_states)-1)]

    # At least one transition should show change
    assert any(d > 1e-6 for d in diffs), \
        "Hidden state did not change across sequence"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
