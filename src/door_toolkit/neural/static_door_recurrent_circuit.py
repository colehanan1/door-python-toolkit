"""
Static DoOR Recurrent Circuit: Connectome-Constrained RNN

Decision → Evidence → Implementation
-----------------------------------
Decision: Use fixed connectome adjacency from FlyWire as non-trainable buffers.
Evidence: Connectome structure is a hard biological constraint (Zheng et al., Cell 2018).
         We test whether fixed wiring + minimal plasticity can explain behavior.
Implementation: orn_pn_W and pn_kc_W are registered as buffers (no gradient).

Decision: Place recurrent dynamics in PN layer, not KC layer.
Evidence: PNs show trial history effects (Gupta & Stopfer, J Neurosci 2012).
         KCs are sparse and respond more stereotypically (Turner et al., eLife 2008).
Implementation: GRUCell operates on PN activations. KCs receive feedforward PN input.

Decision: Enforce KC sparsity with top-k activation.
Evidence: KCs fire sparsely in vivo (~5% active per odor, Turner et al., eLife 2008).
Implementation: Top-k selection (k = 5% of 608 = 30) by default. Configurable.

Decision: Three constraint tiers for ablation studies.
Evidence: Scientific claims require showing minimal model complexity. Tiered training
         allows testing: (Tier 0) readout only, (Tier 1) + per-neuron gains,
         (Tier 2) + recurrent dynamics. Standard practice (Sussillo & Barak, Neuron 2013).
Implementation: set_constraint_tier() enables/disables parameter gradients.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticDoORRecurrentCircuit(nn.Module):
    """
    Connectome-constrained recurrent circuit for fly behavior prediction.

    Architecture:
        Input [batch, 237] → extract test profile [batch, 78]
        ORN [78] → PN [841] (fixed connectivity)
        PN dynamics: GRUCell (recurrent)
        PN [841] → KC [608] (fixed connectivity)
        KC sparsity: top-k selection
        KC [608] → MBON [1] (readout)

    Constraint Tiers:
        Tier 0: Only KC→MBON readout trainable
        Tier 1: Tier 0 + per-neuron gains/biases on PN and/or KC
        Tier 2: Tier 1 + PN recurrent parameters (GRU weights)

    Note: Adjacency structure is always fixed (from FlyWire). Only gains/recurrence vary.
    """

    def __init__(
        self,
        feature_schema_path: str,
        orn_pn_connectivity_path: str,
        pn_kc_connectivity_path: str,
        connectivity_metadata_path: str,
        constraint_tier: int = 0,
        kc_sparsity_k: Optional[int] = None,
        pn_hidden_init: str = 'zero',
        device: str = 'cpu',
    ):
        """
        Args:
            feature_schema_path: Path to feature_metadata.json
            orn_pn_connectivity_path: Path to orn_pn_connectivity.pt [78, 841]
            pn_kc_connectivity_path: Path to pn_kc_connectivity.pt [841, 608]
            connectivity_metadata_path: Path to connectivity_metadata.json
            constraint_tier: 0, 1, or 2 (see class docstring)
            kc_sparsity_k: Number of KCs to keep active (default: 5% of 608 = 30)
            pn_hidden_init: How to initialize PN hidden state ('zero' or 'learned')
            device: 'cpu' or 'cuda'
        """
        super().__init__()
        self.device = device
        self.constraint_tier = constraint_tier
        self.pn_hidden_init = pn_hidden_init

        # Load feature schema
        with open(feature_schema_path) as f:
            self.feature_schema = json.load(f)
        self.test_profile_indices = torch.tensor(
            self.feature_schema['feature_groups']['test_door_profile']
        )

        # Load connectivity metadata
        with open(connectivity_metadata_path) as f:
            self.connectivity_metadata = json.load(f)

        self.n_orns = self.connectivity_metadata['n_receptors']
        self.n_pns = self.connectivity_metadata['n_pns']
        self.n_kcs = self.connectivity_metadata['n_kcs']

        # Load connectome matrices (as buffers - no gradient)
        orn_pn_W = torch.load(orn_pn_connectivity_path, weights_only=False).float()
        pn_kc_W = torch.load(pn_kc_connectivity_path, weights_only=False).float()

        # Validate shapes
        assert orn_pn_W.shape == (self.n_orns, self.n_pns), \
            f"ORN-PN shape {orn_pn_W.shape} != expected {(self.n_orns, self.n_pns)}"
        assert pn_kc_W.shape == (self.n_pns, self.n_kcs), \
            f"PN-KC shape {pn_kc_W.shape} != expected {(self.n_pns, self.n_kcs)}"

        # Register as buffers (not parameters)
        self.register_buffer('orn_pn_W', orn_pn_W)
        self.register_buffer('pn_kc_W', pn_kc_W)

        # KC sparsity
        if kc_sparsity_k is None:
            kc_sparsity_k = max(1, int(0.05 * self.n_kcs))  # 5% default
        self.kc_sparsity_k = kc_sparsity_k

        # PN recurrent cell (GRU)
        self.pn_gru = nn.GRUCell(input_size=self.n_pns, hidden_size=self.n_pns)

        # PN hidden state initialization
        if pn_hidden_init == 'learned':
            self.pn_h0 = nn.Parameter(torch.zeros(self.n_pns))
        else:
            self.register_buffer('pn_h0', torch.zeros(self.n_pns))

        # Tier 1: Per-neuron gains and biases
        self.pn_gain = nn.Parameter(torch.ones(self.n_pns))
        self.pn_bias = nn.Parameter(torch.zeros(self.n_pns))
        self.kc_gain = nn.Parameter(torch.ones(self.n_kcs))
        self.kc_bias = nn.Parameter(torch.zeros(self.n_kcs))

        # Readout: KC → MBON
        self.readout = nn.Linear(self.n_kcs, 1)

        # Set constraint tier
        self.set_constraint_tier(constraint_tier)

    def set_constraint_tier(self, tier: int):
        """
        Set which parameters are trainable.

        Tier 0: Only readout (KC→MBON)
        Tier 1: Tier 0 + PN/KC gains and biases
        Tier 2: Tier 1 + PN GRU recurrent weights
        """
        assert tier in [0, 1, 2], f"Invalid tier: {tier}. Must be 0, 1, or 2."
        self.constraint_tier = tier

        # Tier 0: Only readout
        self.readout.weight.requires_grad = True
        self.readout.bias.requires_grad = True

        # Tier 1: Add gains and biases
        if tier >= 1:
            self.pn_gain.requires_grad = True
            self.pn_bias.requires_grad = True
            self.kc_gain.requires_grad = True
            self.kc_bias.requires_grad = True
            if self.pn_hidden_init == 'learned':
                self.pn_h0.requires_grad = True
        else:
            self.pn_gain.requires_grad = False
            self.pn_bias.requires_grad = False
            self.kc_gain.requires_grad = False
            self.kc_bias.requires_grad = False
            if self.pn_hidden_init == 'learned':
                self.pn_h0.requires_grad = False

        # Tier 2: Add recurrent dynamics
        if tier >= 2:
            for param in self.pn_gru.parameters():
                param.requires_grad = True
        else:
            for param in self.pn_gru.parameters():
                param.requires_grad = False

    def get_trainable_parameters(self) -> List[Tuple[str, torch.Tensor]]:
        """Return list of (name, param) for trainable parameters."""
        return [(name, param) for name, param in self.named_parameters()
                if param.requires_grad]

    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """Initialize PN hidden state."""
        return self.pn_h0.unsqueeze(0).expand(batch_size, -1)

    def forward(
        self,
        features: torch.Tensor,
        pn_hidden: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the circuit.

        Args:
            features: [batch, feature_dim] trial features
            pn_hidden: [batch, n_pns] previous PN hidden state (None = init to zero)

        Returns:
            logits: [batch, 1] output
            pn_hidden_next: [batch, n_pns] updated PN hidden state
            activations: Dict with intermediate activations for analysis
                - orns: [batch, 78]
                - pns: [batch, 841]
                - kcs_dense: [batch, 608] before sparsity
                - kcs_sparse: [batch, 608] after sparsity
        """
        batch_size = features.shape[0]

        # Extract test profile (ORN input)
        orns = features[:, self.test_profile_indices]  # [batch, 78]

        # ORN → PN (fixed connectivity)
        pn_input = torch.matmul(orns, self.orn_pn_W)  # [batch, 841]

        # PN dynamics (recurrent)
        if pn_hidden is None:
            pn_hidden = self.init_hidden(batch_size)

        pn_hidden_next = self.pn_gru(pn_input, pn_hidden)  # [batch, 841]

        # Apply PN gain/bias (Tier 1+)
        pns = pn_hidden_next * self.pn_gain + self.pn_bias  # [batch, 841]
        pns = F.relu(pns)

        # PN → KC (fixed connectivity)
        kcs_dense = torch.matmul(pns, self.pn_kc_W)  # [batch, 608]

        # Apply KC gain/bias (Tier 1+)
        kcs_dense = kcs_dense * self.kc_gain + self.kc_bias

        # KC sparsity (top-k)
        kcs_sparse = self.apply_kc_sparsity(kcs_dense)  # [batch, 608]

        # Readout: KC → MBON
        logits = self.readout(kcs_sparse)  # [batch, 1]

        # Collect activations for analysis
        activations = {
            'orns': orns.detach(),
            'pns': pns.detach(),
            'kcs_dense': kcs_dense.detach(),
            'kcs_sparse': kcs_sparse.detach(),
        }

        return logits, pn_hidden_next, activations

    def apply_kc_sparsity(self, kcs: torch.Tensor) -> torch.Tensor:
        """
        Apply top-k sparsity to KC activations.

        Decision: Top-k over ReLU for simplicity and biological match.
        Evidence: KCs show winner-take-all dynamics (Perez-Orive et al., Science 2002).
        Implementation: ReLU → top-k selection → zero out others.

        Args:
            kcs: [batch, n_kcs] dense KC activations

        Returns:
            kcs_sparse: [batch, n_kcs] with only top-k retained
        """
        kcs_relu = F.relu(kcs)  # Enforce non-negative

        # Top-k selection
        topk_values, topk_indices = torch.topk(
            kcs_relu, k=self.kc_sparsity_k, dim=1
        )

        # Create sparse tensor (zero everywhere except top-k)
        kcs_sparse = torch.zeros_like(kcs_relu)
        kcs_sparse.scatter_(1, topk_indices, topk_values)

        return kcs_sparse

    def forward_sequence(
        self,
        features_seq: torch.Tensor,
        reset_hidden: bool = True,
    ) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]]]:
        """
        Forward pass through a sequence of trials (single fly).

        Args:
            features_seq: [seq_len, feature_dim] sequence of trials
            reset_hidden: If True, reset PN hidden state at start

        Returns:
            logits_seq: [seq_len, 1] predictions for each trial
            activations_seq: List of activation dicts for each trial
        """
        seq_len = features_seq.shape[0]
        logits_list = []
        activations_list = []

        # Initialize or maintain hidden state
        if reset_hidden:
            pn_hidden = self.init_hidden(batch_size=1)
        else:
            pn_hidden = None

        # Process sequence trial-by-trial
        for t in range(seq_len):
            features_t = features_seq[t:t+1]  # [1, feature_dim]
            logits_t, pn_hidden, activations_t = self.forward(features_t, pn_hidden)
            logits_list.append(logits_t)
            activations_list.append(activations_t)

        logits_seq = torch.cat(logits_list, dim=0)  # [seq_len, 1]
        return logits_seq, activations_list

    def compute_sparsity_stats(self, kcs_sparse: torch.Tensor) -> Dict[str, float]:
        """Compute sparsity statistics for logging."""
        active = (kcs_sparse > 0).float()
        frac_active = active.mean().item()
        n_active = active.sum(dim=1).mean().item()
        return {
            'kc_frac_active': frac_active,
            'kc_n_active_mean': n_active,
        }

    def get_config(self) -> Dict:
        """Return model configuration for serialization."""
        return {
            'n_orns': self.n_orns,
            'n_pns': self.n_pns,
            'n_kcs': self.n_kcs,
            'constraint_tier': self.constraint_tier,
            'kc_sparsity_k': self.kc_sparsity_k,
            'pn_hidden_init': self.pn_hidden_init,
            'connectivity_metadata': self.connectivity_metadata,
        }
