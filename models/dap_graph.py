"""
Dynamic Anatomy-Pathology Graph (DAP-G).

Implements Section 3.3 of the paper. Constructs an 18-node disease graph
whose adjacency is dynamically inferred from image-conditioned disease
representations and balanced against a static co-occurrence prior through
a vision-guided gating mechanism.

Equations:
    Eq. 1:  h_i^(0) = MLP([e_i ; p_i])
    Eq. 2:  A_dyn   = Softmax( H W_Q (H W_K)^T / sqrt(d_h) )
    Eq. 3:  A       = g * A_dyn + (1 - g) * A_static,
            g       = sigmoid( MLP(v_global) )
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicAnatomyPathologyGraph(nn.Module):
    """DAP-G as described in Section 3.3 of the paper.

    Args:
        num_diseases: Number of disease nodes (14 CheXpert + 4 auxiliary = 18).
        hidden_dim:   Node embedding dimensionality d_h.
        visual_dim:   Dimensionality of the global visual feature v_global.
        cooccurrence_path: Path to a precomputed (num_diseases x num_diseases)
            co-occurrence matrix saved as .npy. If the file is missing,
            an identity matrix is used as a safe fallback.
    """

    def __init__(
        self,
        num_diseases: int = 18,
        hidden_dim: int = 512,
        visual_dim: int = 2048,
        cooccurrence_path: str = "data/mimic_cxr/disease_cooccurrence.npy",
    ):
        super().__init__()
        self.num_diseases = num_diseases
        self.hidden_dim = hidden_dim

        # Per-disease learnable embedding e_i (Eq. 1).
        self.disease_embedding = nn.Parameter(
            torch.randn(num_diseases, hidden_dim) * 0.02
        )

        # MLP that fuses [e_i ; p_i] into the initial node feature h_i^(0).
        # Concatenated input is (hidden_dim + 4); output is hidden_dim.
        self.node_init_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Query / Key projections for the dynamic adjacency (Eq. 2).
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Vision-guided scalar gate g (Eq. 3).
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.visual_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Two-layer GCN with residual connections (Section 3.3 / 4.1.3).
        self.gcn_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(2)]
        )
        self.gcn_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(2)]
        )

        # Output projection that yields the global graph feature h_graph
        # before mean-pooling. Kept as a light MLP for representation
        # smoothness (does not change the equation).
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Residual mixing coefficient for h_graph injection. Learned end to
        # end and converges to a moderate value during training.
        self.scale = nn.Parameter(torch.tensor(0.1))

        # Static co-occurrence prior A_static (Eq. 3).
        self._load_static_adjacency(cooccurrence_path)

        self._init_weights()

    def _load_static_adjacency(self, cooccurrence_path: str) -> None:
        if os.path.exists(cooccurrence_path):
            adj = np.load(cooccurrence_path).astype(np.float32)
            if adj.shape != (self.num_diseases, self.num_diseases):
                raise ValueError(
                    f"Expected co-occurrence matrix of shape "
                    f"({self.num_diseases}, {self.num_diseases}), "
                    f"got {adj.shape}."
                )
            print(f"[DAP-G] Loaded static prior from {cooccurrence_path}")
        else:
            print(
                f"[DAP-G] Warning: {cooccurrence_path} not found. "
                "Falling back to identity prior."
            )
            adj = np.eye(self.num_diseases, dtype=np.float32)
        self.register_buffer("static_adj", torch.from_numpy(adj))

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        disease_probs: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            disease_probs: Tensor of shape [B, N, 4] containing the softmax
                probabilities over four states (blank, positive, negative,
                uncertain) for each disease node.
            visual_features: Tensor of shape [B, visual_dim] holding the
                global visual representation v_global.

        Returns:
            h_graph: Tensor of shape [B, hidden_dim] holding the pooled
                graph representation.
        """
        batch_size = disease_probs.size(0)

        # Eq. 1: node initialization with concat-MLP.
        e = self.disease_embedding.unsqueeze(0).expand(batch_size, -1, -1)
        h0 = self.node_init_mlp(torch.cat([e, disease_probs], dim=-1))

        # Eq. 2: dynamic adjacency from scaled dot-product attention.
        q = self.query_proj(h0)
        k = self.key_proj(h0)
        scale = self.hidden_dim ** 0.5
        attn = torch.bmm(q, k.transpose(1, 2)) / scale
        a_dyn = F.softmax(attn, dim=-1)

        # Eq. 3: vision-guided gated fusion with the static prior.
        v = self.visual_proj(visual_features)
        g = torch.sigmoid(self.visual_gate_mlp(v))
        g = g.view(batch_size, 1, 1)
        a_static = self.static_adj.unsqueeze(0).expand(batch_size, -1, -1)
        adj = g * a_dyn + (1.0 - g) * a_static

        # Symmetric degree normalization for stable propagation.
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        deg_inv_sqrt = deg.pow(-0.5)
        adj_norm = deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)

        # Two-layer GCN with residual connections.
        h = h0
        for gcn, norm in zip(self.gcn_layers, self.gcn_norms):
            h_msg = torch.bmm(adj_norm, h)
            h_msg = gcn(h_msg)
            h_msg = F.relu(h_msg)
            h = norm(h + h_msg)

        # Mean pooling over disease nodes (Section 3.3).
        h_graph = h.mean(dim=1)
        h_graph = self.output_proj(h_graph)
        return h_graph
