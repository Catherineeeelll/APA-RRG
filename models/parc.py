"""
Pathology-Anchored Representation Calibration (PARC).

Implements Section 3.4 of the paper. Maintains a learnable bank of N
disease prototypes, computes a patient-specific composite prototype via
temperature-scaled softmax over predicted disease probabilities, and
fuses it with the current feature through a gated residual.

Equations:
    Eq. 4:  pi_tilde_i = exp(pi_i / tau_p) / sum_j exp(pi_j / tau_p)
            p_w        = sum_i pi_tilde_i * p_i
    Eq. 5:  h_cal      = (1 - g) * h + g * p_w,
            g          = sigmoid( W_g [h ; p_w] )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PathologyAnchoredCalibration(nn.Module):
    """PARC as described in Section 3.4 of the paper.

    Args:
        feature_dim: Dimensionality of the visual feature h.
        num_prototypes: Number of disease prototypes (matches num_diseases).
        temperature: Softmax temperature tau_p (Eq. 4). The paper sets it
            to 0.5 to sharpen the weighting toward dominant predictions.
        gate_init_bias: Initial bias for the fusion gate. A negative value
            keeps the early gate close to zero so the original feature is
            preserved during the first epochs.
    """

    def __init__(
        self,
        feature_dim: int = 512,
        num_prototypes: int = 18,
        temperature: float = 0.5,
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.temperature = temperature

        # Learnable disease prototypes p_i.
        self.prototypes = nn.Parameter(
            torch.randn(num_prototypes, feature_dim) * 0.02
        )

        # Light prototype transform for representation smoothness.
        self.prototype_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(),
        )

        # Fusion gate W_g (Eq. 5). Initialized to favor the original feature.
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, gate_init_bias)

    def forward(
        self,
        features: torch.Tensor,
        disease_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Tensor of shape [B, feature_dim] (h in Eq. 5).
            disease_probs: Tensor of shape [B, num_prototypes] holding the
                per-disease probability predictions pi.

        Returns:
            h_cal: Calibrated feature of shape [B, feature_dim].
        """
        # Eq. 4: temperature-scaled softmax over disease probabilities.
        weights = F.softmax(disease_probs / self.temperature, dim=-1)

        # Eq. 4: composite prototype as a weighted combination.
        p_w = torch.matmul(weights, self.prototypes)
        p_w = self.prototype_transform(p_w)

        # Eq. 5: gated fusion.
        gate = torch.sigmoid(self.gate(torch.cat([features, p_w], dim=-1)))
        h_cal = (1.0 - gate) * features + gate * p_w
        return h_cal
