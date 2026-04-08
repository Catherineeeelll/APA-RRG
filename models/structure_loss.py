"""
Structure-aware auxiliary loss.

Implements the L_str term in Eq. 8 of the paper. The decoder hidden states
are mean-pooled and projected to six anatomical region logits, then
supervised with binary cross-entropy against region-level targets derived
from per-disease annotations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .apg import ANATOMY_REGIONS, REGION_IDS


class StructureLoss(nn.Module):
    """Region-level structural supervision over decoder hidden states.

    Args:
        decoder_hidden_dim: Hidden size of the BERT decoder (768 for base).
        num_regions: Number of anatomical regions used by APG (six).
    """

    def __init__(self, decoder_hidden_dim: int = 768, num_regions: int = 6):
        super().__init__()
        self.num_regions = num_regions
        self.region_proj = nn.Linear(decoder_hidden_dim, num_regions)
        self._region_to_indices = [ANATOMY_REGIONS[r] for r in REGION_IDS]

    def _region_targets(self, cls_labels: torch.Tensor) -> torch.Tensor:
        """Build [B, num_regions] binary targets from per-disease labels.

        A region is positive if any disease in that region carries the
        positive label (PromptMRG label 1).
        """
        batch_size = cls_labels.size(0)
        device = cls_labels.device
        targets = torch.zeros(batch_size, self.num_regions, device=device)
        for r_idx, indices in enumerate(self._region_to_indices):
            region_labels = cls_labels[:, indices]
            is_positive = (region_labels == 1).any(dim=-1).float()
            targets[:, r_idx] = is_positive
        return targets

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        cls_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the structural BCE loss.

        Args:
            decoder_hidden: Tensor of shape [B, T, decoder_hidden_dim].
            cls_labels: Tensor of shape [B, N] with PromptMRG label codes.
        """
        pooled = decoder_hidden.mean(dim=1)
        region_logits = self.region_proj(pooled)
        targets = self._region_targets(cls_labels)
        return F.binary_cross_entropy_with_logits(region_logits, targets)
