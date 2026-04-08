"""
Anatomy-Aware Prompt Generation (APG).

Implements Section 3.5 of the paper. Aggregates per-disease classification
probabilities into six anatomical regions and emits one discrete prompt
token per region.

Equations:
    Eq. 6:  s_bar_l = (1 / |D_l|) * sum_{i in D_l} s_i
    Eq. 7:  t_l     = [L_l:POS] if s_bar_l > tau else [L_l:NEG]

Region definitions follow the CheXpert label schema and clinical
conventions described in Section 3.5. The Normal (No Finding) region is
included in the prompt for symmetry but is not expected to produce a
dedicated narrative segment.
"""

from typing import List

import torch


# Mapping from anatomical region id (L1..L6) to CheXpert disease indices.
# CheXpert label order: 0 Enlarged Cardiomediastinum, 1 Cardiomegaly,
# 2 Lung Opacity, 3 Lung Lesion, 4 Edema, 5 Consolidation, 6 Pneumonia,
# 7 Atelectasis, 8 Pneumothorax, 9 Pleural Effusion, 10 Pleural Other,
# 11 Fracture, 12 Support Devices, 13 No Finding.
#
# Pneumothorax (8) is grouped under Pulmonary together with the other
# parenchymal pathologies, matching the grouping used in Figures 4 and 5
# of the paper.
ANATOMY_REGIONS = {
    "L1": [0, 1],                         # Cardiac
    "L2": [2, 3, 4, 5, 6, 7, 8],          # Pulmonary
    "L3": [9, 10],                        # Pleural
    "L4": [11],                           # Skeletal
    "L5": [12],                           # Devices
    "L6": [13],                           # Normal (No Finding)
}

REGION_IDS = list(ANATOMY_REGIONS.keys())


def all_region_tokens() -> List[str]:
    """Return the 12 region tokens to be added to the tokenizer."""
    tokens = []
    for region in REGION_IDS:
        for status in ("POS", "NEG"):
            tokens.append(f"[{region}:{status}]")
    return tokens


def build_prompt_from_labels(cls_labels) -> str:
    """Build a region-aware prompt from ground-truth labels (training).

    A region is marked POS if at least one disease in that region carries
    the positive label, otherwise NEG. PromptMRG label encoding is used,
    so 1 denotes positive.

    Args:
        cls_labels: A length-N iterable of integer labels in {0,1,2,3}
            following the PromptMRG convention (BLA, POS, NEG, UNC).

    Returns:
        Prompt string of six space-separated tokens with a trailing space.
    """
    tokens = []
    for region in REGION_IDS:
        indices = ANATOMY_REGIONS[region]
        is_positive = any(int(cls_labels[i]) == 1 for i in indices)
        status = "POS" if is_positive else "NEG"
        tokens.append(f"[{region}:{status}]")
    return " ".join(tokens) + " "


def build_prompt_from_probs(
    pos_probs: torch.Tensor,
    threshold: float = 0.5,
) -> str:
    """Build a region-aware prompt from predicted probabilities (inference).

    Implements Eq. 6 and Eq. 7. The aggregate regional score is the mean
    of the per-disease positive probabilities within the region.

    Args:
        pos_probs: 1-D tensor of length 14 holding the predicted positive
            probability of each CheXpert pathology for one sample.
        threshold: Confidence threshold tau used in Eq. 7.

    Returns:
        Prompt string of six space-separated tokens with a trailing space.
    """
    tokens = []
    for region in REGION_IDS:
        indices = ANATOMY_REGIONS[region]
        region_score = pos_probs[indices].mean().item()
        status = "POS" if region_score > threshold else "NEG"
        tokens.append(f"[{region}:{status}]")
    return " ".join(tokens) + " "


def empty_prompt() -> str:
    """Return a placeholder prompt used to compute prompt_length at init."""
    return " ".join(f"[{r}:NEG]" for r in REGION_IDS) + " "
