"""
Precompute the disease co-occurrence matrix used as the static prior
A_static in the Dynamic Anatomy-Pathology Graph (Eq. 3).

The matrix is built from the training-set CheXpert labels in the
PromptMRG annotation file. Positive co-occurrence counts are
symmetrized, then row-normalized so that each row sums to one and the
matrix can be directly consumed as an adjacency by DAP-G.

Usage:
    python scripts/build_cooccurrence.py --dataset mimic_cxr
    python scripts/build_cooccurrence.py --dataset iu_xray
"""

import argparse
import json
import os

import numpy as np


NUM_DISEASES = 18  # 14 CheXpert + 4 auxiliary nodes.


def collect_labels(ann_path: str):
    with open(ann_path, "r") as f:
        annotation = json.load(f)
    if isinstance(annotation, dict) and "train" in annotation:
        samples = annotation["train"]
    else:
        samples = annotation
    return [s["labels"] for s in samples]


def build_cooccurrence(labels) -> np.ndarray:
    cooccur = np.zeros((NUM_DISEASES, NUM_DISEASES), dtype=np.float64)
    for lbl in labels:
        arr = np.asarray(lbl)
        # Positive entries follow the PromptMRG label convention
        # (0 BLA, 1 POS, 2 NEG, 3 UNC).
        pos_indices = np.where(arr == 1)[0]
        for i in pos_indices:
            for j in pos_indices:
                cooccur[i, j] += 1.0
    return cooccur


def normalize(cooccur: np.ndarray) -> np.ndarray:
    # Symmetrize.
    cooccur = (cooccur + cooccur.T) / 2.0
    # Avoid empty rows by adding self-loops on the diagonal.
    diag = np.diag(cooccur).copy()
    diag[diag == 0] = 1.0
    np.fill_diagonal(cooccur, diag)
    # Row normalize so each row sums to one.
    row_sums = cooccur.sum(axis=1, keepdims=True)
    row_sums = np.maximum(row_sums, 1e-8)
    return (cooccur / row_sums).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="mimic_cxr",
        choices=["mimic_cxr", "iu_xray"],
    )
    args = parser.parse_args()

    if args.dataset == "mimic_cxr":
        ann_path = "data/mimic_cxr/mimic_annotation_promptmrg.json"
        out_path = "data/mimic_cxr/disease_cooccurrence.npy"
    else:
        ann_path = "data/iu_xray/iu_annotation_promptmrg.json"
        out_path = "data/iu_xray/disease_cooccurrence.npy"

    if not os.path.exists(ann_path):
        raise FileNotFoundError(
            f"Annotation file not found: {ann_path}. "
            "Please download the PromptMRG annotation files first."
        )

    print(f"[Build] Reading labels from {ann_path}")
    labels = collect_labels(ann_path)
    print(f"[Build] Number of training samples: {len(labels)}")

    cooccur = build_cooccurrence(labels)
    cooccur = normalize(cooccur)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, cooccur)
    print(f"[Build] Saved disease co-occurrence matrix to {out_path}")
    print(f"[Build] Shape: {cooccur.shape}, dtype: {cooccur.dtype}")


if __name__ == "__main__":
    main()
