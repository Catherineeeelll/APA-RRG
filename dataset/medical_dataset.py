"""
Medical dataset for APA-RRG.

The training prompt is constructed from ground-truth disease labels and
follows the Anatomy-Aware Prompt Generation scheme defined in
models/apg.py. Each sample is prefixed with six region tokens of the form
[L_l:POS] / [L_l:NEG] before the actual report text.
"""

import json
import os

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

from models.apg import build_prompt_from_labels
from .utils import my_pre_caption

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class generation_train(Dataset):
    """Training split with ground-truth-driven APG prompts."""

    def __init__(
        self,
        transform,
        image_root,
        ann_root,
        tokenizer,
        max_words=100,
        dataset="mimic_cxr",
        args=None,
    ):
        with open(os.path.join(ann_root), "r") as f:
            self.annotation = json.load(f)
        self.ann = self.annotation["train"]
        self.transform = transform
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.max_words = max_words
        self.dataset = dataset
        self.args = args

        clip_path = "./data/mimic_cxr/clip_text_features.json"
        if dataset == "iu_xray" and os.path.exists("./data/iu_xray/clip_text_features.json"):
            clip_path = "./data/iu_xray/clip_text_features.json"
        with open(clip_path, "r") as f:
            self.clip_features = np.array(json.load(f))

        print(f"[Dataset] Train samples: {len(self.ann)}")
        print("[Dataset] Using APG region-aware prompts (six tokens)")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]

        image_path = ann["image_path"]
        img_full_path = os.path.join(self.image_root, image_path[0])
        try:
            image = Image.open(img_full_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: image not found: {img_full_path}")
            return self.__getitem__((index + 1) % len(self.ann))
        image = self.transform(image)

        cls_labels = ann["labels"]

        # APG prompt from ground-truth labels (six region tokens).
        prompt = build_prompt_from_labels(cls_labels)
        caption = prompt + my_pre_caption(ann["report"], self.max_words)

        cls_labels = torch.from_numpy(np.array(cls_labels)).long()
        clip_indices = ann["clip_indices"][: self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, caption, cls_labels, clip_memory, []


class generation_eval(Dataset):
    """Validation / test split. Inference prompts are produced by the model."""

    def __init__(
        self,
        transform,
        image_root,
        ann_root,
        tokenizer,
        max_words=100,
        split="val",
        dataset="mimic_cxr",
        args=None,
    ):
        with open(os.path.join(ann_root), "r") as f:
            self.annotation = json.load(f)
        if dataset == "mimic_cxr":
            self.ann = self.annotation[split]
        else:
            self.ann = self.annotation
        self.transform = transform
        self.max_words = max_words
        self.image_root = image_root
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.args = args

        clip_path = "./data/mimic_cxr/clip_text_features.json"
        if dataset == "iu_xray" and os.path.exists("./data/iu_xray/clip_text_features.json"):
            clip_path = "./data/iu_xray/clip_text_features.json"
        with open(clip_path, "r") as f:
            self.clip_features = np.array(json.load(f))

        print(f"[Dataset] {split} samples: {len(self.ann)}")

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        image_path = ann["image_path"]
        image = Image.open(os.path.join(self.image_root, image_path[0])).convert("RGB")
        image = self.transform(image)

        caption = my_pre_caption(ann["report"], self.max_words)
        cls_labels = ann["labels"]
        cls_labels = torch.from_numpy(np.array(cls_labels))
        clip_indices = ann["clip_indices"][: self.args.clip_k]
        clip_memory = self.clip_features[clip_indices]
        clip_memory = torch.from_numpy(clip_memory).float()

        return image, caption, cls_labels, clip_memory, []
