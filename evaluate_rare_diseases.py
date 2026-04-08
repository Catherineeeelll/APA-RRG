"""
Per-class rare-disease evaluation for APA-RRG.

Loads a trained checkpoint, runs the test split through the model, and
computes per-class precision / recall / F1 by labelling generated and
reference reports with CheXbert. Pathologies are partitioned into low-
and high-prevalence groups using the natural gap (5.5%-7.5%) in the
MIMIC-CXR training-set positive rates, following Section 4.4 of the
paper.

The output JSON file is consumed by ``visualize_rare_disease.py`` to
produce Figure 4.

Usage:
    python evaluate_rare_diseases.py \
        --load_pretrained results/apa_rrg/model_best.pth \
        --use_dap_graph --use_parc --use_apg --use_structure_loss
"""

import argparse
import json
import os

import numpy as np
import torch

# CheXpert label order (14 classes).
DISEASES = [
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
    "No Finding",
]


def analyze_disease_distribution(ann_path):
    """Print the per-disease positive rate in the training set."""
    with open(ann_path, "r") as f:
        ann = json.load(f)

    if isinstance(ann, dict) and "train" in ann:
        samples = ann["train"]
    else:
        samples = ann

    labels = np.array([s["labels"] for s in samples])
    total = len(labels)

    print("=" * 65)
    print("Disease distribution (training set)")
    print("=" * 65)

    rare_diseases, common_diseases = [], []
    for i, disease in enumerate(DISEASES):
        if i >= labels.shape[1]:
            break
        pos_count = (labels[:, i] == 1).sum()
        ratio = 100 * pos_count / total
        # Section 4.4 of the paper uses 5.5% as the natural-gap threshold.
        status = "Low" if ratio <= 5.5 else "High"
        print(f"{disease:30s}: {pos_count:6d} ({ratio:5.2f}%) [{status}]")

        if ratio <= 5.5:
            rare_diseases.append(disease)
        elif disease != "No Finding":
            common_diseases.append(disease)

    print("-" * 65)
    print(f"Low-prevalence  ({len(rare_diseases)}): {rare_diseases}")
    print(f"High-prevalence ({len(common_diseases)}): {common_diseases}")
    return rare_diseases, common_diseases


def compute_per_class_metrics(gt_labels, pred_labels):
    """Compute precision, recall, F1, and support for each pathology."""
    results = {}
    for i, disease in enumerate(DISEASES):
        if i >= gt_labels.shape[1]:
            break

        gt_pos = gt_labels[:, i] == 1
        pred_pos = pred_labels[:, i] == 1

        tp = (gt_pos & pred_pos).sum()
        fp = (~gt_pos & pred_pos).sum()
        fn = (gt_pos & ~pred_pos).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[disease] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(gt_pos.sum()),
        }
    return results


def load_checkpoint(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

    print(f"Loading checkpoint from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location="cpu")
    model_dict = model.state_dict()

    filtered = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v

    print(f"Loaded {len(filtered)}/{len(state_dict)} compatible parameters")
    model.load_state_dict(filtered, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="data/mimic_cxr/images/")
    parser.add_argument(
        "--ann_path", type=str, default="data/mimic_cxr/mimic_annotation_promptmrg.json"
    )
    parser.add_argument(
        "--dataset_name", type=str, default="mimic_cxr", choices=["mimic_cxr", "iu_xray"]
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--gen_max_len", type=int, default=150)
    parser.add_argument("--gen_min_len", type=int, default=80)
    parser.add_argument("--load_pretrained", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_k", type=int, default=21)

    parser.add_argument("--use_dap_graph", action="store_true")
    parser.add_argument("--use_parc", action="store_true")
    parser.add_argument("--use_apg", action="store_true")
    parser.add_argument("--use_structure_loss", action="store_true")

    parser.add_argument("--apg_threshold", type=float, default=0.5)
    parser.add_argument("--proto_temperature", type=float, default=0.5)
    parser.add_argument(
        "--cooccurrence_path",
        type=str,
        default="data/mimic_cxr/disease_cooccurrence.npy",
    )

    parser.add_argument("--baseline_results", type=str, default=None)
    parser.add_argument("--analyze_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    if args.dataset_name == "iu_xray":
        args.image_dir = "data/iu_xray/images/"
        args.ann_path = "data/iu_xray/iu_annotation_promptmrg.json"
        args.cooccurrence_path = "data/iu_xray/disease_cooccurrence.npy"

    rare_diseases, common_diseases = analyze_disease_distribution(args.ann_path)

    if args.analyze_only:
        return

    print("\nLoading model...")
    from transformers import BertTokenizer

    from dataset import create_dataset_test, create_loader
    from models.apg import all_region_tokens, empty_prompt
    from models.blip import blip_decoder
    from modules.metrics_clinical import CheXbertMetrics

    device = torch.device(args.device)

    bert_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_tokens(["[BLA]", "[POS]", "[NEG]", "[UNC]"])
    tokenizer.add_tokens(all_region_tokens())
    print(f"[Tokenizer] vocab_size = {len(tokenizer)}")

    prompt_temp = empty_prompt()
    model = blip_decoder(
        args,
        tokenizer,
        image_size=args.image_size,
        prompt=prompt_temp,
        bert_path=bert_model_name,
    )
    model = load_checkpoint(model, args.load_pretrained)
    model = model.to(device)
    model.eval()

    print("\nLoading test dataset...")
    test_dataset = create_dataset_test(f"generation_{args.dataset_name}", tokenizer, args)
    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[args.batch_size],
        num_workers=[args.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]
    print(f"Test samples: {len(test_dataset)}")

    chexbert_metrics = CheXbertMetrics(
        "./checkpoints/stanford/chexbert/chexbert.pth", args.batch_size, device
    )

    print(f"\nGenerating reports ({len(test_dataset)} samples)...")
    all_gt_reports, all_pred_reports = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 5:
                images, captions, cls_labels, clip_memory, _ = batch
            else:
                images, captions, cls_labels, clip_memory = batch

            images = images.to(device)
            clip_memory = clip_memory.to(device)

            reports, _, _ = model.generate(
                images,
                clip_memory,
                sample=False,
                num_beams=args.beam_size,
                max_length=args.gen_max_len,
                min_length=args.gen_min_len,
            )

            all_gt_reports.extend(captions)
            all_pred_reports.extend(reports)

            if (batch_idx + 1) % 50 == 0:
                print(f"  [{batch_idx + 1}/{len(test_loader)}]")

    print("\nExtracting CheXbert labels...")

    def extract_labels_batched(reports, batch_size=32):
        all_labels = []
        for i in range(0, len(reports), batch_size):
            batch = reports[i : i + batch_size]
            with torch.no_grad():
                labels = chexbert_metrics.chexbert(batch)
            all_labels.append(labels.cpu())
        return torch.cat(all_labels, dim=0).numpy()

    gt_labels = extract_labels_batched(all_gt_reports)
    pred_labels = extract_labels_batched(all_pred_reports)

    gt_binary = (gt_labels == 1).astype(int)
    pred_binary = (pred_labels == 1).astype(int)

    per_class = compute_per_class_metrics(gt_binary, pred_binary)

    print("\n" + "=" * 85)
    print("Per-class evaluation results")
    print("=" * 85)
    print(
        f"{'Disease':<32s} {'Group':<8s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Support':>10s}"
    )
    print("-" * 85)

    rare_f1s, common_f1s = [], []
    rare_recalls, common_recalls = [], []
    for disease in DISEASES:
        if disease not in per_class:
            continue
        m = per_class[disease]
        is_rare = disease in rare_diseases
        group = "Low" if is_rare else "High"
        print(
            f"{disease:<32s} {group:<8s} "
            f"{m['precision']:>10.4f} {m['recall']:>10.4f} "
            f"{m['f1']:>10.4f} {m['support']:>10d}"
        )
        if is_rare:
            rare_f1s.append(m["f1"])
            rare_recalls.append(m["recall"])
        elif disease != "No Finding":
            common_f1s.append(m["f1"])
            common_recalls.append(m["recall"])

    print("-" * 85)

    avg_rare_f1 = float(np.mean(rare_f1s)) if rare_f1s else 0.0
    avg_common_f1 = float(np.mean(common_f1s)) if common_f1s else 0.0
    avg_rare_recall = float(np.mean(rare_recalls)) if rare_recalls else 0.0
    avg_common_recall = float(np.mean(common_recalls)) if common_recalls else 0.0

    print("\nSummary")
    print("-" * 45)
    print(f"  Low-prevalence  ({len(rare_f1s)} classes):")
    print(f"    avg F1     = {avg_rare_f1:.4f}")
    print(f"    avg Recall = {avg_rare_recall:.4f}")
    print(f"  High-prevalence ({len(common_f1s)} classes):")
    print(f"    avg F1     = {avg_common_f1:.4f}")
    print(f"    avg Recall = {avg_common_recall:.4f}")

    if args.baseline_results and os.path.exists(args.baseline_results):
        with open(args.baseline_results, "r") as f:
            baseline_data = json.load(f)
        baseline_per_class = baseline_data.get("per_class", {})

        print("\n" + "=" * 85)
        print("Comparison with baseline")
        print("=" * 85)
        print(f"{'Disease':<32s} {'Group':<8s} {'Baseline':>10s} {'Ours':>10s} {'Delta':>10s}")
        print("-" * 85)

        rare_imp, common_imp = [], []
        for disease in DISEASES:
            if disease not in per_class or disease not in baseline_per_class:
                continue
            b_f1 = baseline_per_class[disease]["f1"]
            m_f1 = per_class[disease]["f1"]
            diff = m_f1 - b_f1
            is_rare = disease in rare_diseases
            group = "Low" if is_rare else "High"
            sign = "+" if diff >= 0 else ""
            print(
                f"{disease:<32s} {group:<8s} "
                f"{b_f1:>10.4f} {m_f1:>10.4f} {sign}{diff:>9.4f}"
            )
            if is_rare:
                rare_imp.append(diff)
            elif disease != "No Finding":
                common_imp.append(diff)

        print("-" * 85)
        avg_rare_imp = float(np.mean(rare_imp)) if rare_imp else 0.0
        avg_common_imp = float(np.mean(common_imp)) if common_imp else 0.0
        print(f"Low-prevalence  avg F1 delta: {'+' if avg_rare_imp >= 0 else ''}{avg_rare_imp:.4f}")
        print(f"High-prevalence avg F1 delta: {'+' if avg_common_imp >= 0 else ''}{avg_common_imp:.4f}")

    results = {
        "per_class": per_class,
        "rare_diseases": rare_diseases,
        "common_diseases": common_diseases,
        "avg_rare_f1": avg_rare_f1,
        "avg_common_f1": avg_common_f1,
        "avg_rare_recall": avg_rare_recall,
        "avg_common_recall": avg_common_recall,
    }
    output_path = args.load_pretrained.replace(".pth", "_rare_eval.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
