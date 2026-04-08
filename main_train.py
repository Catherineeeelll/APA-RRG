"""
APA-RRG training entry point.

Loads the BERT tokenizer (extended with the four PromptMRG state tokens
and the twelve APG region tokens), builds the model, and launches the
trainer defined in modules/trainer.py.
"""

import argparse
import json
import os

import numpy as np
import torch
from torch import nn

from dataset import create_dataset, create_loader, create_sampler
from models.apg import all_region_tokens, empty_prompt
from models.blip import blip_decoder
from modules import utils
from modules.metrics import compute_scores
from modules.trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def parse_args():
    parser = argparse.ArgumentParser()

    # Data settings.
    parser.add_argument("--image_dir", type=str, default="data/mimic_cxr/images/")
    parser.add_argument(
        "--ann_path", type=str, default="data/mimic_cxr/mimic_annotation_promptmrg.json"
    )
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument(
        "--dataset_name", type=str, default="mimic_cxr", choices=["iu_xray", "mimic_cxr"]
    )
    parser.add_argument("--threshold", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)

    # Model settings.
    parser.add_argument(
        "--load_pretrained",
        type=str,
        default="results/model_promptmrg/model_promptmrg_20240305.pth",
        help="Optional PromptMRG checkpoint used to warm-start the encoder.",
    )

    # Generation settings.
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--gen_max_len", type=int, default=150)
    parser.add_argument("--gen_min_len", type=int, default=80)

    # Training settings.
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--save_dir", type=str, default="results/apa_rrg")
    parser.add_argument("--monitor_metric", type=str, default="ce_f1")

    # Optimization.
    parser.add_argument("--init_lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=5e-6)
    parser.add_argument("--warmup_lr", type=float, default=5e-7)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=5000)

    # Distributed.
    parser.add_argument("--seed", type=int, default=9233)
    parser.add_argument("--distributed", default=False, type=bool)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")

    # Loss weights (Eq. 8).
    parser.add_argument("--lambda_cls", type=float, default=4.0)
    parser.add_argument("--lambda_str", type=float, default=0.1)

    # Memory retrieval and CheXpert classification.
    parser.add_argument("--clip_k", type=int, default=21)

    # Module switches.
    parser.add_argument("--use_dap_graph", action="store_true", help="Enable DAP-G")
    parser.add_argument("--use_parc", action="store_true", help="Enable PARC")
    parser.add_argument("--use_apg", action="store_true", help="Enable APG")
    parser.add_argument(
        "--use_structure_loss", action="store_true", help="Enable structure loss"
    )

    # Module hyperparameters.
    parser.add_argument(
        "--apg_threshold", type=float, default=0.5, help="APG confidence threshold tau."
    )
    parser.add_argument(
        "--proto_temperature", type=float, default=0.5, help="PARC temperature tau_p."
    )
    parser.add_argument(
        "--cooccurrence_path",
        type=str,
        default="data/mimic_cxr/disease_cooccurrence.npy",
        help="Path to the precomputed disease co-occurrence matrix.",
    )

    args = parser.parse_args()

    if args.dataset_name == "iu_xray":
        args.image_dir = "data/iu_xray/images/"
        args.ann_path = "data/iu_xray/iu_annotation_promptmrg.json"
        args.cooccurrence_path = "data/iu_xray/disease_cooccurrence.npy"

    return args


def setup_tokenizer():
    """Build the BERT tokenizer extended with PromptMRG state tokens and
    APG region tokens. Uses the public HuggingFace ``bert-base-uncased``
    checkpoint, which is downloaded on first use.
    """
    from transformers import BertTokenizer

    bert_model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
    tokenizer.add_tokens(["[BLA]", "[POS]", "[NEG]", "[UNC]"])
    tokenizer.add_tokens(all_region_tokens())
    print(f"[Tokenizer] vocab_size = {len(tokenizer)}")
    return tokenizer, bert_model_name


def load_pretrained_weights(model, pretrained_path):
    if not pretrained_path or not os.path.exists(pretrained_path):
        print(f"[Warning] No pretrained weights at {pretrained_path}")
        return model

    print(f"[Loading] Pretrained weights from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location="cpu")
    model_dict = model.state_dict()

    # The new model adds 12 region tokens to the tokenizer on top of the
    # 4 PromptMRG state tokens. The token embedding and the LM head bias
    # therefore have a larger first dimension than the source checkpoint.
    # We expand the source tensors row-wise so that the original
    # PromptMRG embeddings (rows 0..n_src-1, including [BLA]/[POS]/[NEG]/
    # [UNC]) are preserved while the new region-token rows fall back to
    # the freshly resized embedding values.
    expandable_keys = [
        "text_decoder.bert.embeddings.word_embeddings.weight",
        "text_decoder.cls.predictions.decoder.weight",
        "text_decoder.cls.predictions.decoder.bias",
        "text_decoder.cls.predictions.bias",
    ]
    for key in expandable_keys:
        if key in state_dict and key in model_dict:
            src = state_dict[key]
            tgt = model_dict[key]
            if src.shape == tgt.shape:
                continue
            if src.dim() != tgt.dim():
                continue
            if src.shape[0] < tgt.shape[0] and src.shape[1:] == tgt.shape[1:]:
                padded = tgt.clone()
                padded[: src.shape[0]] = src
                state_dict[key] = padded
                print(
                    f"[Loading] Padded {key}: {tuple(src.shape)} -> {tuple(tgt.shape)}"
                )

    filtered = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v

    new_keys = [k for k in model_dict if k not in state_dict]
    print(f"[Loading] Compatible: {len(filtered)}/{len(state_dict)}")
    if new_keys:
        print(f"[Loading] Newly initialized parameters: {len(new_keys)}")

    model.load_state_dict(filtered, strict=False)
    return model


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("APA-RRG Training")
    print("=" * 70)
    print("[Modules]")
    print(f"  DAP-G:          {args.use_dap_graph}")
    print(f"  PARC:           {args.use_parc}")
    print(f"  APG:            {args.use_apg}")
    print(f"  Structure loss: {args.use_structure_loss}")
    print("[Training]")
    print(f"  lambda_cls = {args.lambda_cls}, lambda_str = {args.lambda_str}")
    print(f"  init_lr    = {args.init_lr}, epochs    = {args.epochs}")
    print("=" * 70 + "\n")

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    tokenizer, bert_model_name = setup_tokenizer()

    print("[Creating] Datasets...")
    train_dataset, val_dataset, test_dataset = create_dataset(
        f"generation_{args.dataset_name}", tokenizer, args
    )
    print(
        f"[Dataset] Train: {len(train_dataset)}, "
        f"Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    base_probs_path = f"./data/{args.dataset_name}/base_probs.json"
    if os.path.exists(base_probs_path):
        with open(base_probs_path, "r") as f:
            base_probs = json.load(f)
        base_probs = np.array(base_probs) / np.max(base_probs)
    else:
        base_probs = np.ones(14)
    base_probs = np.append(base_probs, [1, 1, 1, 1])

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler(
            [train_dataset, val_dataset, test_dataset],
            [True, False, False],
            num_tasks,
            global_rank,
        )
        samplers = [samplers[0], None, None]
    else:
        samplers = [None, None, None]

    train_dataloader, val_dataloader, test_dataloader = create_loader(
        [train_dataset, val_dataset, test_dataset],
        samplers,
        batch_size=[args.batch_size] * 3,
        num_workers=[args.num_workers] * 3,
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    print("\n[Building] Model...")
    prompt_temp = empty_prompt()  # Six region tokens, used for prompt_length.
    model = blip_decoder(
        args,
        tokenizer,
        image_size=args.image_size,
        prompt=prompt_temp,
        bert_path=bert_model_name,
    )
    model = load_pretrained_weights(model, args.load_pretrained)

    criterion_cls = nn.CrossEntropyLoss()
    metrics = compute_scores

    model = model.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    trainer = Trainer(
        model,
        criterion_cls,
        base_probs,
        metrics,
        args,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        utils.is_main_process,
    )

    print("\n[Starting] Training...")
    trainer.train()


if __name__ == "__main__":
    main()
