"""
APA-RRG evaluation entry point.

Loads a trained APA-RRG checkpoint and runs the test split through the
NLG and Clinical Efficacy metric pipelines.
"""

import argparse
import os

import numpy as np
import torch
from torch import nn

from dataset import create_dataset_test, create_loader
from models.apg import all_region_tokens, empty_prompt
from models.blip import blip_decoder
from modules import utils
from modules.metrics import compute_scores
from modules.tester import Tester

STATE_TOKENS = ["[BLA]", "[POS]", "[NEG]", "[UNC]"]


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

    parser.add_argument(
        "--load_pretrained",
        type=str,
        required=True,
        help="Path to the trained APA-RRG checkpoint.",
    )

    # Generation settings.
    parser.add_argument("--beam_size", type=int, default=3)
    parser.add_argument("--gen_max_len", type=int, default=150)
    parser.add_argument("--gen_min_len", type=int, default=80)

    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results/apa_rrg",
        help="Directory used to dump generated reports.",
    )
    parser.add_argument("--epochs", type=int, default=1)

    parser.add_argument("--seed", type=int, default=9233)
    parser.add_argument("--distributed", default=False, type=bool)
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--device", default="cuda")

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

    args = parser.parse_args()

    if args.dataset_name == "iu_xray":
        args.image_dir = "data/iu_xray/images/"
        args.ann_path = "data/iu_xray/iu_annotation_promptmrg.json"
        args.cooccurrence_path = "data/iu_xray/disease_cooccurrence.npy"

    return args


def setup_tokenizer():
    """Build the BERT tokenizer extended with PromptMRG state tokens and
    APG region tokens. The set of added tokens must match the one used by
    main_train.py so that the checkpoint vocabulary aligns.
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


def load_checkpoint(model, pretrained_path):
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrained_path}")

    print(f"[Loading] Checkpoint from {pretrained_path}")
    state_dict = torch.load(pretrained_path, map_location="cpu")
    model_dict = model.state_dict()

    filtered = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            filtered[k] = v

    missing_in_ckpt = [k for k in model_dict if k not in state_dict]
    missing_in_model = [k for k in state_dict if k not in model_dict]

    print(f"[Loading] Compatible: {len(filtered)}/{len(state_dict)}")
    if missing_in_ckpt:
        print(f"[Loading] Parameters missing in checkpoint: {len(missing_in_ckpt)}")
    if missing_in_model:
        print(f"[Loading] Parameters missing in model:      {len(missing_in_model)}")

    model.load_state_dict(filtered, strict=False)
    return model


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("APA-RRG Evaluation")
    print("=" * 70)
    print(f"  Dataset:    {args.dataset_name}")
    print(f"  Checkpoint: {args.load_pretrained}")
    print("[Modules]")
    print(f"  DAP-G:          {args.use_dap_graph}")
    print(f"  PARC:           {args.use_parc}")
    print(f"  APG:            {args.use_apg}")
    print(f"  Structure loss: {args.use_structure_loss}")
    print("=" * 70 + "\n")

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    tokenizer, bert_model_name = setup_tokenizer()

    print("[Creating] Test dataset...")
    test_dataset = create_dataset_test(f"generation_{args.dataset_name}", tokenizer, args)
    print(f"[Dataset] Test samples: {len(test_dataset)}")

    test_dataloader = create_loader(
        [test_dataset],
        [None],
        batch_size=[args.batch_size],
        num_workers=[args.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]

    print("\n[Building] Model...")
   
    prompt_temp = empty_prompt() + " ".join([STATE_TOKENS[0]] * 18) + " "
    model = blip_decoder(
        args,
        tokenizer,
        image_size=args.image_size,
        prompt=prompt_temp,
        bert_path=bert_model_name,
    )
    model = load_checkpoint(model, args.load_pretrained)

    criterion_cls = nn.CrossEntropyLoss()
    metrics = compute_scores

    model = model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )

    tester = Tester(model, criterion_cls, metrics, args, device, test_dataloader)
    log = tester.test_blip()

    print("\n" + "=" * 50)
    print(f"Test results on {args.dataset_name.upper()}")
    print("=" * 50)
    print("\n[NLG Metrics]")
    for key in ["test_BLEU_1", "test_BLEU_4", "test_METEOR", "test_ROUGE_L"]:
        if key in log:
            print(f"  {key.replace('test_', ''):12s}: {log[key]:.4f}")
    print("\n[Clinical Efficacy]")
    for key in ["test_ce_precision", "test_ce_recall", "test_ce_f1"]:
        if key in log:
            print(f"  {key.replace('test_ce_', ''):12s}: {log[key]:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
