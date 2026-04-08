"""
APA-RRG tester.

Runs the test split through the trained model and computes both NLG and
Clinical Efficacy metrics. Optionally dumps the generated reports to the
save directory for downstream qualitative analysis.
"""

import json
import logging
import os
import time
from abc import abstractmethod

import torch

from .metrics_clinical import CheXbertMetrics


class BaseTester:
    def __init__(self, model, criterion_cls, metric_ftns, args, device):
        self.args = args
        self.model = model
        self.device = device

        self.chexbert_metrics = CheXbertMetrics(
            "./checkpoints/stanford/chexbert/chexbert.pth",
            args.batch_size,
            device,
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        self.logger = logging.getLogger(__name__)

        self.criterion_cls = criterion_cls
        self.metric_ftns = metric_ftns
        self.save_dir = getattr(self.args, "save_dir", None)

    @abstractmethod
    def test(self):
        raise NotImplementedError


class Tester(BaseTester):
    def __init__(self, model, criterion_cls, metric_ftns, args, device, test_dataloader):
        super().__init__(model, criterion_cls, metric_ftns, args, device)
        self.test_dataloader = test_dataloader

    def test_blip(self):
        """Run inference on the test split and return the metric log."""
        if self.args.distributed:
            self.model.module.eval()
        else:
            self.model.eval()

        self.model_ref = self.model.module if self.args.distributed else self.model

        log = {}
        test_gts, test_res = [], []
        total_batches = len(self.test_dataloader)

        print(f"\nRunning inference on {total_batches} batches...")
        start_time = time.time()

        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_dataloader):
                if len(batch) == 5:
                    images, captions, cls_labels, clip_memory, _ = batch
                else:
                    images, captions, cls_labels, clip_memory = batch

                images = images.to(self.device)
                clip_memory = clip_memory.to(self.device)

                reports, _, _ = self.model_ref.generate(
                    images,
                    clip_memory,
                    sample=False,
                    num_beams=self.args.beam_size,
                    max_length=self.args.gen_max_len,
                    min_length=self.args.gen_min_len,
                )

                test_res.extend(reports)
                test_gts.extend(captions)

                if (batch_idx + 1) % 50 == 0 or batch_idx == total_batches - 1:
                    elapsed = time.time() - start_time
                    eta = elapsed / (batch_idx + 1) * (total_batches - batch_idx - 1)
                    print(
                        f"  [{batch_idx + 1}/{total_batches}] "
                        f"elapsed {elapsed:.1f}s, eta {eta:.1f}s"
                    )

        print("\nComputing NLG metrics...")
        test_met = self.metric_ftns(
            {i: [gt] for i, gt in enumerate(test_gts)},
            {i: [re] for i, re in enumerate(test_res)},
        )

        print("Computing Clinical Efficacy metrics...")
        test_ce = self.chexbert_metrics.compute(test_gts, test_res)

        log.update(**{f"test_{k}": v for k, v in test_met.items()})
        log.update(**{f"test_{k}": v for k, v in test_ce.items()})

        if self.save_dir:
            self._save_reports(test_gts, test_res)

        total_time = time.time() - start_time
        print(f"\nDone. Total time: {total_time:.1f}s")
        return log

    def _save_reports(self, gts, preds):
        os.makedirs(self.save_dir, exist_ok=True)
        records = [
            {"id": i, "ground_truth": gt, "generated": pred}
            for i, (gt, pred) in enumerate(zip(gts, preds))
        ]
        out_path = os.path.join(self.save_dir, "generated_reports.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"Generated reports saved to: {out_path}")
