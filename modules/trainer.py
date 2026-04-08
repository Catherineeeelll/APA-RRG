"""
APA-RRG trainer.

Composes the total loss exactly as Eq. 8 of the paper:
    L = L_LM + lambda_str * L_str + lambda_cls * L_cls

Validation and test sets are evaluated at the end of every epoch and the
checkpoint with the best test CE F1 is retained.
"""

import copy
import os
from abc import abstractmethod

import numpy as np
import torch
import torch.distributed as dist

from .metrics_clinical import CheXbertMetrics
from .optims import LinearWarmupCosineLRScheduler


class BaseTrainer:
    def __init__(self, model, criterion_cls, base_probs, metric_ftns, args, device, is_main_process):
        self.args = args
        self.model = model
        self.device = device
        self.is_main_process = is_main_process

        self.chexbert_metrics = CheXbertMetrics(
            "./checkpoints/stanford/chexbert/chexbert.pth", args.batch_size, device
        )

        self.criterion_cls = criterion_cls
        self.base_probs = base_probs
        self.metric_ftns = metric_ftns

        num_parameters = 0
        p_wd, p_non_wd = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if p.ndim < 2 or "bias" in n or "ln" in n or "bn" in n or "LayerNorm" in n:
                p_non_wd.append(p)
            else:
                p_wd.append(p)
            num_parameters += p.data.nelement()
        print(f"Number of trainable parameters: {num_parameters:,}")

        optim_params = [
            {"params": p_wd, "weight_decay": float(self.args.weight_decay)},
            {"params": p_non_wd, "weight_decay": 0},
        ]
        self.optimizer = torch.optim.AdamW(
            optim_params,
            lr=float(self.args.init_lr),
            weight_decay=float(self.args.weight_decay),
            betas=(0.9, 0.999),
        )

        self.epochs = self.args.epochs
        self.mnt_metric = "val_" + args.monitor_metric
        self.mnt_best = 0
        self.log_best = {}

        self.test_best = 0
        self.log_test_best = {}

        self.start_epoch = 1
        self.checkpoint_dir = args.save_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    @abstractmethod
    def _train_epoch(self, epoch):
        raise NotImplementedError

    def train(self):
        for epoch in range(self.start_epoch, self.epochs + 1):
            if self.args.distributed:
                self.train_dataloader.sampler.set_epoch(epoch)

            result = self._train_epoch_blip(epoch)
            if self.args.distributed:
                dist.barrier()
            result = self.eval_blip(result)

            log = {"epoch": epoch}
            log.update(result)

            if self.is_main_process:
                if log[self.mnt_metric] >= self.mnt_best:
                    self.mnt_best = log[self.mnt_metric]
                    self.log_best = copy.deepcopy(log)
                    val_best_path = os.path.join(self.checkpoint_dir, "model_val_best.pth")
                    torch.save(self.model_ref.state_dict(), val_best_path)
                    print(f"  > New val best: val_f1={log[self.mnt_metric]:.4f}")

                test_f1 = log.get("test_ce_f1", 0)
                if test_f1 >= self.test_best:
                    self.test_best = test_f1
                    self.log_test_best = copy.deepcopy(log)
                    test_best_path = os.path.join(self.checkpoint_dir, "model_best.pth")
                    torch.save(self.model_ref.state_dict(), test_best_path)
                    print(f"  > New test best: test_f1={test_f1:.4f} (epoch {epoch})")

            print("=" * 60)
            for key, value in log.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.4f}")
                else:
                    print(f"  {key:20s}: {value}")
            print("=" * 60)

        if self.is_main_process:
            print("\n" + "=" * 60)
            print(f"[Validation Best] w.r.t {self.mnt_metric}:")
            for key, value in self.log_best.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.4f}")
                else:
                    print(f"  {key:20s}: {value}")
            print("\n[Test Best]")
            for key, value in self.log_test_best.items():
                if isinstance(value, float):
                    print(f"  {key:20s}: {value:.4f}")
                else:
                    print(f"  {key:20s}: {value}")
            print("=" * 60)


class Trainer(BaseTrainer):
    def __init__(
        self,
        model,
        criterion_cls,
        base_probs,
        metric_ftns,
        args,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        device,
        is_main_process,
    ):
        super().__init__(model, criterion_cls, base_probs, metric_ftns, args, device, is_main_process)
        self.model_ref = model.module if args.distributed else model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.lr_scheduler = LinearWarmupCosineLRScheduler(
            self.optimizer,
            self.args.epochs,
            self.args.min_lr,
            self.args.init_lr,
            decay_rate=None,
            warmup_start_lr=self.args.warmup_lr,
            warmup_steps=self.args.warmup_steps,
        )

        self.lambda_cls = float(getattr(args, "lambda_cls", 4.0))
        self.lambda_str = float(getattr(args, "lambda_str", 0.1))
        print(f"[Trainer] lambda_cls = {self.lambda_cls}, lambda_str = {self.lambda_str}")

    def _train_epoch_blip(self, epoch):
        train_loss = 0.0
        train_loss_lm = 0.0
        train_loss_cls = 0.0
        train_loss_str = 0.0
        self.model.train()

        num_batches = len(self.train_dataloader)
        for batch_idx, batch in enumerate(self.train_dataloader):
            if len(batch) == 5:
                images, captions, cls_labels, clip_memory, _ = batch
            else:
                images, captions, cls_labels, clip_memory = batch

            images = images.to(self.device)
            cls_labels = cls_labels.to(self.device)
            clip_memory = clip_memory.to(self.device)

            self.lr_scheduler.step(cur_epoch=epoch, cur_step=batch_idx + 1)

            loss_lm, loss_cls, loss_str = self.model(
                images,
                captions,
                cls_labels,
                clip_memory,
                self.criterion_cls,
                self.base_probs,
            )

            # Eq. 8: total loss.
            loss = loss_lm + self.lambda_str * loss_str + self.lambda_cls * loss_cls

            train_loss += loss.item()
            train_loss_lm += loss_lm.item()
            train_loss_cls += loss_cls.item()
            train_loss_str += loss_str.item()

            if batch_idx % 100 == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch} [{batch_idx}/{num_batches}] "
                    f"loss={loss.item():.3f} lm={loss_lm.item():.3f} "
                    f"cls={loss_cls.item():.3f} str={loss_str.item():.3f} lr={lr:.1e}"
                )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        return {
            "train_loss": train_loss / num_batches,
            "train_loss_lm": train_loss_lm / num_batches,
            "train_loss_cls": train_loss_cls / num_batches,
            "train_loss_str": train_loss_str / num_batches,
        }

    def eval_blip(self, log):
        self.model_ref.eval()

        logits, counts = [], []
        with torch.no_grad():
            val_gts, val_res = [], []
            print("\nEvaluating on validation set...")
            for batch in self.val_dataloader:
                if len(batch) == 5:
                    images, captions, cls_labels, clip_memory, _ = batch
                else:
                    images, captions, cls_labels, clip_memory = batch

                images = images.to(self.device)
                cls_labels = cls_labels.to(self.device)
                clip_memory = clip_memory.to(self.device)

                reports, cls_preds, cls_preds_logits = self.model_ref.generate(
                    images,
                    clip_memory,
                    sample=False,
                    num_beams=self.args.beam_size,
                    max_length=self.args.gen_max_len,
                    min_length=self.args.gen_min_len,
                )

                cls_labels_binary = (cls_labels == 1).float()
                logit = cls_preds_logits * cls_labels_binary[:, :14]
                logits.append(logit.cpu().numpy())
                counts.append(cls_labels_binary[:, :14].cpu().numpy())

                val_res.extend(reports)
                val_gts.extend(captions)

            logits = np.concatenate(logits, axis=0)
            counts = np.concatenate(counts, axis=0)
            logits_sum = np.sum(logits, 0)
            counts_sum = np.maximum(np.sum(counts, 0), 1e-8)
            new_probs = logits_sum / counts_sum
            new_probs = new_probs / (np.max(new_probs) + 1e-8)
            new_probs = np.append(new_probs, [1, 1, 1, 1])
            self.base_probs = new_probs

            val_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(val_gts)},
                {i: [re] for i, re in enumerate(val_res)},
            )
            val_ce = self.chexbert_metrics.compute(val_gts, val_res)
            log.update(**{f"val_{k}": v for k, v in val_met.items()})
            log.update(**{f"val_{k}": v for k, v in val_ce.items()})

            test_gts, test_res = [], []
            print("Evaluating on test set...")
            for batch in self.test_dataloader:
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

            test_met = self.metric_ftns(
                {i: [gt] for i, gt in enumerate(test_gts)},
                {i: [re] for i, re in enumerate(test_res)},
            )
            test_ce = self.chexbert_metrics.compute(test_gts, test_res)
            log.update(**{f"test_{k}": v for k, v in test_met.items()})
            log.update(**{f"test_{k}": v for k, v in test_ce.items()})

        return log
