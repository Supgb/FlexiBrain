from __future__ import annotations

import json
import math
import os
import threading
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from flexibrain.config import RunConfig
from flexibrain.data import build_downstream_dataloaders
from flexibrain.data.classification import prepare_batch_data
from flexibrain.engine.pretrainer import _to_cpu_snapshot, _write_checkpoint
from flexibrain.models import build_downstream_model
from flexibrain.utils.logging import setup_logger
from flexibrain.utils.seed import set_seed


VALID_SELECTION_METRICS = ("val_auroc", "val_loss", "f1_macro", "f1_weighted")


class DownstreamTrainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.rank = cfg.training.local_rank
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("downstream", cfg.logging.log_dir, rank=self.rank)
        self._save_thread: Optional[threading.Thread] = None

    def _wait_for_save(self) -> None:
        thread = getattr(self, "_save_thread", None)
        if thread is not None:
            thread.join()
            self._save_thread = None

    def _validate_selection_metric(self) -> str:
        metric = self.cfg.training.selection_metric
        if metric not in VALID_SELECTION_METRICS:
            raise ValueError(
                f"Unsupported selection_metric {metric!r}; expected one of {VALID_SELECTION_METRICS}"
            )
        return metric

    def build(self):
        self._validate_selection_metric()
        set_seed(self.cfg.training.seed)
        self.model = build_downstream_model(
            self.cfg.model,
            self.device,
            logger=self.logger,
            checkpoint_path=self.cfg.pretrain_checkpoint,
            from_scratch=self.cfg.from_scratch,
            use_checkpoint_config=self.cfg.use_checkpoint_config,
        )
        self.train_loader, self.val_loader, self.test_loader = build_downstream_dataloaders(self.cfg.data, self.cfg.training, rank=self.rank, world_size=self.cfg.training.world_size)
        if self.cfg.training.lr_backbone is not None or self.cfg.training.lr_head is not None:
            backbone_params = list(self.model.backbone.parameters())
            head_params = [p for n, p in self.model.named_parameters() if not n.startswith("backbone.")]
            self.optimizer = optim.AdamW([
                {"params": backbone_params, "lr": self.cfg.training.lr_backbone or self.cfg.training.lr},
                {"params": head_params, "lr": self.cfg.training.lr_head or self.cfg.training.lr},
            ], betas=(0.9, 0.999), weight_decay=self.cfg.training.weight_decay, fused=(self.device.type == "cuda"))
        else:
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.training.lr,
                betas=(0.9, 0.999),
                weight_decay=self.cfg.training.weight_decay,
                fused=(self.device.type == "cuda"),
            )
        self.use_amp = bool(self.cfg.training.use_amp and self.device.type == "cuda")
        self.amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[self.cfg.training.amp_dtype]
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp and self.amp_dtype == torch.float16)
        accum = max(1, int(self.cfg.training.grad_accumulation_steps))
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / accum))
        total_steps = max(1, steps_per_epoch * self.cfg.training.epochs)
        warmup_steps = max(0, steps_per_epoch * self.cfg.training.warmup_epochs)

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return (step + 1) / warmup_steps
            decay_steps = max(1, total_steps - warmup_steps)
            return max(0.0, (total_steps - step) / decay_steps)

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return self

    def _optimizer_step(self) -> None:
        if self.cfg.training.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
        scale_before = self.scaler.get_scale() if self.use_amp else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        stepped = (not self.use_amp) or (self.scaler.get_scale() >= scale_before)
        if stepped:
            self.scheduler.step()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        num_batches = 0
        accum = max(1, int(self.cfg.training.grad_accumulation_steps))
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(self.train_loader):
            x, meta, orig_Ts, labels, affines = prepare_batch_data(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=getattr(self, "amp_dtype", torch.float16), enabled=self.use_amp):
                logits = self.model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)
                loss = criterion(logits, labels)
            self.scaler.scale(loss / accum).backward()
            if (batch_idx + 1) % accum == 0:
                self._optimizer_step()
            total_loss += loss.detach().to(torch.float64)
            num_batches += 1
            if self.rank == 0 and (batch_idx + 1) % self.cfg.logging.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.info(
                    "Epoch %d [%d/%d] loss=%.6f lr=%.3e",
                    epoch + 1,
                    batch_idx + 1,
                    len(self.train_loader),
                    float(loss.detach().item()),
                    current_lr,
                )
        if num_batches % accum != 0:
            self._optimizer_step()
        return float(total_loss.item()) / max(1, num_batches)

    @torch.no_grad()
    def evaluate(self, loader, split_name: str) -> Dict[str, Optional[float]]:
        self.model.eval()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        probs_list, preds_list, labels_list = [], [], []
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        num_samples = 0
        for batch in loader:
            x, meta, orig_Ts, labels, affines = prepare_batch_data(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=getattr(self, "amp_dtype", torch.float16), enabled=self.use_amp):
                logits = self.model(x, meta=meta, orig_Ts=orig_Ts, affines=affines)
                loss = criterion(logits, labels)
            total_loss += loss.detach().to(torch.float64)
            num_samples += int(labels.size(0))
            probs_list.append(torch.softmax(logits.detach().float(), dim=1))
            preds_list.append(torch.argmax(logits.detach(), dim=1))
            labels_list.append(labels.detach())

        total_loss_val = float(total_loss.item())
        if probs_list:
            probs_arr = torch.cat(probs_list, dim=0).cpu().numpy()
            preds_arr = torch.cat(preds_list, dim=0).cpu().numpy()
            labels_arr = torch.cat(labels_list, dim=0).cpu().numpy()
        else:
            probs_arr = np.empty((0, 0), dtype=np.float32)
            preds_arr = np.empty((0,), dtype=np.int64)
            labels_arr = np.empty((0,), dtype=np.int64)

        auroc: Optional[float] = None
        if len(labels_arr) > 0 and probs_arr.ndim == 2:
            n_classes = probs_arr.shape[1]
            unique_labels = np.unique(labels_arr)
            try:
                if n_classes == 2:
                    if len(unique_labels) == 2:
                        auroc = float(roc_auc_score(labels_arr, probs_arr[:, 1]))
                elif n_classes > 2:
                    if len(unique_labels) == n_classes:
                        auroc = float(roc_auc_score(labels_arr, probs_arr, multi_class="ovr", average="macro"))
            except ValueError:
                auroc = None

        metrics: Dict[str, Optional[float]] = {
            "loss": total_loss_val / max(1, num_samples),
            "accuracy": float(accuracy_score(labels_arr, preds_arr)) if len(labels_arr) > 0 else 0.0,
            "balanced_accuracy": float(balanced_accuracy_score(labels_arr, preds_arr)) if len(labels_arr) > 0 else 0.0,
            "auroc": auroc,
            "precision_macro": float(precision_score(labels_arr, preds_arr, average="macro", zero_division=0)) if len(labels_arr) > 0 else 0.0,
            "recall_macro": float(recall_score(labels_arr, preds_arr, average="macro", zero_division=0)) if len(labels_arr) > 0 else 0.0,
            "f1_macro": float(f1_score(labels_arr, preds_arr, average="macro", zero_division=0)) if len(labels_arr) > 0 else 0.0,
            "f1_weighted": float(f1_score(labels_arr, preds_arr, average="weighted", zero_division=0)) if len(labels_arr) > 0 else 0.0,
        }
        if self.rank == 0:
            self.logger.info("%s metrics: %s", split_name, metrics)
        return metrics

    @staticmethod
    def _selection_candidate(val_metrics: dict, selection_metric: str) -> Tuple[int, float, Optional[float]]:
        """Return (priority, comparison_score, display_value) where larger (priority, comparison_score) is better."""
        if selection_metric == "val_auroc":
            auroc = val_metrics.get("auroc")
            if auroc is not None and math.isfinite(float(auroc)):
                val = float(auroc)
                return 1, val, val
            loss = val_metrics.get("loss")
            if loss is not None and math.isfinite(float(loss)):
                loss_val = float(loss)
                return 0, -loss_val, loss_val
            return 0, float("-inf"), None
        if selection_metric == "val_loss":
            loss = val_metrics.get("loss")
            if loss is not None and math.isfinite(float(loss)):
                loss_val = float(loss)
                return 1, -loss_val, loss_val
            return 0, float("-inf"), None
        if selection_metric in ("f1_macro", "f1_weighted"):
            val = val_metrics.get(selection_metric)
            if val is not None and math.isfinite(float(val)):
                f_val = float(val)
                return 1, f_val, f_val
            return 0, float("-inf"), None
        raise ValueError(f"Unsupported selection_metric {selection_metric!r}")

    def save(
        self,
        epoch: int,
        metrics: dict,
        is_best: bool,
        best_score: Optional[float] = None,
        selection_metric: Optional[str] = None,
        best_is_fallback: bool = False,
    ):
        if self.rank != 0:
            return
        os.makedirs(self.cfg.logging.checkpoint_dir, exist_ok=True)
        metric_name = selection_metric or self.cfg.training.selection_metric
        payload = {
            "epoch": epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.scaler.state_dict(),
            "metrics": metrics,
            "selection_metric": metric_name,
            "best_score": best_score,
            "best_is_fallback": best_is_fallback,
            "config": vars(self.cfg.model),
        }
        latest_path = os.path.join(self.cfg.logging.checkpoint_dir, "downstream_latest.pt")
        best_path = os.path.join(self.cfg.logging.checkpoint_dir, "downstream_best.pt") if is_best else None
        if getattr(self.cfg.logging, "async_checkpoint", False):
            self._wait_for_save()
            cpu_payload = _to_cpu_snapshot(payload)
            self._save_thread = threading.Thread(
                target=_write_checkpoint,
                args=(cpu_payload, latest_path, best_path),
                daemon=True,
            )
            self._save_thread.start()
        else:
            _write_checkpoint(payload, latest_path, best_path)

    def _resume(self) -> Tuple[int, Tuple[int, float], Optional[float]]:
        selection_metric = self._validate_selection_metric()
        path = self.cfg.logging.resume
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])
        next_epoch = int(checkpoint["epoch"]) + 1

        best_path = os.path.join(self.cfg.logging.checkpoint_dir, "downstream_best.pt")
        best_checkpoint = checkpoint
        if os.path.abspath(best_path) != os.path.abspath(path) and os.path.exists(best_path):
            best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)

        best_key = (-1, float("-inf"))
        best_display: Optional[float] = None

        saved_metric = checkpoint.get("selection_metric")
        saved_best = checkpoint.get("best_score")
        if saved_best is not None and (saved_metric is None or saved_metric == selection_metric):
            saved_is_fallback = bool(checkpoint.get("best_is_fallback", False))
            priority = 0 if saved_is_fallback else 1
            comp = -float(saved_best) if (selection_metric == "val_loss" or (selection_metric == "val_auroc" and saved_is_fallback)) else float(saved_best)
            best_key = (priority, comp)
            best_display = float(saved_best)
        else:
            best_val_metrics = (best_checkpoint.get("metrics") or {}).get("val")
            if isinstance(best_val_metrics, dict):
                prio, comp, disp = self._selection_candidate(best_val_metrics, selection_metric)
                if math.isfinite(comp):
                    best_key = (prio, comp)
                    best_display = disp
            if best_display is None and selection_metric == "f1_macro" and "best_f1" in checkpoint and checkpoint["best_f1"] is not None:
                val = float(checkpoint["best_f1"])
                best_key = (1, val)
                best_display = val

        if self.rank == 0:
            self.logger.info(
                "Resumed downstream from %s at epoch %d metric=%s best_score=%s",
                path,
                next_epoch + 1,
                selection_metric,
                best_display,
            )
        return next_epoch, best_key, best_display

    def _load_best_for_test(self) -> None:
        best_path = os.path.join(self.cfg.logging.checkpoint_dir, "downstream_best.pt")
        if not os.path.exists(best_path):
            return
        checkpoint = torch.load(best_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model"])

    def _save_test_metrics(self, metrics: Dict[str, Optional[float]]) -> None:
        if self.rank != 0:
            return
        os.makedirs(self.cfg.logging.checkpoint_dir, exist_ok=True)
        path = os.path.join(self.cfg.logging.checkpoint_dir, "test_metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def fit(self):
        selection_metric = self._validate_selection_metric()
        self.build()
        if self.rank == 0:
            self.logger.info("Starting downstream on %s (selection_metric=%s)", self.device, selection_metric)
            self.logger.info("Train size=%d Val size=%d", len(self.train_loader.dataset), len(self.val_loader.dataset))
        start_epoch = 0
        best_key = (-1, float("-inf"))
        best_display: Optional[float] = None
        warned_auroc_fallback = False
        if self.cfg.logging.resume:
            start_epoch, best_key, best_display = self._resume()
        try:
            for epoch in range(start_epoch, self.cfg.training.epochs):
                train_loss = self.train_one_epoch(epoch)
                val_metrics = self.evaluate(self.val_loader, "Validation")
                metrics = {"val": val_metrics, "train_loss": train_loss}
                prio, comp, disp = self._selection_candidate(val_metrics, selection_metric)
                if selection_metric == "val_auroc" and prio == 0 and not warned_auroc_fallback and self.rank == 0:
                    self.logger.warning(
                        "Validation AUROC unavailable (e.g., missing class in val split); "
                        "temporarily falling back to -val_loss for checkpoint selection."
                    )
                    warned_auroc_fallback = True
                cand_key = (prio, comp)
                is_best = cand_key > best_key
                if is_best:
                    best_key = cand_key
                    best_display = disp
                best_is_fallback = (best_key[0] == 0)
                self.save(
                    epoch,
                    metrics,
                    is_best=is_best,
                    best_score=best_display,
                    selection_metric=selection_metric,
                    best_is_fallback=best_is_fallback,
                )
                if self.rank == 0:
                    self.logger.info(
                        "Epoch %d done train=%.6f val_loss=%.6f val_auroc=%s %s best=%s",
                        epoch + 1,
                        train_loss,
                        float(val_metrics.get("loss", float("nan"))),
                        val_metrics.get("auroc"),
                        selection_metric,
                        best_display,
                    )
            if self.test_loader is not None:
                self._wait_for_save()
                self._load_best_for_test()
                test_metrics = self.evaluate(self.test_loader, "Test")
                self._save_test_metrics(test_metrics)
        finally:
            self._wait_for_save()
