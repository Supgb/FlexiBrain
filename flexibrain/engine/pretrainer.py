from __future__ import annotations

import math
import os
import shutil
import threading
from typing import Optional

import torch
import torch.optim as optim

from flexibrain.config import RunConfig
from flexibrain.data import build_pretrain_dataloaders
from flexibrain.data.collate import prepare_batch_data
from flexibrain.distributed import cleanup_distributed, setup_distributed
from flexibrain.models import build_pretrain_model
from flexibrain.utils.logging import setup_logger
from flexibrain.utils.seed import set_seed
from flexibrain.utils.training import get_dynamic_momentum, update_ema


def _to_cpu_snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone() if value.device.type == "cpu" else value.detach().cpu()
    if isinstance(value, dict):
        return {k: _to_cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_snapshot(v) for v in value)
    return value


def _write_checkpoint(payload: dict, latest_path: str, best_path: Optional[str]) -> None:
    latest_tmp = f"{latest_path}.{os.getpid()}.tmp"
    torch.save(payload, latest_tmp)
    os.replace(latest_tmp, latest_path)
    if best_path is not None:
        best_tmp = f"{best_path}.{os.getpid()}.tmp"
        shutil.copyfile(latest_path, best_tmp)
        os.replace(best_tmp, best_path)


class Pretrainer:
    def __init__(self, cfg: RunConfig):
        self.cfg = cfg
        self.rank = cfg.training.local_rank
        self.world_size = cfg.training.world_size
        if self.world_size > 1:
            setup_distributed(self.rank, self.world_size)
        self.device = torch.device(f"cuda:{self.rank}" if torch.cuda.is_available() else "cpu")
        self.logger = setup_logger("pretrain", cfg.logging.log_dir, rank=self.rank)
        self._save_thread: Optional[threading.Thread] = None

    def _wait_for_save(self) -> None:
        thread = getattr(self, "_save_thread", None)
        if thread is not None:
            thread.join()
            self._save_thread = None

    def build(self):
        set_seed(self.cfg.training.seed)
        self.model = build_pretrain_model(self.cfg.model, self.device)
        self.train_loader, self.val_loader = build_pretrain_dataloaders(self.cfg.data, self.cfg.training, rank=self.rank, world_size=self.world_size)
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
            progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        return self

    def _optimizer_step(self, momentum: float) -> None:
        if self.cfg.training.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.training.grad_clip)
        scale_before = self.scaler.get_scale() if self.use_amp else None
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        stepped = (not self.use_amp) or (self.scaler.get_scale() >= scale_before)
        if stepped:
            update_ema(self.model, momentum)
            self.scheduler.step()

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        num_batches = 0
        accumulation_steps = max(1, int(self.cfg.training.grad_accumulation_steps))
        momentum = get_dynamic_momentum(
            epoch,
            self.cfg.training.epochs,
            self.cfg.model.momentum,
            self.cfg.model.final_momentum,
            schedule=self.cfg.model.ema_schedule,
        )
        self.optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(self.train_loader):
            x, meta, orig_Ts, affines = prepare_batch_data(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=getattr(self, "amp_dtype", torch.float16), enabled=self.use_amp):
                loss, _, _, _ = self.model(x, mask_ratio=self.cfg.training.mask_ratio, meta=meta, orig_Ts=orig_Ts, affines=affines)
            self.scaler.scale(loss / accumulation_steps).backward()
            total_loss += loss.detach().to(torch.float64)
            num_batches += 1
            if (batch_idx + 1) % accumulation_steps == 0:
                self._optimizer_step(momentum)
            if self.rank == 0 and (batch_idx + 1) % self.cfg.logging.log_interval == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                loss_val = float(loss.detach().item())
                avg_val = float(total_loss.item()) / num_batches
                self.logger.info(
                    "Epoch %d [%d/%d] loss=%.6f avg=%.6f lr=%.3e momentum=%.6f",
                    epoch + 1,
                    batch_idx + 1,
                    len(self.train_loader),
                    loss_val,
                    avg_val,
                    current_lr,
                    momentum,
                )
        if num_batches % accumulation_steps != 0:
            self._optimizer_step(momentum)
        return float(total_loss.item()) / max(1, num_batches)

    @torch.no_grad()
    def validate(self, epoch: int) -> float:
        self.model.eval()
        total_loss = torch.zeros((), device=self.device, dtype=torch.float64)
        num_batches = 0
        for batch in self.val_loader:
            x, meta, orig_Ts, affines = prepare_batch_data(batch, self.device)
            with torch.autocast(device_type=self.device.type, dtype=getattr(self, "amp_dtype", torch.float16), enabled=self.use_amp):
                loss, _, _, _ = self.model(x, mask_ratio=self.cfg.training.mask_ratio, meta=meta, orig_Ts=orig_Ts, affines=affines)
            total_loss += loss.detach().to(torch.float64)
            num_batches += 1
        avg = float(total_loss.item()) / max(1, num_batches)
        if self.rank == 0:
            self.logger.info("Epoch %d validation loss=%.6f", epoch + 1, avg)
        return avg

    def save(self, epoch: int, val_loss: float, best_loss: float, is_best: bool):
        if self.rank != 0:
            return
        os.makedirs(self.cfg.logging.checkpoint_dir, exist_ok=True)
        payload = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
            "val_loss": val_loss,
            "best_loss": best_loss,
            "config": vars(self.cfg.model),
        }
        latest_path = os.path.join(self.cfg.logging.checkpoint_dir, "checkpoint_latest.pt")
        best_path = os.path.join(self.cfg.logging.checkpoint_dir, "checkpoint_best.pt") if is_best else None
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

    def _resume(self) -> tuple[int, float]:
        path = self.cfg.logging.resume
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        next_epoch = int(checkpoint["epoch"]) + 1
        best_loss = float(checkpoint["best_loss"])
        if self.rank == 0:
            self.logger.info("Resumed pretrain from %s at epoch %d best_loss=%.6f", path, next_epoch + 1, best_loss)
        return next_epoch, best_loss

    def fit(self):
        self.build()
        if self.rank == 0:
            self.logger.info("Starting pretrain on %s", self.device)
            self.logger.info("Train size=%d Val size=%d", len(self.train_loader.dataset), len(self.val_loader.dataset))
        start_epoch = 0
        best_loss = float("inf")
        if self.cfg.logging.resume:
            start_epoch, best_loss = self._resume()
        try:
            for epoch in range(start_epoch, self.cfg.training.epochs):
                if hasattr(self.train_loader.sampler, "set_epoch"):
                    self.train_loader.sampler.set_epoch(epoch)
                train_loss = self.train_one_epoch(epoch)
                val_loss = self.validate(epoch)
                is_best = val_loss < best_loss
                if is_best:
                    best_loss = val_loss
                self.save(epoch, val_loss, best_loss, is_best=is_best)
                if self.rank == 0:
                    self.logger.info("Epoch %d done train=%.6f val=%.6f best=%.6f", epoch + 1, train_loss, val_loss, best_loss)
        finally:
            self._wait_for_save()
        if self.world_size > 1:
            cleanup_distributed()
