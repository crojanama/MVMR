"""Training loop utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.checkpoint import save_checkpoint


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float


class Trainer:
    """End-to-end trainer with validation tracking + best-checkpoint saving."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
        output_dir: str,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        writer: Optional[SummaryWriter] = None,
        logger: Optional[logging.Logger] = None,
        log_interval_batches: int = 0,
        log_every_n_batches: Optional[int] = None,
        log_every_n_steps: Optional[int] = None,
        use_amp: bool = False,
        channels_last: bool = False,
        compile_model: bool = False,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.writer = writer
        self.logger = logger
        if log_every_n_steps is not None:
            log_interval_batches = log_every_n_steps
        elif log_every_n_batches is not None:
            log_interval_batches = log_every_n_batches
        self.log_interval_batches = max(0, log_interval_batches)

        # Mixed precision / memory format are only meaningful on CUDA.
        self.use_amp = bool(use_amp) and self.device.type == "cuda"
        self.channels_last = bool(channels_last) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        # Keep a handle to the eager module so checkpoints keep their original
        # state_dict keys even when the forward module is wrapped by torch.compile
        # (which would otherwise prefix keys with ``_orig_mod.``).
        self._checkpoint_model = self.model
        if compile_model:
            # torch.compile traces the model on first use; gains vary by GPU arch.
            self.model = torch.compile(self.model)

        self.best_val_accuracy = 0.0
        self.best_checkpoint_path = self.output_dir / "best_model.pt"

    def _run_epoch(self, dataloader: DataLoader, train_mode: bool, epoch: int) -> EpochMetrics:
        if train_mode:
            self.model.train()
        else:
            self.model.eval()

        # Accumulate on-device to avoid a per-batch GPU->CPU sync (.item()), which
        # serializes the pipeline and leaves the GPU idle waiting for the host.
        running_loss = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device)
        total = 0

        phase = "train" if train_mode else "val"
        if self.logger is not None:
            self.logger.info("Epoch %d %s: starting (%d batches)", epoch, phase, len(dataloader))

        for batch_idx, (images, labels) in enumerate(dataloader, start=1):
            images = images.to(self.device, non_blocking=True)
            if self.channels_last:
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(self.device, non_blocking=True)

            with torch.set_grad_enabled(train_mode):
                with torch.autocast(device_type="cuda", enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                if train_mode:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

            batch_size = labels.size(0)
            running_loss += loss.detach() * batch_size
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum()
            total += batch_size

            if (
                self.logger is not None
                and self.log_interval_batches > 0
                and batch_idx % self.log_interval_batches == 0
            ):
                # .item() here forces a sync, but only once per log interval.
                self.logger.info(
                    "Epoch %d %s: batch %d/%d (loss=%.4f, acc=%.4f)",
                    epoch,
                    phase,
                    batch_idx,
                    len(dataloader),
                    running_loss.item() / total if total > 0 else 0.0,
                    correct.item() / total if total > 0 else 0.0,
                )

        if total == 0:
            return EpochMetrics(loss=0.0, accuracy=0.0)

        avg_loss = running_loss.item() / total
        accuracy = correct.item() / total
        return EpochMetrics(loss=avg_loss, accuracy=accuracy)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        class_names: List[str],
        model_name: str,
        early_stopping_patience: int = 0,
    ) -> Dict[str, float]:
        """Run full train/validation loop and save best checkpoint.

        If ``early_stopping_patience > 0``, training stops once validation accuracy
        has not improved for that many consecutive epochs.
        """

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            train_metrics = self._run_epoch(train_loader, train_mode=True, epoch=epoch)
            val_metrics = self._run_epoch(val_loader, train_mode=False, epoch=epoch)

            if self.scheduler is not None:
                self.scheduler.step()

            history["train_loss"].append(train_metrics.loss)
            history["train_accuracy"].append(train_metrics.accuracy)
            history["val_loss"].append(val_metrics.loss)
            history["val_accuracy"].append(val_metrics.accuracy)

            if self.writer is not None:
                self.writer.add_scalar("train/loss", train_metrics.loss, epoch)
                self.writer.add_scalar("train/accuracy", train_metrics.accuracy, epoch)
                self.writer.add_scalar("val/loss", val_metrics.loss, epoch)
                self.writer.add_scalar("val/accuracy", val_metrics.accuracy, epoch)

            if self.logger is not None:
                self.logger.info(
                    "Epoch %d summary: train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                    epoch,
                    train_metrics.loss,
                    train_metrics.accuracy,
                    val_metrics.loss,
                    val_metrics.accuracy,
                )

            improved = val_metrics.accuracy > self.best_val_accuracy
            if val_metrics.accuracy >= self.best_val_accuracy:
                self.best_val_accuracy = val_metrics.accuracy
                checkpoint_payload = {
                    "model_state_dict": self._checkpoint_model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                    "class_names": class_names,
                    "model_name": model_name,
                    "best_val_accuracy": self.best_val_accuracy,
                    "epoch": epoch,
                    "history": history,
                }
                save_checkpoint(checkpoint_payload, str(self.best_checkpoint_path))

            epochs_without_improvement = 0 if improved else epochs_without_improvement + 1
            if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
                if self.logger is not None:
                    self.logger.info(
                        "Early stopping at epoch %d: val accuracy has not improved for %d epochs "
                        "(best=%.4f).",
                        epoch,
                        epochs_without_improvement,
                        self.best_val_accuracy,
                    )
                break

        final_metrics = {
            "best_val_accuracy": self.best_val_accuracy,
            "final_train_accuracy": history["train_accuracy"][-1] if history["train_accuracy"] else 0.0,
            "final_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else 0.0,
        }

        if self.writer is not None:
            self.writer.flush()

        return final_metrics
