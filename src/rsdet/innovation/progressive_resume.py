"""Audited, importable trainer for a single-GPU to DDP hardware migration.

Training math comes from the installed DetectionTrainer. This subclass only
asserts the frozen recipe and records restoration evidence on every rank.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import RANK


def assert_restored(actual, expected, path="optimizer") -> None:
    """Compare restored state, allowing the optimizer's device/dtype conversion."""
    if isinstance(expected, torch.Tensor):
        if not isinstance(actual, torch.Tensor):
            raise RuntimeError(f"{path}: missing restored tensor")
        converted = expected.detach().to(device=actual.device, dtype=actual.dtype)
        if not torch.equal(actual, converted):
            raise RuntimeError(f"{path}: restored tensor mismatch")
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise RuntimeError(f"{path}: restored keys mismatch")
        for key in expected:
            assert_restored(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected):
            raise RuntimeError(f"{path}: restored sequence mismatch")
        for index, (got, want) in enumerate(zip(actual, expected, strict=True)):
            assert_restored(got, want, f"{path}.{index}")
    elif actual != expected:
        raise RuntimeError(f"{path}: restored value mismatch")


class AuditedProgressiveResumeTrainer(DetectionTrainer):
    def _load_checkpoint_state(self, ckpt):
        super()._load_checkpoint_state(ckpt)
        if not ckpt.get("optimizer") or not ckpt.get("ema"):
            raise RuntimeError("resume requires optimizer and EMA state")
        assert_restored(self.optimizer.state_dict(), ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            assert_restored(self.scaler.state_dict(), ckpt["scaler"], "scaler")
        assert_restored(self.ema.ema.state_dict(), ckpt["ema"].state_dict(), "ema")
        if self.ema.updates != ckpt["updates"]:
            raise RuntimeError("EMA update counter was not restored")
        self._restoration_verified = True

    def _setup_train(self):
        super()._setup_train()
        expected = {
            "epochs": 40, "imgsz": 1280, "batch": 24, "seed": 42,
            "optimizer": "AdamW", "lr0": 0.0002, "lrf": 0.1,
            "mosaic": 0.0, "close_mosaic": 40, "cos_lr": True,
            "nbs": 64, "val": False, "deterministic": True,
        }
        for key, value in expected.items():
            if getattr(self.args, key) != value:
                raise RuntimeError(f"frozen recipe mismatch: {key}")
        if self.world_size != 3 or self.start_epoch != 2:
            raise RuntimeError("expected three ranks resuming after epoch 2")
        if not getattr(self, "_restoration_verified", False):
            raise RuntimeError("optimizer/EMA restoration not verified")
        transforms = [repr(item) for item in self.args.augmentations]
        if transforms != ["RandomRotate90(p=1.0)"]:
            raise RuntimeError(f"rotation augmentation was not restored: {transforms}")
        audit = {
            "status": "pass", "rank": RANK, "world_size": self.world_size,
            "start_epoch_zero_based": self.start_epoch,
            "global_batch": self.batch_size,
            "per_rank_batch": self.train_loader.batch_size,
            "gradient_accumulation": self.accumulate,
            "nominal_effective_batch": self.batch_size * self.accumulate,
            "training_image_count": len(self.train_loader.dataset),
            "optimizer_state_restored_exact_after_dtype_conversion": True,
            "scaler_state_restored": True, "ema_state_restored": True,
            "ema_updates": self.ema.updates,
            "optimizer_lr": [group["lr"] for group in self.optimizer.param_groups],
            "optimizer_weight_decay": [group["weight_decay"] for group in self.optimizer.param_groups],
            "augmentations": transforms,
            "scheduler_last_epoch": self.scheduler.last_epoch,
        }
        if audit["training_image_count"] != 4481:
            raise RuntimeError("expected 4481 official training images")
        output = Path(os.environ["SCALEROUTE_RESUME_AUDIT_DIR"])
        output.mkdir(parents=True, exist_ok=True)
        (output / f"rank_{RANK}.json").write_text(
            json.dumps(audit, indent=2) + "\n", encoding="utf-8"
        )
