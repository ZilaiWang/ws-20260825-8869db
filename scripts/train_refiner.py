#!/usr/bin/env python3
"""训练项目自研 HPR 少样本细粒度原型分支。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

from rsdet.data.object_crops import ObjectCropDataset
from rsdet.models.prototype_refiner import (
    HierarchicalPrototypeLoss,
    HierarchicalPrototypeRefiner,
)
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging
from rsdet.utils.seed import set_seed

logger = setup_logging(name="train_refiner")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 HPR 少样本细粒度原型分支")
    parser.add_argument("--config", type=Path, required=True, help="HPR 训练配置")
    parser.add_argument("--device", type=str, default=None, help="覆盖训练设备")
    parser.add_argument(
        "--resume", type=Path, default=None, help="从 HPR checkpoint 恢复"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="检查数据和配置，并在单个 train/val batch 上做 GPU 冒烟测试",
    )
    return parser.parse_args(argv)


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} 必须是映射")
    return dict(value)


def _run_epoch(
    model,
    loader,
    criterion,
    *,
    device,
    optimizer=None,
    scaler=None,
    amp: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    class_correct = np.zeros(25, dtype=np.int64)
    class_total = np.zeros(25, dtype=np.int64)
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with (
            torch.set_grad_enabled(training),
            torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=amp,
            ),
        ):
            outputs = model(images)
            losses = criterion(outputs, labels)
        if training:
            scaler.scale(losses["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_prototypes(outputs["embeddings"], labels)

        predictions = model.fused_logits(outputs).argmax(dim=1)
        batch_size = len(labels)
        total_loss += float(losses["loss"].detach()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_samples += batch_size
        labels_cpu = labels.detach().cpu().numpy()
        predictions_cpu = predictions.detach().cpu().numpy()
        class_total += np.bincount(labels_cpu, minlength=25)
        class_correct += np.bincount(
            labels_cpu[labels_cpu == predictions_cpu], minlength=25
        )

    present = class_total > 0
    per_class_recall = np.divide(
        class_correct,
        class_total,
        out=np.zeros_like(class_correct, dtype=np.float64),
        where=present,
    )
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "macro_recall": (
            float(per_class_recall[present].mean()) if np.any(present) else 0.0
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        logger.error("配置文件不存在: %s", args.config)
        return 1
    try:
        config = load_config(args.config)
        seed = int(config.get("seed", 42))
        set_seed(seed)
        data_config = _mapping(config.get("data", {}), "data")
        model_config = _mapping(config.get("model", {}), "model")
        train_config = _mapping(config.get("train", {}), "train")
        loss_config = _mapping(config.get("loss", {}), "loss")
        output_dir = Path(str(config.get("output_dir", "outputs/hpr"))).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device(args.device or str(train_config.get("device", "cuda:0")))
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("配置要求 CUDA，但当前 PyTorch 环境检测不到可用 GPU")

        input_size = int(data_config.get("input_size", 128))
        context_ratio = float(data_config.get("context_ratio", 0.15))
        train_dataset = ObjectCropDataset(
            data_config["root"],
            data_config["manifest"],
            "train",
            output_size=input_size,
            context_ratio=context_ratio,
            box_jitter=float(data_config.get("box_jitter", 0.08)),
            augment=True,
        )
        val_dataset = ObjectCropDataset(
            data_config["root"],
            data_config["manifest"],
            "val",
            output_size=input_size,
            context_ratio=context_ratio,
            box_jitter=0.0,
            augment=False,
        )
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            train_dataset.sampling_weights(
                power=float(train_config.get("sampler_power", 0.5))
            ),
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        loader_args = {
            "batch_size": int(train_config.get("batch_size", 128)),
            "num_workers": int(train_config.get("num_workers", 4)),
            "pin_memory": device.type == "cuda",
        }
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            drop_last=True,
            **loader_args,
        )
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_args)

        model = HierarchicalPrototypeRefiner(**model_config).to(device)
        criterion = HierarchicalPrototypeLoss(
            train_dataset.class_counts.tolist(), **loss_config
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_config.get("lr", 0.001)),
            weight_decay=float(train_config.get("weight_decay", 0.0005)),
        )
        epochs = int(train_config.get("epochs", 60))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(epochs, 1),
            eta_min=float(train_config.get("min_lr", 1e-5)),
        )
        amp = bool(train_config.get("amp", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        start_epoch = 0
        best_macro_recall = -1.0
        if args.resume is not None:
            checkpoint = torch.load(
                args.resume, map_location=device, weights_only=False
            )
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_macro_recall = float(checkpoint.get("best_macro_recall", -1.0))

        if args.dry_run:
            train_image_count = len(
                {record.image_id for record in train_dataset.records}
            )
            val_image_count = len({record.image_id for record in val_dataset.records})
            train_metrics = _run_epoch(
                model,
                [next(iter(train_loader))],
                criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp=amp,
            )
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model,
                    [next(iter(val_loader))],
                    criterion,
                    device=device,
                    amp=amp,
                )
            summary = {
                "dry_run": True,
                "device": str(device),
                "cuda_name": (
                    torch.cuda.get_device_name(device) if device.type == "cuda" else None
                ),
                "train_images": train_image_count,
                "val_images": val_image_count,
                "train_objects": len(train_dataset),
                "val_objects": len(val_dataset),
                "parameters": sum(parameter.numel() for parameter in model.parameters()),
                "train_batch": train_metrics,
                "val_batch": val_metrics,
            }
            (output_dir / "dry_run_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (output_dir / "resolved_config.yaml").write_text(
                yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            logger.info(
                "HPR dry-run 完成: %d train / %d val 目标，device=%s",
                len(train_dataset),
                len(val_dataset),
                device,
            )
            return 0

        history: list[dict[str, Any]] = []
        for epoch in range(start_epoch, epochs):
            train_metrics = _run_epoch(
                model,
                train_loader,
                criterion,
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp=amp,
            )
            with torch.no_grad():
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    criterion,
                    device=device,
                    amp=amp,
                )
            scheduler.step()
            record = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(record)
            logger.info(
                "epoch %d/%d train_loss=%.4f val_acc=%.4f val_macro=%.4f",
                epoch + 1,
                epochs,
                train_metrics["loss"],
                val_metrics["accuracy"],
                val_metrics["macro_recall"],
            )
            state = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_macro_recall": max(
                    best_macro_recall, val_metrics["macro_recall"]
                ),
                "model_config": model_config,
                "input_size": input_size,
                "context_ratio": context_ratio,
                "class_counts": train_dataset.class_counts.tolist(),
                "config": config,
                "metrics": record,
            }
            torch.save(state, output_dir / "last.pt")
            if val_metrics["macro_recall"] > best_macro_recall:
                best_macro_recall = val_metrics["macro_recall"]
                torch.save(state, output_dir / "best.pt")
            (output_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        (output_dir / "resolved_config.yaml").write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
    except (
        ImportError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as error:
        logger.error("HPR 训练失败: %s", error)
        return 1
    logger.info(
        "HPR 训练完成，最佳 val macro recall=%.4f: %s", best_macro_recall, output_dir
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
