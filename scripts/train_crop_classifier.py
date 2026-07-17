#!/usr/bin/env python3
"""P0-3 ConvNeXt-Tiny crop 分类训练/评估入口。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from PIL import Image

from rsdet.data.crop_classification import (
    CropClassificationDataset,
    CropRecord,
    class_counts,
    load_crop_records,
    select_deterministic_subset,
    validate_fold_isolation,
)
from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.classification import evaluate_classification
from rsdet.models.crop_classifier import (
    CONVNEXT_TINY_WEIGHT_SHA256,
    build_convnext_tiny_classifier,
    parameter_summary,
    sha256_file,
)
from rsdet.utils.config import load_config


class RandomQuarterTurn:
    """不做任意角插值，只在遥感合理的 90° 方向中均匀取样。"""

    _METHODS = (
        None,
        Image.Transpose.ROTATE_90,
        Image.Transpose.ROTATE_180,
        Image.Transpose.ROTATE_270,
    )

    def __call__(self, image: Image.Image) -> Image.Image:
        method = random.choice(self._METHODS)
        return image if method is None else image.transpose(method)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="P0-3: 在 P0-2 防泄漏 crop manifest 上训练普通分类上限"
    )
    parser.add_argument("--config", default="configs/experiments/p03_convnext_tiny.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument(
        "--policy", required=True, choices=("tight", "context_1p25", "jitter_light")
    )
    parser.add_argument("--resolution", type=int, required=True, choices=(224, 336))
    parser.add_argument("--regime", required=True, choices=("linear_probe", "fine_tune"))
    parser.add_argument("--sampler", default="natural", choices=("natural", "sqrt_inverse"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--checkpoint", help="--eval-only 时必需")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="强制 1 epoch 和小样本，仅验证通路")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _prepare_output(path: str | Path, overwrite: bool) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not overwrite:
        raise FileExistsError(f"输出目录非空: {output}；如确需覆盖请加 --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    return output


def _require_gpu_stack(device_text: str) -> tuple[Any, Any]:
    try:
        import torch
        import torchvision
    except ImportError as error:
        raise RuntimeError(
            "P0-3 需要 torch==2.5.1, torchvision==0.20.1；请先执行服务器任务单的环境安装步骤"
        ) from error
    if torch.__version__.split("+")[0] != "2.5.1":
        raise RuntimeError(f"本实验锁定 torch==2.5.1，当前为 {torch.__version__}")
    if torchvision.__version__.split("+")[0] != "0.20.1":
        raise RuntimeError(f"本实验锁定 torchvision==0.20.1，当前为 {torchvision.__version__}")
    if device_text.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("命令要求 CUDA，但 torch.cuda.is_available() 为 False")
    return torch, torchvision


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _worker_init(worker_id: int) -> None:
    import torch

    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))


def _build_transforms(torchvision: Any, *, train: bool) -> Any:
    transforms = torchvision.transforms
    operations: list[Any] = []
    if train:
        operations.extend(
            [
                RandomQuarterTurn(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )
    return transforms.Compose(operations)


def _sampler_weights(records: Sequence[CropRecord], exponent: float, cap: float) -> np.ndarray:
    counts = class_counts(records)
    maximum = max(counts.values())
    weights = np.asarray(
        [min((maximum / counts[item.class_id]) ** exponent, cap) for item in records],
        dtype=np.float64,
    )
    return weights


def _sampler_audit(
    records: Sequence[CropRecord], sampler_name: str, sampler_config: Mapping[str, Any]
) -> dict[str, Any]:
    """记录 sampler 的理论类别抽样质量，不将期望值冒充实际 epoch 计数。"""

    counts = class_counts(records)
    class_ids = sorted(counts)
    if sampler_name == "natural":
        class_weights = {class_id: 1.0 for class_id in class_ids}
    elif sampler_name == "sqrt_inverse":
        exponent = float(sampler_config["exponent"])
        cap = float(sampler_config["max_weight_ratio"])
        maximum = max(counts.values())
        class_weights = {
            class_id: min((maximum / counts[class_id]) ** exponent, cap) for class_id in class_ids
        }
    else:
        raise ValueError(f"未知 sampler: {sampler_name}")
    mass = {class_id: counts[class_id] * class_weights[class_id] for class_id in class_ids}
    total_mass = sum(mass.values())
    return {
        "name": sampler_name,
        "replacement": sampler_name == "sqrt_inverse",
        "num_samples_per_epoch": len(records),
        "class_weights": class_weights,
        "expected_class_probability": {
            class_id: mass[class_id] / total_mass for class_id in class_ids
        },
        "note": "expected probabilities; actual sampled counts vary by epoch",
    }


def _make_loader(
    torch: Any,
    dataset: CropClassificationDataset,
    records: Sequence[CropRecord],
    *,
    batch_size: int,
    num_workers: int,
    train: bool,
    sampler_name: str,
    sampler_config: Mapping[str, Any],
    seed: int,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    sampler = None
    shuffle = train
    if train and sampler_name == "sqrt_inverse":
        weights = _sampler_weights(
            records,
            exponent=float(sampler_config["exponent"]),
            cap=float(sampler_config["max_weight_ratio"]),
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(records),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init,
        generator=generator,
        drop_last=False,
    )


def _optimizer_and_scheduler(
    torch: Any,
    model: Any,
    regime: str,
    config: Mapping[str, Any],
    steps_per_epoch: int,
    epochs: int,
) -> tuple[Any, Any]:
    regime_config = config["training"][regime]
    if regime == "linear_probe":
        groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "lr": float(regime_config["lr"]),
            }
        ]
    else:
        head_ids = {id(parameter) for parameter in model.classifier.parameters()}
        groups = [
            {
                "params": [
                    p for p in model.parameters() if p.requires_grad and id(p) not in head_ids
                ],
                "lr": float(regime_config["backbone_lr"]),
            },
            {
                "params": [p for p in model.classifier.parameters() if p.requires_grad],
                "lr": float(regime_config["head_lr"]),
            },
        ]
    optimizer = torch.optim.AdamW(groups, weight_decay=float(regime_config["weight_decay"]))
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = min(
        total_steps - 1, int(steps_per_epoch * float(regime_config["warmup_epochs"]))
    )

    def factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _train_one_epoch(
    torch: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    scheduler: Any,
    device: Any,
    scaler: Any,
    grad_clip_norm: float,
    regime: str,
) -> dict[str, float]:
    if regime == "linear_probe":
        # 冻结骨干也必须冻结其运行行为，避免 stochastic depth 使特征漂移。
        model.eval()
        model.classifier[-1].train()
    else:
        model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    start = time.perf_counter()
    amp_enabled = device.type == "cuda"
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        batch_size = labels.numel()
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        total_samples += batch_size
    elapsed = time.perf_counter() - start
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "samples_per_second": total_samples / elapsed,
    }


def _evaluate(
    torch: Any,
    model: Any,
    loader: Any,
    criterion: Any,
    device: Any,
    training_counts: Mapping[int, int],
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    logits_values: list[np.ndarray] = []
    label_values: list[np.ndarray] = []
    index_values: list[np.ndarray] = []
    total_loss = 0.0
    total_samples = 0
    start = time.perf_counter()
    amp_enabled = device.type == "cuda"
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            batch_size = labels.numel()
            total_loss += float(loss) * batch_size
            total_samples += batch_size
            logits_values.append(logits.float().cpu().numpy())
            label_values.append(labels.cpu().numpy())
            index_values.append(indices.cpu().numpy())
    elapsed = time.perf_counter() - start
    all_logits = np.concatenate(logits_values, axis=0)
    all_labels = np.concatenate(label_values, axis=0)
    all_indices = np.concatenate(index_values, axis=0)
    metrics, confusion = evaluate_classification(
        all_labels, all_logits, training_class_counts=training_counts
    )
    metrics["loss"] = total_loss / total_samples
    metrics["samples_per_second"] = total_samples / elapsed
    return metrics, confusion, all_logits, all_indices


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _write_history(path: Path, history: Sequence[Mapping[str, Any]]) -> None:
    if not history:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


def _write_evaluation_artifacts(
    output: Path,
    records: Sequence[CropRecord],
    indices: np.ndarray,
    logits: np.ndarray,
    metrics: Mapping[str, Any],
    confusion: np.ndarray,
) -> None:
    probabilities = _softmax(logits)
    predictions = logits.argmax(axis=1)
    labels = np.asarray([records[int(index)].class_id for index in indices], dtype=np.int64)
    np.savez_compressed(
        output / "validation_logits.npz",
        logits=logits.astype(np.float32),
        labels=labels,
        record_indices=indices.astype(np.int64),
    )
    np.savetxt(output / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    _json_dump(output / "metrics.json", metrics)
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        rows = list(metrics["per_class"])
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        fields = [
            "crop_id",
            "annotation_uid",
            "source_image_id",
            "true_class_id",
            "true_class_name",
            "pred_class_id",
            "pred_class_name",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for offset, record_index in enumerate(indices):
            record = records[int(record_index)]
            pred = int(predictions[offset])
            writer.writerow(
                {
                    "crop_id": record.crop_id,
                    "annotation_uid": record.annotation_uid,
                    "source_image_id": record.source_image_id,
                    "true_class_id": record.class_id,
                    "true_class_name": record.class_name,
                    "pred_class_id": pred,
                    "pred_class_name": FINE_NAMES[pred],
                    "confidence": format(float(probabilities[offset, pred]), ".9g"),
                    "correct": pred == record.class_id,
                }
            )


def _git_state() -> dict[str, Any]:
    try:
        root = Path(__file__).resolve().parent.parent
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        return {"commit": commit, "dirty": bool(status), "status_line_count": len(status)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None, "status_line_count": None}


def _environment_meta(torch: Any, torchvision: Any, device: Any) -> dict[str, Any]:
    gpu = None
    if device.type == "cuda":
        gpu = {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
        }
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": gpu,
        "git": _git_state(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    manifest_sha = sha256_file(args.manifest)
    if manifest_sha != config["data"]["manifest_sha256"]:
        raise ValueError(f"manifest SHA256 与冻结配置不一致: {manifest_sha}")
    output = _prepare_output(args.output_dir, args.overwrite)
    torch, torchvision = _require_gpu_stack(args.device)
    if args.smoke:
        args.epochs = 1
        args.max_train_samples = args.max_train_samples or 256
        args.max_val_samples = args.max_val_samples or 128
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval-only 必须提供 --checkpoint")
    if args.checkpoint and not args.eval_only:
        raise ValueError("--checkpoint 当前只允许与 --eval-only 一起使用")
    if args.regime == "linear_probe" and args.sampler != "natural":
        raise ValueError("首轮 linear probe 预注册只允许 natural sampler")
    if args.policy == "jitter_light" and not args.eval_only:
        raise ValueError("jitter_light 首轮只作已训练 clean 模型的配对评估")

    device = torch.device(args.device)
    _seed_everything(torch, args.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_records = load_crop_records(
        args.manifest, crop_policy=args.policy, held_out_fold=args.fold, split="train"
    )
    val_records = load_crop_records(
        args.manifest, crop_policy=args.policy, held_out_fold=args.fold, split="val"
    )
    manifest_versions = {item.manifest_version for item in (*train_records, *val_records)}
    if manifest_versions != {config["data"]["manifest_version"]}:
        raise ValueError(f"manifest_version 与冻结配置不一致: {manifest_versions}")
    validate_fold_isolation(train_records, val_records)
    train_records = select_deterministic_subset(train_records, args.max_train_samples)
    val_records = select_deterministic_subset(val_records, args.max_val_samples)
    training_counts = class_counts(train_records)

    regime_config = config["training"][args.regime]
    epochs = int(args.epochs or regime_config["epochs"])
    batch_size = int(args.batch_size or config["runtime"][f"batch_size_{args.resolution}"])
    num_workers = int(
        args.num_workers if args.num_workers is not None else config["runtime"]["num_workers"]
    )
    resolved = {
        "experiment_id": config["experiment"]["id"],
        "model": "convnext_tiny",
        "pretraining": "ImageNet-1K V1",
        "task": "all25",
        "fold": args.fold,
        "policy": args.policy,
        "resolution": args.resolution,
        "regime": args.regime,
        "sampler": args.sampler,
        "seed": args.seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "smoke": args.smoke,
        "eval_only": args.eval_only,
        "manifest": str(Path(args.manifest).expanduser().resolve()),
        "data_root": str(Path(args.data_root).expanduser().resolve()),
        "weights": str(Path(args.weights).expanduser().resolve()),
        "checkpoint": str(Path(args.checkpoint).expanduser().resolve())
        if args.checkpoint
        else None,
        "n_train": len(train_records),
        "n_val": len(val_records),
        "training_class_counts": training_counts,
        "sampler_audit": _sampler_audit(
            train_records, args.sampler, config["sampler"]["sqrt_inverse"]
        ),
    }
    with (output / "resolved_config.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(resolved, file, allow_unicode=True, sort_keys=False)

    model = build_convnext_tiny_classifier(
        len(FINE_NAMES), weight_path=args.weights, regime=args.regime
    ).to(device)
    checkpoint_config: Mapping[str, Any] = {}
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        checkpoint_config = checkpoint.get("resolved_config", {})
        for key in ("fold", "resolution", "regime"):
            if checkpoint_config.get(key) != resolved[key]:
                raise ValueError(
                    f"checkpoint {key}={checkpoint_config.get(key)!r} "
                    f"与评估条件 {resolved[key]!r} 不一致"
                )
        source_policy = checkpoint_config.get("policy")
        if args.policy == "jitter_light":
            if source_policy not in {"tight", "context_1p25"}:
                raise ValueError("jitter 评估必须加载 clean crop checkpoint")
        elif source_policy != args.policy:
            raise ValueError("clean 评估必须使用相同 crop policy checkpoint")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model_parameters = parameter_summary(model)

    train_dataset = CropClassificationDataset(
        train_records,
        args.data_root,
        args.resolution,
        transform=_build_transforms(torchvision, train=True),
        image_cache_size=int(config["runtime"]["image_cache_size_per_worker"]),
    )
    val_dataset = CropClassificationDataset(
        val_records,
        args.data_root,
        args.resolution,
        transform=_build_transforms(torchvision, train=False),
        image_cache_size=int(config["runtime"]["image_cache_size_per_worker"]),
    )
    train_loader = _make_loader(
        torch,
        train_dataset,
        train_records,
        batch_size=batch_size,
        num_workers=num_workers,
        train=True,
        sampler_name=args.sampler,
        sampler_config=config["sampler"]["sqrt_inverse"],
        seed=args.seed,
    )
    val_loader = _make_loader(
        torch,
        val_dataset,
        val_records,
        batch_size=batch_size,
        num_workers=num_workers,
        train=False,
        sampler_name="natural",
        sampler_config=config["sampler"]["sqrt_inverse"],
        seed=args.seed + 1,
    )
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=float(regime_config["label_smoothing"]))
    history: list[dict[str, Any]] = []
    best_epoch = None
    best_macro_recall = -math.inf
    best_checkpoint_path = output / "best_checkpoint.pt"

    if not args.eval_only:
        optimizer, scheduler = _optimizer_and_scheduler(
            torch, model, args.regime, config, len(train_loader), epochs
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        patience = int(regime_config["early_stopping_patience"])
        stale_epochs = 0
        for epoch in range(1, epochs + 1):
            train_metrics = _train_one_epoch(
                torch,
                model,
                train_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                scaler,
                float(regime_config["grad_clip_norm"]),
                args.regime,
            )
            val_metrics, _, _, _ = _evaluate(
                torch, model, val_loader, criterion, device, training_counts
            )
            row = {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_recall": val_metrics["macro_recall"],
                "val_macro_f1": val_metrics["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
                "train_samples_per_second": train_metrics["samples_per_second"],
                "val_samples_per_second": val_metrics["samples_per_second"],
            }
            history.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            score = float(val_metrics["macro_recall"])
            if score > best_macro_recall + float(regime_config["early_stopping_min_delta"]):
                best_macro_recall = score
                best_epoch = epoch
                stale_epochs = 0
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "resolved_config": resolved,
                        "epoch": epoch,
                        "selection_metric": "macro_recall",
                        "selection_value": score,
                    },
                    best_checkpoint_path,
                )
            else:
                stale_epochs += 1
            if epoch >= int(regime_config["minimum_epochs"]) and stale_epochs >= patience:
                break
        _write_history(output / "history.csv", history)
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    else:
        best_epoch = int(checkpoint.get("epoch", 0))
        best_macro_recall = float(checkpoint.get("selection_value", float("nan")))

    final_metrics, confusion, logits, indices = _evaluate(
        torch, model, val_loader, criterion, device, training_counts
    )
    _write_evaluation_artifacts(output, val_records, indices, logits, final_metrics, confusion)
    meta = _environment_meta(torch, torchvision, device)
    meta.update(
        {
            "manifest_sha256": manifest_sha,
            "weight_sha256": sha256_file(args.weights),
            "expected_weight_sha256": CONVNEXT_TINY_WEIGHT_SHA256,
            "loaded_checkpoint_sha256": (sha256_file(args.checkpoint) if args.checkpoint else None),
            "model_parameters": model_parameters,
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            ),
        }
    )
    _json_dump(output / "meta.json", meta)
    summary = {
        "condition": {
            key: resolved[key]
            for key in (
                "fold",
                "policy",
                "resolution",
                "regime",
                "sampler",
                "seed",
                "smoke",
                "eval_only",
            )
        },
        "n_train": len(train_records),
        "n_val": len(val_records),
        "best_epoch": best_epoch,
        "epochs_completed": len(history) if not args.eval_only else None,
        "max_epochs": epochs if not args.eval_only else None,
        "early_stopped": len(history) < epochs if not args.eval_only else None,
        "selection_metric": "macro_recall",
        "best_during_training": best_macro_recall if math.isfinite(best_macro_recall) else None,
        "final_metrics": {
            key: final_metrics[key]
            for key in (
                "accuracy",
                "macro_recall",
                "macro_f1",
                "top5_accuracy",
                "loss",
                "samples_per_second",
            )
        },
        "aircraft20": final_metrics["subgroups"]["aircraft20"],
        "checkpoint_source_condition": (
            {
                key: checkpoint_config.get(key)
                for key in ("fold", "policy", "resolution", "regime", "sampler", "seed")
            }
            if args.checkpoint
            else None
        ),
        "artifact_contract": [
            "resolved_config.yaml",
            "meta.json",
            "metrics.json",
            "per_class_metrics.csv",
            "confusion_matrix.csv",
            "predictions.csv",
            "validation_logits.npz",
            "run_summary.json",
        ],
    }
    if not args.eval_only:
        summary["artifact_contract"].extend(["history.csv", "best_checkpoint.pt"])
        summary["best_checkpoint_sha256"] = sha256_file(best_checkpoint_path)
    _json_dump(output / "run_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
