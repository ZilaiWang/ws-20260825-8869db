#!/usr/bin/env python3
"""P04 缓存特征的统一线性探针。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.evaluation.classification import evaluate_classification
from rsdet.features.p04_probe import (
    deterministic_view_index,
    fit_pca_train_only,
    l2_normalize,
    load_probe_data,
    training_class_counts,
    transform_pca,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="P04 cached-feature linear probe")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--feature-name", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fold", type=int, required=True, choices=(0, 1, 2))
    parser.add_argument("--policy", default="tight")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    # 保持 P03 tight-224 linear probe 的优化步数和学习率日程；缓存特征
    # 不能成为偷偷增大 batch 的理由，否则 TASK-01 不再是等价门禁。
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--minimum-epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-epochs", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--normalization", choices=("none", "l2"), default="l2")
    parser.add_argument("--pca-dim", type=int)
    parser.add_argument(
        "--head-init",
        default="p04_default",
        choices=("p04_default", "p03_convnext_compat"),
    )
    parser.add_argument(
        "--allow-cache-subset",
        action="store_true",
        help="只用于 P04-3 calibration 诊断，禁止形成 formal 结论",
    )
    return parser.parse_args(argv)


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _evaluate(
    torch: Any,
    model: Any,
    x: np.ndarray,
    y: np.ndarray,
    device: Any,
    *,
    amp_enabled: bool,
) -> tuple:
    model.eval()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=amp_enabled
    ):
        logits = model(torch.from_numpy(x).to(device)).float().cpu().numpy()
    metrics, confusion = evaluate_classification(y, logits)
    return metrics, confusion, logits


def _write_predictions(
    output: Path,
    indices: np.ndarray,
    data: Any,
    logits: np.ndarray,
) -> None:
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = logits.argmax(axis=1)
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as file:
        fields = [
            "annotation_uid",
            "crop_id",
            "source_image_id",
            "leakage_group_id",
            "true_class_id",
            "true_class_name",
            "pred_class_id",
            "pred_class_name",
            "confidence",
            "correct",
        ]
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for offset, index in enumerate(indices):
            true = int(data.class_ids[index])
            pred = int(predictions[offset])
            writer.writerow(
                {
                    "annotation_uid": data.annotation_uids[index],
                    "crop_id": data.crop_ids[index],
                    "source_image_id": data.source_image_ids[index],
                    "leakage_group_id": data.leakage_group_ids[index],
                    "true_class_id": true,
                    "true_class_name": FINE_NAMES[true],
                    "pred_class_id": pred,
                    "pred_class_name": FINE_NAMES[pred],
                    "confidence": format(float(probabilities[offset, pred]), ".9g"),
                    "correct": true == pred,
                }
            )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.batch_size <= 0 or args.epochs <= 0 or args.minimum_epochs <= 0:
        raise ValueError("batch/epoch 参数必须大于 0")
    if args.minimum_epochs > args.epochs:
        raise ValueError("minimum_epochs 不得大于 epochs")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("feature probe 需要 PyTorch") from error
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("命令要求 CUDA，但 torch.cuda.is_available() 为 False")
    _seed_everything(torch, args.seed)
    output = Path(args.output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空: {output}")
    output.mkdir(parents=True, exist_ok=True)
    data = load_probe_data(
        args.cache_dir,
        feature_name=args.feature_name,
        manifest_path=args.manifest,
        crop_policy=args.policy,
        allow_cache_subset=args.allow_cache_subset,
    )
    train_indices = np.flatnonzero(data.folds != args.fold)
    val_indices = np.flatnonzero(data.folds == args.fold)
    if not len(train_indices) or not len(val_indices):
        raise ValueError("空训练或验证折")
    if set(data.source_image_ids[index] for index in train_indices) & set(
        data.source_image_ids[index] for index in val_indices
    ):
        raise ValueError("训练/验证 source_image_id 泄漏")
    if set(data.leakage_group_ids[index] for index in train_indices) & set(
        data.leakage_group_ids[index] for index in val_indices
    ):
        raise ValueError("训练/验证 leakage_group_id 泄漏")

    features = np.asarray(data.features, dtype=np.float32)
    if args.normalization == "l2":
        features = l2_normalize(features)
    pca: dict[str, np.ndarray] | None = None
    if args.pca_dim:
        pca = fit_pca_train_only(
            features[train_indices], n_components=args.pca_dim, seed=args.seed
        )
        features = l2_normalize(transform_pca(features, pca))
        np.savez(
            output / "pca_train_only.npz",
            components=pca["components"],
            mean=pca["mean"],
            explained_variance_ratio=pca["explained_variance_ratio"],
        )

    val_view = data.view_ids.index("r0")
    val_x = features[val_indices, val_view]
    val_y = data.class_ids[val_indices]
    train_y = data.class_ids[train_indices]
    counts = training_class_counts(train_y)
    if args.head_init == "p03_convnext_compat":
        try:
            from torchvision.models import convnext_tiny
        except ImportError as error:
            raise RuntimeError("P03 head-init 兼容模式需要 torchvision") from error
        # P03 在 seed 后先构建整个 ConvNeXt，然后才新建 768→25 线性层。
        # 这一步只用于 TASK-01 缓存等价门禁，不进入 P04 主比较。
        compatibility_model = convnext_tiny(weights=None)
        del compatibility_model
    model = torch.nn.Linear(features.shape[-1], len(FINE_NAMES)).to(device)
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    steps_per_epoch = math.ceil(len(train_indices) / args.batch_size)
    total_steps = max(1, steps_per_epoch * args.epochs)
    warmup_steps = min(total_steps - 1, int(steps_per_epoch * args.warmup_epochs))

    def lr_factor(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_epoch = -1
    stale = 0
    started = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        chosen_views = np.asarray(
            [
                deterministic_view_index(
                    data.annotation_uids[index], epoch, args.seed, len(data.view_ids)
                )
                for index in train_indices
            ],
            dtype=np.int64,
        )
        epoch_x = features[train_indices, chosen_views]
        generator = torch.Generator().manual_seed(args.seed + epoch)
        order = torch.randperm(len(train_indices), generator=generator).numpy()
        train_loss = 0.0
        train_correct = 0
        for start in range(0, len(order), args.batch_size):
            batch = order[start : start + args.batch_size]
            x = torch.from_numpy(epoch_x[batch]).to(device)
            y = torch.from_numpy(train_y[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=amp_enabled
            ):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss += float(loss.detach()) * len(batch)
            train_correct += int((logits.argmax(dim=1) == y).sum())
        metrics, _, _ = _evaluate(
            torch, model, val_x, val_y, device, amp_enabled=amp_enabled
        )
        score = float(metrics["macro_recall"])
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss / len(order),
                "train_accuracy": train_correct / len(order),
                "val_accuracy": metrics["accuracy"],
                "val_macro_recall": score,
                "val_macro_f1": metrics["macro_f1"],
                "lr": optimizer.param_groups[0]["lr"],
            }
        )
        if score > best_score + args.min_delta:
            best_score = score
            best_epoch = epoch + 1
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_name": args.feature_name,
                    "feature_dimension": features.shape[-1],
                    "fold": args.fold,
                    "seed": args.seed,
                    "normalization": args.normalization,
                    "pca_dim": args.pca_dim,
                    "cache_fingerprint": data.cache_fingerprint,
                    "manifest_sha256": data.manifest_sha256,
                },
                output / "best_checkpoint.pt",
            )
        else:
            stale += 1
        if epoch + 1 >= args.minimum_epochs and stale >= args.patience:
            break

    checkpoint = torch.load(output / "best_checkpoint.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    metrics, confusion, logits = _evaluate(
        torch, model, val_x, val_y, device, amp_enabled=amp_enabled
    )
    metrics["training_class_counts"] = counts
    _json(output / "metrics.json", metrics)
    np.savetxt(output / "confusion_matrix.csv", confusion, delimiter=",", fmt="%d")
    np.savez_compressed(
        output / "validation_logits.npz",
        logits=logits.astype(np.float32),
        labels=val_y.astype(np.int64),
        object_indices=val_indices.astype(np.int64),
    )
    _write_predictions(output, val_indices, data, logits)
    with (output / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(metrics["per_class"][0]))
        writer.writeheader()
        writer.writerows(metrics["per_class"])
    with (output / "history.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "status": "complete",
        "feature_name": args.feature_name,
        "cache_fingerprint": data.cache_fingerprint,
        "manifest_sha256": data.manifest_sha256,
        "ignored_cache_annotations": data.ignored_cache_annotations,
        "fold": args.fold,
        "seed": args.seed,
        "normalization": args.normalization,
        "pca_dim": args.pca_dim,
        "head_init": args.head_init,
        "amp": "fp16" if amp_enabled else "disabled",
        "n_train": len(train_indices),
        "n_val": len(val_indices),
        "n_views": len(data.view_ids),
        "feature_dimension": features.shape[-1],
        "best_epoch": best_epoch,
        "completed_epochs": len(history),
        "early_stopped": len(history) < args.epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "metrics": {
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "aircraft20_macro_recall": metrics["subgroups"]["aircraft20"][
                "macro_recall"
            ],
        },
    }
    _json(output / "run_summary.json", summary)
    _json(output / "resolved_config.json", vars(args))
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
