#!/usr/bin/env python3
"""N2-1：ConvNeXt 对象学生训练（Pred-OOF proposal crop）。

用法（GPU）：
    PYTHONPATH=src python scripts/train_object_student.py \
        --manifest outputs/N2-PROPO-CROP/proposal_crop_manifest.csv \
        --data-root /workspace/data \
        --weights /workspace/pretrained/convnext_tiny-983f1562.pth \
        --output-dir /workspace/results/N2-OBJECT-STUDENT \
        --resolution 224 --epochs 15 --seed 42

设计（对齐总纲 N2-1）：
- 训练集 = 非 held-out fold 的候选（按 leakage_group_id 划分，防泄漏）；
- 验证集 = held-out fold 候选（与训练同源候选、同 fold 划分）；
- 从原图实时渲染 crop（tight policy、pad、direct square resize）；
- ConvNeXt-T ImageNet 初始化，25 类 + 背景（26 类）输出；
- 输出：best checkpoint + 验证集重分类预测 + 逐视图指标。

说明：N2 是探索性实验（非 formal），不做 P03 那样的冻结 CLI 校验。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from rsdet.data.crop_classification import FINE_NAMES, render_crop
from rsdet.models.crop_classifier import build_convnext_tiny_classifier
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n2_object_student")

N_FINE = len(FINE_NAMES)  # 25
N_CLASSES = N_FINE + 1  # 25 细类 + background=25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="N2-1 对象学生训练")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=15,
        help="最少训练 epoch（对象学生探索默认跑满）",
    )
    parser.add_argument(
        "--train-views",
        nargs="+",
        default=["deployable_positive", "oracle_positive", "hard_negative"],
        help="训练使用的视图",
    )
    return parser.parse_args(argv)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _rows_by_fold(rows: Sequence[Mapping[str, str]]) -> dict[int, list[dict[str, str]]]:
    folded: defaultdict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        folded[int(row["fold"])].append(dict(row))
    return dict(folded)


def _parse_xyxy(value: str) -> tuple[float, float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"crop_xyxy 必须 4 个数: {value!r}")
    return (parts[0], parts[1], parts[2], parts[3])


class ProposalCropDataset(torch.utils.data.Dataset):
    """从 proposal crop manifest 实时渲染 crop。"""

    def __init__(
        self,
        rows: Sequence[Mapping[str, str]],
        data_root: str | Path,
        resolution: int,
        *,
        image_cache_size: int = 8,
    ) -> None:
        if not rows:
            raise ValueError("rows 不能为空")
        self.rows = list(rows)
        self.data_root = Path(data_root).expanduser().resolve()
        self.resolution = resolution
        self._cache: dict[Path, Image.Image] = {}
        self._cache_keys: list[Path] = []
        self._cache_size = image_cache_size

    def __len__(self) -> int:
        return len(self.rows)

    def _image(self, relative_path: str) -> Image.Image:
        path = (self.data_root / relative_path).resolve()
        if path not in self._cache:
            self._cache[path] = Image.open(path)
            self._cache_keys.append(path)
            if len(self._cache_keys) > self._cache_size:
                old = self._cache_keys.pop(0)
                self._cache.pop(old, None)
        return self._cache[path]

    def __getitem__(self, index: int) -> tuple[Any, int]:
        row = self.rows[index]
        image = self._image(row["source_relative_path"])
        crop = render_crop(image, _parse_xyxy(row["crop_xyxy"]), self.resolution)
        tensor = torch.from_numpy(np.asarray(crop, dtype=np.float32).transpose(2, 0, 1))
        tensor = tensor / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor, int(row["class_id"])


def _standard_transform(rows: Sequence[Mapping[str, str]], *, seed: int) -> None:
    """打乱行（确定性种子）。"""
    rng = random.Random(seed)
    rng.shuffle(rows)


def _train_val_split(
    rows: Sequence[Mapping[str, str]],
    *,
    held_out_fold: int,
    train_views: Sequence[str],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """按 fold 拆分：train=非 held-out，val=held-out。"""
    train: list[dict[str, str]] = []
    val: list[dict[str, str]] = []
    for row in rows:
        if str(row["view"]) not in train_views:
            continue
        if int(row["fold"]) == held_out_fold:
            val.append(dict(row))
        else:
            train.append(dict(row))
    return train, val


def _train_one_fold(
    train_rows: Sequence[Mapping[str, str]],
    val_rows: Sequence[Mapping[str, str]],
    *,
    data_root: Path,
    resolution: int,
    weights_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
    output_dir: Path,
    fold: int,
) -> dict[str, Any]:
    """训练单个 held-out fold 的对象学生，返回验证指标。"""
    train_rows = list(train_rows)
    _standard_transform(train_rows, seed=seed)
    train_dataset = ProposalCropDataset(
        train_rows, data_root, resolution, image_cache_size=16
    )
    val_dataset = ProposalCropDataset(val_rows, data_root, resolution, image_cache_size=8)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = build_convnext_tiny_classifier(
        num_classes=N_CLASSES,
        weight_path=weights_path,
        regime="fine_tune",
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None
    metrics = {
        "fold": fold,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": 0.0,
        "val_macro_recall": 0.0,
        "per_class_recall": {},
    }

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        avg_train_loss = total_loss / max(total_batches, 1)
        metrics["train_loss"].append(avg_train_loss)

        val_loss, acc, macro_recall, per_class = _evaluate(
            model, val_loader, criterion, device
        )
        metrics["val_loss"].append(val_loss)
        metrics["val_accuracy"] = acc
        metrics["val_macro_recall"] = macro_recall
        metrics["per_class_recall"] = per_class
        logger.info(
            "fold %d epoch %d/%d train_loss %.4f val_loss %.4f acc %.4f macro %.4f",
            fold, epoch, epochs, avg_train_loss, val_loss, acc, macro_recall,
        )
        if acc > best_acc and epoch >= 1:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    fold_dir = output_dir / f"fold{fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if best_state is not None:
        torch.save(best_state, fold_dir / "best_checkpoint.pt")
    summary_path = fold_dir / "run_summary.json"
    summary = {
        **metrics,
        "resolution": resolution,
        "epochs": epochs,
        "best_val_accuracy": best_acc,
        "checkpoint": "best_checkpoint.pt",
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float, dict[str, float]]:
    """评估：返回 (val_loss, accuracy, macro_recall, per_class_recall)。"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    conf_matrix: dict[int, Counter[int]] = defaultdict(Counter)
    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += images.size(0)
        for pred, target in zip(preds.tolist(), targets.tolist()):
            conf_matrix[target][pred] += 1

    val_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    per_class: dict[str, float] = {}
    macro_total = 0.0
    for target, pred_counts in conf_matrix.items():
        class_total = sum(pred_counts.values())
        class_correct = pred_counts.get(target, 0)
        recall = class_correct / class_total if class_total else 0.0
        per_class[str(target)] = recall
        macro_total += recall
    macro_recall = macro_total / max(len(conf_matrix), 1)
    return val_loss, acc, macro_recall, per_class


def main(argv: list[str] | None = None) -> int:
    """运行 N2-1 对象学生三折训练。"""
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("设备: %s", device)

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", output_dir)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        rows = _load_rows(args.manifest)
        folds = sorted({int(row["fold"]) for row in rows})
        if folds != [0, 1, 2]:
            raise ValueError(f"manifest fold 必须是 0/1/2，实际 {folds}")
        summaries: list[dict[str, Any]] = []
        for fold in folds:
            train_rows, val_rows = _train_val_split(
                rows, held_out_fold=fold, train_views=args.train_views
            )
            if not train_rows or not val_rows:
                raise ValueError(f"fold {fold} 无训练或验证样本")
            logger.info(
                "fold %d: train %d / val %d",
                fold, len(train_rows), len(val_rows),
            )
            summary = _train_one_fold(
                train_rows,
                val_rows,
                data_root=args.data_root,
                resolution=args.resolution,
                weights_path=args.weights,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                seed=args.seed + fold,
                device=device,
                output_dir=output_dir,
                fold=fold,
            )
            summaries.append(summary)
    except (OSError, KeyError, TypeError, ValueError) as error:
        logger.error("N2-1 训练失败: %s", error)
        return 1

    final_summary = {
        "analysis": "N2-1_object_student",
        "manifest": str(args.manifest),
        "resolution": args.resolution,
        "epochs": args.epochs,
        "seed": args.seed,
        "folds": summaries,
        "mean_val_macro_recall": sum(s["val_macro_recall"] for s in summaries) / len(summaries),
        "mean_val_accuracy": sum(s["val_accuracy"] for s in summaries) / len(summaries),
    }
    (output_dir / "final_summary.json").write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("=== N2-1 完成 ===")
    logger.info("三折平均 macro_recall %.4f / accuracy %.4f",
                final_summary["mean_val_macro_recall"],
                final_summary["mean_val_accuracy"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
