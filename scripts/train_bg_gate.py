#!/usr/bin/env python3
"""N2-CFG 训练：粗类条件式前景门控（二分类 BCE）。

训练合同（《改进方案 1》第 2.4/2.5/5.2 节，全部冻结）：

- 输入：``context_1.25`` 224x224 单视图 crop（manifest 已含 context 坐标）；
- 模型：ConvNeXt-T shared trunk + shared head + 3 粗类 residual head；
- 损失：粗类宏均衡 BCE；
- 采样：每 batch 50% foreground / 50% audited clear background；
  正样本粗类等概率、粗类内细类近似等概率；
- 冻结策略：快筛 ``freeze_backbone``，正式 ``freeze_first_three_stages``；
- 固定 5 epoch，final-epoch checkpoint，不 early stopping，单一 seed；
- held-out fold 不选 checkpoint、不调阈值（由 evaluate 阶段在
  inner_score_calib 上独立完成）。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from rsdet.analysis.background_gate import COARSE_CLASSES, INPUT_RESOLUTION

_SCORE_BIN_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0 + 1e-9)


def load_manifest_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _score_bin(score: float) -> str:
    for left, right in zip(_SCORE_BIN_EDGES[:-1], _SCORE_BIN_EDGES[1:]):
        if left <= score < right:
            return f"{left:.1f}_{right:.1f}"
    return f"{_SCORE_BIN_EDGES[-2]:.1f}_1.0"


def build_strata(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[int]]:
    """把 manifest 行索引按采样层分组。

    - ``fg/{coarse}/{class_id}``：正样本，粗类内细类分组（细类均衡用）；
    - ``bg/{coarse}/{fold}/{score_bin}``：负样本，粗类 x fold x score bin 分层。
    """
    strata: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        is_fg = str(row["is_foreground"]).strip() in {"1", "true", "True"}
        coarse = str(row["coarse"]).strip()
        if is_fg:
            strata[f"fg/{coarse}/{int(row['class_id'])}"].append(index)
        else:
            strata[f"bg/{coarse}/{int(row['fold'])}/{_score_bin(float(row['score']))}"].append(index)
    return dict(strata)


def build_balanced_batch_indices(
    rows: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    seed: int,
    batches_per_epoch: int,
    active_coarse: Sequence[str] = COARSE_CLASSES,
) -> list[list[int]]:
    """生成一个 epoch 的均衡 batch 索引列表。

    每个 batch：前一半从正样本（粗类等概率、粗类内细类等概率）采样，
    后一半从负样本（粗类等概率、层内均匀）采样。只覆盖 ``active_coarse``
    粗类（首轮 ship/vehicle/aircraft 三粗类都训练，aircraft 评估时 shadow）。
    """
    strata = build_strata(rows)
    rng = random.Random(seed)
    active = [str(value) for value in active_coarse]
    fg_fine: dict[str, list[str]] = defaultdict(list)
    bg_strata: dict[str, list[str]] = defaultdict(list)
    for key in strata:
        parts = key.split("/")
        if parts[0] == "fg":
            if parts[1] in active:
                fg_fine[parts[1]].append(key)
        else:
            if parts[1] in active:
                bg_strata[parts[1]].append(key)

    for coarse in active:
        if not fg_fine.get(coarse) or not bg_strata.get(coarse):
            raise ValueError(f"粗类 {coarse} 缺少正样本或负样本，无法均衡采样")

    half = batch_size // 2
    batches: list[list[int]] = []
    for _ in range(batches_per_epoch):
        batch: list[int] = []
        # 正样本：粗类等概率，粗类内细类等概率，层内均匀。
        for _ in range(half):
            coarse = rng.choice(active)
            fine_key = rng.choice(fg_fine[coarse])
            batch.append(rng.choice(strata[fine_key]))
        # 负样本：粗类等概率，层内均匀。
        for _ in range(batch_size - half):
            coarse = rng.choice(active)
            bg_key = rng.choice(bg_strata[coarse])
            batch.append(rng.choice(strata[bg_key]))
        batches.append(batch)
    return batches


def _validate_manifest(rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("manifest 为空")
    required = {
        "proposal_uid",
        "fold",
        "coarse",
        "class_id",
        "is_foreground",
        "score",
        "context_x0",
        "context_y0",
        "context_x1",
        "context_y1",
        "source_relative_path",
    }
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"manifest 行缺少字段: {sorted(missing)}")
        if str(row["coarse"]).strip() not in COARSE_CLASSES:
            raise ValueError(f"非法 coarse={row['coarse']!r}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--data-root", type=Path, required=True)
    value.add_argument("--convnext-weights", type=Path, required=True)
    value.add_argument("--output-dir", type=Path, required=True)
    value.add_argument("--held-out-fold", type=int, required=True, choices=(0, 1, 2))
    value.add_argument("--freeze", required=True)
    value.add_argument("--epochs", type=int, default=5)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--batches-per-epoch", type=int, default=200)
    value.add_argument("--learning-rate", type=float, default=1e-3)
    value.add_argument("--seed", type=int, default=202625)
    value.add_argument("--resolution", type=int, default=INPUT_RESOLUTION)
    value.add_argument("--device", default="cuda")
    value.add_argument("--verify-weight-sha256", action="store_true")
    return value


def _render_batch(
    batch_rows: Sequence[Mapping[str, Any]],
    data_root: Path,
    resolution: int,
) -> tuple[Any, list[int], list[str]]:
    """渲染一批 context crop，返回 (tensor, foreground_labels, coarses)。"""
    import numpy as np
    import torch
    from PIL import Image

    from rsdet.data.crop_classification import render_crop

    image_cache: dict[Path, Image.Image] = {}
    tensors: list[Any] = []
    labels: list[int] = []
    coarses: list[str] = []
    for row in batch_rows:
        path = (data_root / row["source_relative_path"]).resolve()
        if path not in image_cache:
            image_cache[path] = Image.open(path)
        crop = render_crop(
            image_cache[path],
            (
                float(row["context_x0"]),
                float(row["context_y0"]),
                float(row["context_x1"]),
                float(row["context_y1"]),
            ),
            resolution,
        )
        array = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)
        tensor = torch.from_numpy(array)
        tensors.append(tensor)
        labels.append(1 if str(row["is_foreground"]).strip() in {"1", "true", "True"} else 0)
        coarses.append(str(row["coarse"]).strip())
    return torch.stack(tensors), labels, coarses


def train_one_fold(
    *,
    manifest: str | Path,
    data_root: str | Path,
    convnext_weights: str | Path,
    output_dir: str | Path,
    held_out_fold: int,
    freeze: str,
    epochs: int,
    batch_size: int,
    batches_per_epoch: int,
    learning_rate: float,
    seed: int,
    resolution: int,
    device: str,
    verify_weight_sha256: bool,
) -> dict[str, Any]:
    import torch
    from torch import nn

    from rsdet.models.background_gate_classifier import (
        build_coarse_foreground_gate,
        parameter_summary,
    )

    rows = load_manifest_rows(manifest)
    _validate_manifest(rows)
    train_rows = [row for row in rows if int(row["fold"]) != held_out_fold]
    if not train_rows:
        raise ValueError(f"held_out_fold={held_out_fold} 后无训练样本")

    torch.manual_seed(seed)
    device_obj = torch.device(device)
    model = build_coarse_foreground_gate(
        weight_path=str(convnext_weights),
        freeze=freeze,
        verify_weight_sha256=verify_weight_sha256,
        device=device_obj,
    )
    summary = parameter_summary(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device_obj)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device_obj)

    model.train()
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        indices = build_balanced_batch_indices(
            train_rows,
            batch_size=batch_size,
            seed=seed + epoch,
            batches_per_epoch=batches_per_epoch,
        )
        epoch_loss = 0.0
        for batch_indices in indices:
            batch_rows = [train_rows[index] for index in batch_indices]
            x, labels, coarses = _render_batch(batch_rows, Path(data_root), resolution)
            x = x.to(device_obj)
            x = (x / 255.0 - mean) / std
            labels_t = torch.tensor(labels, dtype=torch.float32, device=device_obj)
            out = model(x)
            # 粗类宏均衡 BCE：每个样本用其粗类条件 logit 计算损失。
            logits = torch.stack(
                [out.fg_logit(coarse).squeeze(1) for coarse in coarses], dim=0
            )
            loss = loss_fn(logits, labels_t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * len(batch_indices) / batches_per_epoch
        history.append({"epoch": epoch, "loss": epoch_loss / max(len(indices), 1)})

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / f"bg_gate_fold{held_out_fold}_final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    result = {
        "held_out_fold": held_out_fold,
        "freeze": freeze,
        "epochs": epochs,
        "train_sample_count": len(train_rows),
        "parameter_summary": summary,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
    (destination / f"train_result_fold{held_out_fold}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = train_one_fold(
        manifest=args.manifest,
        data_root=args.data_root,
        convnext_weights=args.convnext_weights,
        output_dir=args.output_dir,
        held_out_fold=args.held_out_fold,
        freeze=args.freeze,
        epochs=args.epochs,
        batch_size=args.batch_size,
        batches_per_epoch=args.batches_per_epoch,
        learning_rate=args.learning_rate,
        seed=args.seed,
        resolution=args.resolution,
        device=args.device,
        verify_weight_sha256=args.verify_weight_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
