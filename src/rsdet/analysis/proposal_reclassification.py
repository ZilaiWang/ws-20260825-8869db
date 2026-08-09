"""N2-2：对象学生重分类与融合。

背景
----
N2-1 训练出对象学生（ConvNeXt-T，25 细类 + background 哨兵）。N2-2 用学生
对 M1 的候选 crop 重分类，产出与 M1 原始预测对齐的**重分类预测**，供
cross-fit 阈值评估消融链：

- ``reclassify``（25 类重分类）：每个候选的学生细类预测替换 M1 类别；
- ``reclassify_bg``（背景拒识）：学生判为 background 的候选直接丢弃；
- ``joint``（联合）：重分类 + 背景拒识同时生效。

融合保持 M1 的框与 score，只替换类别与过滤状态——这样消融差异干净归因于
对象学生，符合总纲 N2-2 固定顺序。
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from rsdet.analysis.proposal_crops import BACKGROUND_CLASS_ID
from rsdet.data.crop_classification import render_crop
from rsdet.models.crop_classifier import build_convnext_tiny_classifier

# 融合模式。
MODE_RECLASSIFY = "reclassify"          # 25 类重分类（学生类别替换 M1 类别）
MODE_BACKGROUND = "background_reject"    # 只拒背景（学生判 background 则丢弃）
MODE_JOINT = "joint"                    # 重分类 + 背景拒识


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path) -> None:
    """加载对象学生 checkpoint。"""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return None


def _parse_xyxy(value: str) -> tuple[float, float, float, float]:
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError(f"crop_xyxy 必须 4 个数: {value!r}")
    return (parts[0], parts[1], parts[2], parts[3])


@torch.no_grad()
def reclassify_proposals(
    *,
    manifest_rows: Sequence[Mapping[str, str]],
    data_root: str | Path,
    checkpoint_path: str | Path,
    weights_path: str | Path,
    resolution: int,
    batch_size: int,
    device: torch.device,
    mode: str,
) -> list[dict[str, Any]]:
    """用对象学生对候选 crop 重分类，返回重分类预测记录。

    Returns:
        每项 ``{proposal_uid, image_id, category_id, score, dropped}``。
        ``dropped=True`` 表示被背景拒识丢弃（仅 MODE_BACKGROUND / MODE_JOINT）。
    """
    if mode not in {MODE_RECLASSIFY, MODE_BACKGROUND, MODE_JOINT}:
        raise ValueError(f"未知模式 {mode!r}")
    if not manifest_rows:
        raise ValueError("manifest_rows 不能为空")

    model = build_convnext_tiny_classifier(
        num_classes=26,
        weight_path=weights_path,
        regime="fine_tune",
    )
    load_checkpoint(model, checkpoint_path)
    model = model.to(device)
    model.eval()

    data_root = Path(data_root).expanduser().resolve()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    results: list[dict[str, Any]] = []
    image_cache: dict[Path, Image.Image] = {}

    for start in range(0, len(manifest_rows), batch_size):
        batch_rows = manifest_rows[start : start + batch_size]
        batch_tensors: list[torch.Tensor] = []
        for row in batch_rows:
            image_path = (data_root / row["source_relative_path"]).resolve()
            if image_path not in image_cache:
                image_cache[image_path] = Image.open(image_path)
            crop = render_crop(
                image_cache[image_path],
                _parse_xyxy(row["crop_xyxy"]),
                resolution,
            )
            tensor = torch.from_numpy(
                np.asarray(crop, dtype=np.float32).transpose(2, 0, 1)
            )
            tensor = (tensor / 255.0 - mean) / std
            batch_tensors.append(tensor)
        batch = torch.stack(batch_tensors).to(device)
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        scores = probs.max(dim=1).values

        for row, pred, prob_score in zip(batch_rows, preds.tolist(), scores.tolist()):
            dropped = False
            category_id = int(pred)
            if pred == BACKGROUND_CLASS_ID:
                if mode in {MODE_BACKGROUND, MODE_JOINT}:
                    dropped = True
                else:
                    # reclassify 模式下背景哨兵退化为保留原类别（保守）。
                    category_id = int(row["class_id"])
            elif mode == MODE_RECLASSIFY or mode == MODE_JOINT:
                category_id = int(pred)
            results.append(
                {
                    "proposal_uid": str(row["proposal_uid"]),
                    "image_id": int(row["image_id"]),
                    "fold": int(row["fold"]),
                    "original_category_id": int(row["class_id"]),
                    "category_id": category_id,
                    "score": float(row["score"]),
                    "student_score": float(prob_score),
                    "dropped": dropped,
                }
            )

    return results


def build_reclassified_predictions(
    *,
    oof_predictions: Mapping[int, list[dict[str, Any]]],
    reclassified: Sequence[Mapping[str, Any]],
    mode: str,
) -> dict[int, list[dict[str, Any]]]:
    """把重分类结果融合回 M1 原始预测，输出 COCO 格式预测。

    匹配键：``(image_id, score)``。重分类结果只覆盖 manifest 中的候选
    （score >= manifest 阈值），OOF 聚合里更低的候选没有重分类记录，
    保持原样（它们低于工作点，不影响 cross-fit 的评估窗口）。

    - MODE_RECLASSIFY / MODE_JOINT：替换类别；
    - MODE_BACKGROUND / MODE_JOINT：dropped 的候选被剔除。
    """
    by_image_score: dict[tuple[int, float], Mapping[str, Any]] = {}
    for record in reclassified:
        key = (int(record["image_id"]), float(record["score"]))
        by_image_score[key] = record

    result: dict[int, list[dict[str, Any]]] = {}
    for image_id, records in oof_predictions.items():
        kept: list[dict[str, Any]] = []
        for record in records:
            key = (int(record["image_id"]), float(record["score"]))
            rc = by_image_score.get(key)
            if rc is not None:
                # manifest 中的背景训练样本（原类别 25）不属于检测候选，
                # 一律剔除；学生判背景的按模式处理。
                if int(rc["original_category_id"]) == BACKGROUND_CLASS_ID:
                    continue
                if rc.get("dropped"):
                    continue
                new_record = dict(record)
                new_record["category_id"] = int(rc["category_id"])
                new_record["student_score"] = float(rc["student_score"])
                kept.append(new_record)
            else:
                kept.append(dict(record))
        if kept:
            result[image_id] = kept
    return result
