#!/usr/bin/env python3
"""A 主线 1：车辆 near-miss 特征响应审计（GPU）。

对 80 个无候选车辆 GT 所在图，用对应 fold 的 M1 checkpoint（YOLO26s-1024）
做 hooked 推理：
- 提取 P2/P3/P4（stride 4/8/16）backbone/neck 特征图；
- 在 GT 中心位置采样特征响应（L2 范数、通道激活率）；
- 对比组：251 个匹配车辆、71 个低分被滤车辆。

用法（GPU）：
    PYTHONPATH=src python scripts/n1_b_feature_response_audit.py \
        --checkpoint /workspace/M1/fold_0/runs/foundation/weights/best.pt \
        --formal-manifest /workspace/N1A/formal_crop_manifest.csv \
        --aggregate /workspace/M1/M1-CV3-OOF-aggregate \
        --data-root /workspace/data \
        --fold 0 \
        --output /workspace/N1A/feature_response_fold0.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from rsdet.analysis.crossfit_thresholds import load_cv3_aggregate
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n1b_feature_audit")

VEHICLE_CLASS = 24
WORK_THRESHOLD = 0.051
IOU_THRESHOLD = 0.35

# YOLO26s 的 detect 层位置（第 22 层是 head，前层是 neck 特征）。
# 通过 stride 区分 P3/P4/P5；P2 通过额外的高分辨率分支获得。
FEATURE_STRIDES = {4: "P2", 8: "P3", 16: "P4", 32: "P5"}


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="车辆 near-miss 特征响应审计")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    return parser.parse_args(argv)


def _load_gt_vehicles(manifest: Path, fold: int) -> dict[int, list[dict]]:
    """加载指定 fold 的车辆 GT。"""
    vehicles: dict[int, list[dict]] = {}
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["crop_policy"] != "tight" or int(row["class_id"]) != VEHICLE_CLASS:
                continue
            if int(row["fold"]) != fold:
                continue
            image_id = int(row["formal_image_id"])
            vehicles.setdefault(image_id, []).append(
                {
                    "uid": row["annotation_uid"],
                    "bbox": [
                        float(row["gt_x0"]),
                        float(row["gt_y0"]),
                        float(row["gt_x1"]),
                        float(row["gt_y1"]),
                    ],
                }
            )
    return vehicles


def _classify_vehicle(gt_bbox, cands):
    """对车辆 GT 分类：matched / low_score / no_candidate。"""
    best_iou = 0.0
    best_score = 0.0
    for c in cands:
        if int(c["category_id"]) != VEHICLE_CLASS:
            continue
        i = _iou_xyxy(gt_bbox, [float(v) for v in c["bbox_xyxy"]])
        if i > best_iou:
            best_iou = i
        if i >= IOU_THRESHOLD:
            best_score = max(best_score, float(c["score"]))
    if best_iou >= IOU_THRESHOLD and best_score >= WORK_THRESHOLD:
        return "matched"
    if best_iou >= IOU_THRESHOLD:
        return "low_score"
    return "no_candidate"


@torch.no_grad()
def _sample_features(model, image: Image.Image, bbox: list[float], imgsz: int):
    """推理单图，在 GT 中心采样各尺度特征响应。"""
    model.eval()
    orig_w, orig_h = image.size
    # letterbox 预处理（与 YOLO train 一致）。
    ratio = min(imgsz / orig_w, imgsz / orig_h)
    new_w, new_h = int(orig_w * ratio), int(orig_h * ratio)
    img = image.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), (114, 114, 114))
    canvas.paste(img, (0, 0))
    tensor = (
        torch.from_numpy(np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1))
        / 255.0
    )
    tensor = tensor.unsqueeze(0).to(next(model.parameters()).device)

    # 目标中心在原图 → letterbox 后坐标。
    cx = (bbox[0] + bbox[2]) / 2 * ratio
    cy = (bbox[1] + bbox[3]) / 2 * ratio

    features: dict[str, float] = {}
    activations: dict[str, float] = {}
    # 注册 hook：收集 Detect 前各尺度特征。
    hooks = []
    storage = {}
    candidates: dict[str, torch.nn.Module] = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            feat = out[0] if isinstance(out, tuple) else out
            storage[name] = feat.detach()

        return hook_fn

    model_nn = model.model if hasattr(model, "model") else model
    # Detect 层（最后一层）的输入来源 = P3/P4/P5 三路径。
    detect_idx = len(model_nn) - 1
    detect = model_nn[detect_idx]
    # stride 由 Detect 层提供：strides = [8, 16, 32]。
    strides = [int(s) for s in detect.stride.tolist()]
    source_layers = [int(f) for f in detect.f]
    stride_to_name = {4: "P2", 8: "P3", 16: "P4", 32: "P5"}
    for src_idx, stride in zip(source_layers, strides):
        name = stride_to_name.get(stride, f"P{stride}x")
        candidates[name] = model_nn[src_idx]
        hooks.append(
            model_nn[src_idx].register_forward_hook(make_hook(name))
        )
    # 同时 hook 前两个 C2f（浅层特征，若 stride 不同会重复捕获）。
    # 该模型 stride 只到 8，无 P2；浅层 C2f 层 6/8 是 backbone 中段。
    for extra_idx in (6, 8):
        extra_name = f"backbone{extra_idx}"
        candidates[extra_name] = model_nn[extra_idx]
        hooks.append(
            model_nn[extra_idx].register_forward_hook(make_hook(extra_name))
        )

    try:
        model(tensor)
    finally:
        for hook in hooks:
            hook.remove()

    # 计算每个候选特征在目标中心邻域的响应。
    for name, feat in storage.items():
        f = feat[0]  # C x H x W
        h, w = f.shape[1], f.shape[2]
        fx = int(cx / imgsz * w)
        fy = int(cy / imgsz * h)
        fx = min(max(fx, 0), w - 1)
        fy = min(max(fy, 0), h - 1)
        # 3x3 邻域 L2。
        patch = f[:, max(0, fy - 1) : fy + 2, max(0, fx - 1) : fx + 2]
        l2 = float(patch.norm().item())
        channel_norm = float(f[:, fy, fx].norm().item())
        global_norm = float(f.norm().item())
        features[name] = {
            "l2_3x3": round(l2, 4),
            "center_l2": round(channel_norm, 4),
            "global_l2": round(global_norm, 4),
            "rel_center": round(channel_norm / (global_norm + 1e-6), 4),
        }
    return features


def main(argv: list[str] | None = None) -> int:
    """运行特征响应审计。"""
    args = parse_args(argv)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        from ultralytics import YOLO

        model = YOLO(str(args.checkpoint)).model.to(device)

        _, predictions, image_folds = load_cv3_aggregate(
            args.aggregate, candidate_floor=0.001
        )
        vehicles = _load_gt_vehicles(args.formal_manifest, args.fold)
        data_root = Path(args.data_root).expanduser().resolve()

        # 图片路径映射（从 aggregate 的 oof_images.csv 或按 image_id）。
        image_paths = {}
        images_csv = Path(args.aggregate) / "oof_images.csv"
        with images_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                image_paths[int(row["image_id"])] = row["relative_path"]

        results = []
        for image_id, gts in vehicles.items():
            cands = predictions.get(image_id, [])
            rel_path = image_paths.get(image_id)
            if rel_path is None:
                logger.warning("image %s 缺路径，跳过", image_id)
                continue
            image_path = (data_root / rel_path).resolve()
            if not image_path.is_file():
                logger.warning("图像缺失: %s", image_path)
                continue
            image = Image.open(image_path).convert("RGB")
            for gt in gts:
                bucket = _classify_vehicle(gt["bbox"], cands)
                feats = _sample_features(model, image, gt["bbox"], args.imgsz)
                results.append(
                    {
                        "image_id": image_id,
                        "uid": gt["uid"],
                        "bucket": bucket,
                        "bbox": [round(v, 2) for v in gt["bbox"]],
                        "features": feats,
                    }
                )
            logger.info("image %s: %d 车辆完成", image_id, len(gts))

        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("=== 完成 ===")
        logger.info("图像数: %d | 车辆数: %d", len(results), len(results))
        from collections import Counter

        logger.info("分桶: %s", dict(Counter(r["bucket"] for r in results)))
        logger.info("已保存: %s", output)
    except Exception as error:
        logger.exception("特征审计失败: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
