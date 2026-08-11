#!/usr/bin/env python3
"""V1-FULL-P2 机制诊断：P2 模型的车辆候选覆盖。

这不是官方 Recall/FDR 评估：它按 GT 独立搜索最佳车辆候选，不做
全局分数降序一对一匹配，也不计重复框/FP/FDR。只用于判断 P2 是否产生
原本没有的小车候选。
- 车辆 Recall（工作点 0.051）
- 无候选目标（fold 内车辆 GT 无任何 IoU≥0.35 候选）数量
- 与 M1 基线（feature_response 审计）对比

用法（GPU）：
    PYTHONPATH=src python scripts/n1d_p2_evaluate.py \
        --weights /workspace/results/P2-FOLD0/p2_run/weights/best.pt \
        --cv3-manifest data/splits/cv3_airport_proxy_k60_v2.json \
        --formal-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
        --data-root /workspace/data \
        --fold 0 \
        --output /workspace/results/P2-FOLD0/eval_fold0.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n1d_p2_eval")

VEHICLE_CLASS = 24
WORK_THRESHOLD = 0.051
IOU_THRESHOLD = 0.35


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V1-FULL-P2 车辆 Recall 评估")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--cv3-manifest", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001, help="低阈值保持候选")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """运行 P2 评估。"""
    args = parse_args(argv)
    try:
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        device = "cuda" if torch.cuda.is_available() else "cpu"

        cv3 = json.loads(args.cv3_manifest.read_text(encoding="utf-8"))
        fold_paths = {
            sample["image_id"]: sample["relative_path"]
            for sample in cv3["samples"]
            if sample["fold"] == args.fold
        }

        # 该 fold 的车辆 GT。
        vehicle_gt: list[tuple[int, str, list[float]]] = []
        with args.formal_manifest.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if (
                    row["crop_policy"] != "tight"
                    or int(row["class_id"]) != VEHICLE_CLASS
                    or int(row["fold"]) != args.fold
                ):
                    continue
                vehicle_gt.append(
                    (
                        int(row["formal_image_id"]),
                        row["annotation_uid"],
                        [
                            float(row["gt_x0"]),
                            float(row["gt_y0"]),
                            float(row["gt_x1"]),
                            float(row["gt_y1"]),
                        ],
                    )
                )

        data_root = Path(args.data_root).expanduser().resolve()
        matched = 0
        low_score = 0
        no_candidate = 0
        detail = []
        for image_id, uid, gt_bbox in vehicle_gt:
            rel = fold_paths.get(image_id)
            if rel is None:
                raise ValueError(f"image_id {image_id} 不在 fold {args.fold}")
            image_path = (data_root / rel).resolve()
            if not image_path.is_file():
                raise ValueError(f"图像缺失: {image_path}")

            result = model.predict(
                str(image_path),
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                verbose=False,
            )[0]
            best_iou = 0.0
            best_score = 0.0
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.tolist()
                classes = result.boxes.cls.tolist()
                scores = result.boxes.conf.tolist()
                for box, cls, score in zip(boxes, classes, scores):
                    if int(cls) != VEHICLE_CLASS:
                        continue
                    i = _iou_xyxy(gt_bbox, box)
                    if i > best_iou:
                        best_iou = i
                    if i >= IOU_THRESHOLD:
                        best_score = max(best_score, float(score))

            if best_iou >= IOU_THRESHOLD and best_score >= WORK_THRESHOLD:
                matched += 1
                bucket = "matched"
            elif best_iou >= IOU_THRESHOLD:
                low_score += 1
                bucket = "low_score"
            else:
                no_candidate += 1
                bucket = "no_candidate"
            detail.append(
                {
                    "image_id": image_id,
                    "uid": uid,
                    "bucket": bucket,
                    "best_iou": round(best_iou, 3),
                    "best_score": round(best_score, 4),
                }
            )

        summary = {
            "scientific_scope": "mechanism_diagnostic_only",
            "official_metric": False,
            "formal_admission": False,
            "matching_policy": "per_gt_best_same_vehicle_candidate_not_one_to_one",
            "fold": args.fold,
            "vehicle_gt": len(vehicle_gt),
            "matched": matched,
            "low_score": low_score,
            "no_candidate": no_candidate,
            "vehicle_recall": round(matched / len(vehicle_gt), 4),
            "note": "M1 基线同口径: 车辆 Recall 0.6244 (matched 251/402), fold0 见 near-miss 审计",
        }
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"summary": summary, "detail": detail}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        logger.info("=== V1-FULL-P2 fold %d 评估 ===", args.fold)
        logger.info(
            "车辆 GT: %d | matched %d (Recall %.4f) | 低分 %d | 无候选 %d",
            len(vehicle_gt),
            matched,
            summary["vehicle_recall"],
            low_score,
            no_candidate,
        )
        logger.info("已保存: %s", output)
    except Exception as error:
        logger.exception("P2 评估失败: %s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
