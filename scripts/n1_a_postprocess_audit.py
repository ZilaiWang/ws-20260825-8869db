#!/usr/bin/env python3
"""A 主线 1：车辆候选后处理审计（CPU，无需推理）。

分析 M1 低阈值 OOF 中车辆目标的候选形成情况，区分：
- 有车辆候选但被阈值滤掉（low_score_only）；
- 完全无车辆候选（no_candidate），进一步看同图其他类别候选、
  低分车辆候选（score 分布）、以及同图候选总数（max_det 饱和代理）。

用法：
    PYTHONPATH=src python scripts/n1_a_postprocess_audit.py \
        --aggregate outputs/M1-CV3-OOF-.../M1-CV3-OOF-aggregate \
        --formal-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
        --output outputs/N1A-VEHICLE-AUDIT
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

from rsdet.analysis.crossfit_thresholds import load_cv3_aggregate
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n1a_vehicle_audit")

VEHICLE_CLASS = 24


def _iou_xyxy(a: list[float], b: list[float]) -> float:
    xa, ya = max(a[0], b[0]), max(a[1], b[1])
    xb, yb = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _coco_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A 主线 1 车辆后处理审计")
    parser.add_argument("--aggregate", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--work-threshold", type=float, default=0.051)
    parser.add_argument("--candidate-floor", type=float, default=0.001)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """运行审计。"""
    args = parse_args(argv)
    try:
        metadata, predictions, image_folds = load_cv3_aggregate(
            args.aggregate, candidate_floor=args.candidate_floor
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        logger.error("加载 OOF 失败: %s", error)
        return 1

    # 车辆 GT。
    vehicle_gt: list[tuple[int, str, dict]] = []
    with args.formal_manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["crop_policy"] != "tight" or int(row["class_id"]) != VEHICLE_CLASS:
                continue
            vehicle_gt.append(
                (
                    int(row["formal_image_id"]),
                    row["annotation_uid"],
                    {
                        "bbox": [
                            float(row["gt_x0"]),
                            float(row["gt_y0"]),
                            float(row["gt_x1"]),
                            float(row["gt_y1"]),
                        ],
                        "group": row["group_id"],
                        "fold": row["fold"],
                        "edge_risk": row["source_edge_risk"],
                    },
                )
            )

    # 每图候选。
    pred_by_img: dict[int, list[dict]] = {}
    for record in predictions.values():
        for pred in record:
            pred_by_img.setdefault(int(pred["image_id"]), []).append(pred)

    matched: list[dict] = []
    low_score: list[dict] = []
    no_candidate: list[dict] = []

    for image_id, uid, info in vehicle_gt:
        cands = pred_by_img.get(image_id, [])
        best_iou = 0.0
        best_vehicle_score = 0.0
        vehicle_cands = []
        for c in cands:
            if int(c["category_id"]) != VEHICLE_CLASS:
                continue
            i = _iou_xyxy(info["bbox"], [float(v) for v in c["bbox_xyxy"]])
            if i > best_iou:
                best_iou = i
            if i >= 0.35:
                best_vehicle_score = max(best_vehicle_score, float(c["score"]))
                vehicle_cands.append((i, float(c["score"])))

        all_scores = sorted((float(c["score"]) for c in cands), reverse=True)
        total_cands = len(cands)
        record = {
            "image_id": image_id,
            "uid": uid,
            "group": info["group"],
            "fold": int(info["fold"]),
            "edge_risk": info["edge_risk"],
            "best_iou": best_iou,
            "best_vehicle_score": best_vehicle_score,
            "total_candidates_in_image": total_cands,
            "top_score_in_image": all_scores[0] if all_scores else 0.0,
        }
        if best_iou >= 0.35 and best_vehicle_score >= args.work_threshold:
            record["bucket"] = "matched"
            matched.append(record)
        elif best_iou >= 0.35:
            record["bucket"] = "low_score_only"
            record["best_vehicle_iou"] = max(i for i, _ in vehicle_cands)
            low_score.append(record)
        else:
            record["bucket"] = "no_candidate"
            no_candidate.append(record)

    # 统计。
    summary = {
        "vehicle_gt_total": len(vehicle_gt),
        "matched": len(matched),
        "low_score_only": len(low_score),
        "no_candidate": len(no_candidate),
        "work_recall": round(len(matched) / len(vehicle_gt), 4),
        "no_candidate_groups": dict(Counter(r["group"] for r in no_candidate).most_common()),
        "no_candidate_folds": dict(Counter(r["fold"] for r in no_candidate)),
        "no_candidate_edge_risk": dict(Counter(r["edge_risk"] for r in no_candidate)),
        "no_candidate_total_cands_dist": dict(
            Counter(r["total_candidates_in_image"] for r in no_candidate)
        ),
        "no_candidate_maxdet_saturation_proxy": sum(
            1 for r in no_candidate if r["total_candidates_in_image"] >= 500
        ),
        "no_candidate_top_score_lt_005": sum(
            1 for r in no_candidate if r["top_score_in_image"] < 0.05
        ),
        "low_score_groups": dict(Counter(r["group"] for r in low_score).most_common()),
    }

    output_dir = Path(args.output_dir).expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        logger.error("输出目录非空，禁止覆盖: %s", output_dir)
        return 1
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "analysis": "A_mainline1_vehicle_postprocess_audit",
        "work_threshold": args.work_threshold,
        "candidate_floor": args.candidate_floor,
        "summary": summary,
        "records": matched + low_score + no_candidate,
    }
    (output_dir / "vehicle_audit_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    logger.info("=== A 主线 1 车辆后处理审计 ===")
    logger.info("车辆 GT: %d | 匹配 %d (Recall %.4f) | 低分被滤 %d | 无候选 %d",
                len(vehicle_gt), len(matched), summary["work_recall"],
                len(low_score), len(no_candidate))
    logger.info("无候选来源组数: %d", len(summary["no_candidate_groups"]))
    logger.info("无候选 fold: %s", summary["no_candidate_folds"])
    logger.info("无候选 edge_risk: %s", summary["no_candidate_edge_risk"])
    logger.info("无候选图候选总数分布: %s", summary["no_candidate_total_cands_dist"])
    logger.info("无候选图中候选≥500(max_det饱和代理): %d", summary["no_candidate_maxdet_saturation_proxy"])
    logger.info("无候选图 top-score<0.05: %d", summary["no_candidate_top_score_lt_005"])
    logger.info("已保存: %s", output_dir / "vehicle_audit_result.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
