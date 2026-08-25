#!/usr/bin/env python3
"""E 主线 3：全局对象聚合的可复现评估脚本（WP3 real eval）。

补材料 37 指出的"评估脚本缺失"缺口：从 OOF proposals 出发，经过
``global_aggregation.aggregate`` 聚合后，用官方 evaluator 计算 pooled 与
V1.6 macro 指标，并记录全部输入 SHA，保证结果可复现。

输入：
- ``--proposals``：OOF 候选 CSV（xywh，55,548 条，``oof_proposals.csv``）；
- ``--formal-crop-manifest``：冻结 formal crop manifest（GT，SHA a3bed44f）。

输出 ``wp3_real_eval.json``，字段与 E 原始存档对齐（``agg_s`` 口径待 E 成员
确认，本脚本不复现该字段）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from rsdet.analysis.oof_detection import decompose_official_errors, load_formal_ground_truth
from rsdet.evaluation.official_metric import (
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.global_aggregation import aggregate
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_proposals(path: Path) -> tuple[int, dict[int, list[dict[str, Any]]]]:
    """读 OOF 候选，按 image_id 分组为 aggregate 需要的 xywh proposal 列表。"""
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    count = 0
    with path.expanduser().resolve().open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[int(row["image_id"])].append(
                {
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                    "width": float(row["width"]),
                    "height": float(row["height"]),
                    "category_id": int(row["category_id"]),
                    "score": float(row["score"]),
                }
            )
            count += 1
    return count, dict(grouped)


def _xywh_to_xyxy(box: dict[str, Any]) -> list[float]:
    x0 = float(box["x"])
    y0 = float(box["y"])
    return [x0, y0, x0 + float(box["width"]), y0 + float(box["height"])]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--proposals", type=Path, required=True)
    value.add_argument("--formal-crop-manifest", type=Path, required=True)
    value.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--cluster-eps", type=float, default=50.0)
    value.add_argument("--merge-iou", type=float, default=0.3)
    value.add_argument("--nms-iou", type=float, default=0.5)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = parse_evaluation_protocol(load_config(args.project_config))

    n_proposals, grouped = _load_proposals(args.proposals)
    formal = load_formal_ground_truth(
        args.formal_crop_manifest,
        expected_images=4481,
        expected_annotations=20933,
    )

    pred_boxes: dict[int, list[dict[str, Any]]] = {}
    n_objects = 0
    conflict_objects = 0
    for image_id, proposals in grouped.items():
        objects = aggregate(
            proposals,
            cluster_eps=args.cluster_eps,
            merge_iou=args.merge_iou,
            nms_iou=args.nms_iou,
        )
        n_objects += len(objects)
        records: list[dict[str, Any]] = []
        for obj in objects:
            votes = obj.get("category_votes")
            if votes and sum(1 for v in votes.values() if v > 0) > 1:
                conflict_objects += 1
            records.append(
                {
                    "bbox_xyxy": _xywh_to_xyxy(obj),
                    "score": float(obj["score"]),
                    "category_id": int(obj["category_id"]),
                }
            )
        pred_boxes[image_id] = records

    overall, trace = evaluate_predictions_with_trace(
        formal.boxes,
        pred_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        formal.boxes,
        pred_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )

    n_tp = sum(1 for _ in trace.matches)
    n_fp = sum(1 for _ in trace.unmatched_predictions)

    # 聚合前基线：直接用原始 OOF proposals 评估（不经过 aggregate），
    # 补材料 37 缺陷 2 的"聚合前 Recall"缺失。
    pre_pred_boxes: dict[int, list[dict[str, Any]]] = {}
    for image_id, proposals in grouped.items():
        pre_pred_boxes[image_id] = [
            {
                "bbox_xyxy": _xywh_to_xyxy(p),
                "score": float(p["score"]),
                "category_id": int(p["category_id"]),
            }
            for p in proposals
        ]
    pre_overall, _ = evaluate_predictions_with_trace(
        formal.boxes,
        pre_pred_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )

    # FDR@score>=0.25：补材料 37 缺陷 2 的"FDR(score≥0.25)=3.83%"缺失。
    filtered_boxes: dict[int, list[dict[str, Any]]] = {
        image_id: [r for r in records if r["score"] >= 0.25]
        for image_id, records in pred_boxes.items()
    }
    fdr_overall, _ = evaluate_predictions_with_trace(
        formal.boxes,
        filtered_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )

    # class-agnostic 定位召回对照：复现 E 原始 wp3 存档的 recall（~0.9694）。
    # 官方口径要求「框匹配且细类正确」；class-agnostic 只要求「框覆盖目标」，
    # 两者之差即 FP_CLS（框覆盖目标但细类错误）。
    _, cases, _ = decompose_official_errors(
        formal,
        pred_boxes,
        threshold=0.0,
        protocol=protocol,
        model_key="WP3",
        include_cases=True,
    )
    fp_cls = sum(
        1 for case in cases if case["case_side"] == "prediction" and case["reason"] == "FP_CLS"
    )
    class_agnostic_tp = n_tp + fp_cls
    class_agnostic_recall = class_agnostic_tp / formal.annotation_count

    payload = {
        "n_images": len(grouped),
        "n_proposals": n_proposals,
        "n_objects": n_objects,
        "n_gt": formal.annotation_count,
        "n_tp": n_tp,
        "n_fp": n_fp,
        "recall": overall.recall,
        "precision": n_tp / (n_tp + n_fp) if (n_tp + n_fp) else 0.0,
        "fdr": overall.fdr,
        "pre_aggregation_recall": pre_overall.recall,
        "pre_aggregation_fdr": pre_overall.fdr,
        "fdr_at_025": fdr_overall.fdr,
        "fdr_at_025_recall": fdr_overall.recall,
        "class_agnostic_tp": class_agnostic_tp,
        "class_agnostic_recall": class_agnostic_recall,
        "fp_cls": fp_cls,
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
        "conflict_objects": conflict_objects,
        "dedup_rate": 1.0 - n_objects / n_proposals if n_proposals else 0.0,
        "agg_s": None,
        "agg_s_note": "agg_s 口径待 E 成员确认，本脚本不复现",
        "input_sha256": {
            "proposals": _sha256(args.proposals),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
        "aggregate_params": {
            "cluster_eps": args.cluster_eps,
            "merge_iou": args.merge_iou,
            "nms_iou": args.nms_iou,
        },
        "macro_note": "macro 为官方 V1.6 口径：大类内细类简单平均",
        "口径说明": (
            "recall/fdr/macro 为官方细类匹配口径；class_agnostic_recall 复现 E 原始"
            " wp3 存档的 recall(~0.9694)，即定位召回、不校验细类。两者之差为 FP_CLS。"
        ),
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
