#!/usr/bin/env python3
"""E 主线 3：WP3 真实评估的补充口径——score 阈值扫描 + 聚合前基线。

补材料 37 评审缺陷 2（报告 "FDR(score≥0.25)=3.83%" 与 "聚合前 Recall 98.4%"
无脚本/存档支撑）：

- 对同一份 OOF proposals 分别按「原始 proposal（聚合前）」与「聚合后对象」
  两条链路，在 score 阈值 [0.0, 0.25, 0.5] 下计算官方细类口径的 Recall/FDR；
- 复用 ``rsdet.evaluation.official_metric`` 官方 evaluator，口径与
  ``eval_wp3_global_aggregation.py`` 完全一致；
- 输出 ``wp3_threshold_baseline.json``，含全部输入 SHA，保证可复现。

用法：
    python scripts/eval_wp3_threshold_baseline.py \
        --proposals outputs/.../oof_proposals.csv \
        --formal-crop-manifest outputs/.../formal_crop_manifest.csv \
        --output outputs/e_wp3/wp3_threshold_baseline.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.global_aggregation import aggregate
from rsdet.utils.config import load_config

SCORE_THRESHOLDS = (0.0, 0.25, 0.5)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_proposals(path: Path) -> tuple[int, dict[int, list[dict[str, Any]]]]:
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


def _to_records(objects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "bbox_xyxy": _xywh_to_xyxy(obj),
            "score": float(obj["score"]),
            "category_id": int(obj["category_id"]),
        }
        for obj in objects
    ]


def _filter_score(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    return [r for r in records if r["score"] >= threshold]


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

    # ---- 聚合前（原始 proposal）链路 ----
    pre_agg: dict[int, list[dict[str, Any]]] = {
        img_id: _to_records(props) for img_id, props in grouped.items()
    }
    # ---- 聚合后对象链路 ----
    post_agg: dict[int, list[dict[str, Any]]] = {}
    n_objects = 0
    for image_id, proposals in grouped.items():
        objects = aggregate(
            proposals,
            cluster_eps=args.cluster_eps,
            merge_iou=args.merge_iou,
            nms_iou=args.nms_iou,
        )
        n_objects += len(objects)
        post_agg[image_id] = _to_records(objects)

    rows: list[dict[str, Any]] = []
    for threshold in SCORE_THRESHOLDS:
        for chain_name, pred_boxes in (
            ("pre_aggregation", pre_agg),
            ("post_aggregation", post_agg),
        ):
            filtered = {
                img_id: _filter_score(records, threshold)
                for img_id, records in pred_boxes.items()
            }
            total = sum(len(v) for v in filtered.values())
            overall, _ = evaluate_predictions_with_trace(
                formal.boxes,
                filtered,
                class_names=protocol.class_names,
                category_mapping=protocol.category_mapping,
                iou_thresholds=protocol.iou_thresholds,
            )
            rows.append(
                {
                    "score_threshold": threshold,
                    "chain": chain_name,
                    "n_boxes": total,
                    "recall": overall.recall,
                    "precision": 1.0 - overall.fdr if overall.fdr is not None else None,
                    "fdr": overall.fdr,
                }
            )

    payload = {
        "n_images": len(grouped),
        "n_proposals": n_proposals,
        "n_objects": n_objects,
        "n_gt": formal.annotation_count,
        "rows": rows,
        "aggregate_params": {
            "cluster_eps": args.cluster_eps,
            "merge_iou": args.merge_iou,
            "nms_iou": args.nms_iou,
        },
        "input_sha256": {
            "proposals": _sha256(args.proposals),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
        "口径说明": (
            "官方 V1.6 细类匹配口径：框 IoU 达标且细类正确才算 TP。"
            "pre_aggregation=原始 M1 proposal 直接评估；post_aggregation=聚合后对象。"
        ),
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
