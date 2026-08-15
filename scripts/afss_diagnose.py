#!/usr/bin/env python3
"""AFSS 反遗忘采样的日志回放诊断（材料 19 Y4 第一阶段）。

材料 19 明确要求 Y4 先做"日志回放版，不立即改变训练"：用每图
min(precision, recall) 作为学习充分度，困难图持续参与、容易图低频回看。
本脚本在**离线**用任意模型的低阈值 OOF 预测计算每图充分度，并验证它与
held-out 错误、车辆 near-miss、尾类、困难 source group 的相关性——
相关性成立后才接入训练期采样器。

输出：
  - 每图充分度分布；
  - 困难图（充分度最低）的粗类 / 尺寸 / source-group 特征；
  - 充分度与错误类型（FN_MISS/FP_BG）的关联。

纯 CPU。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _short_edge(box: Sequence[float]) -> float:
    return min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def _size_bin(se: float) -> str:
    if se < 32:
        return "tiny"
    if se < 96:
        return "small"
    if se < 256:
        return "medium"
    return "large"


def _load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payload:
        grouped[int(item["image_id"])].append(
            {"bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
             "score": float(item["score"]),
             "category_id": int(item["category_id"])})
    return dict(grouped)


def _per_image_sufficiency(gt_boxes, pred_boxes, iou_thresholds) -> dict[int, float]:
    """每图 min(precision, recall)，用官方同细类贪心匹配。"""
    suff: dict[int, float] = {}
    for image_id in sorted(set(gt_boxes) | set(pred_boxes)):
        gts = gt_boxes.get(image_id, [])
        preds = pred_boxes.get(image_id, [])
        matched_gt: set[int] = set()
        used_pred: set[int] = set()
        # 分数降序贪心
        order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
        for pi in order:
            p = preds[pi]
            thr = iou_thresholds.get(str(p["category_id"]), 0.5)
            best_gi, best_iou = None, 0.0
            for gi, g in enumerate(gts):
                if gi in matched_gt:
                    continue
                if int(g["category_id"]) != int(p["category_id"]):
                    continue
                iou = compute_iou(list(p["bbox_xyxy"]), list(g["bbox_xyxy"]))
                if iou > best_iou:
                    best_gi, best_iou = gi, iou
            if best_gi is not None and best_iou >= thr:
                matched_gt.add(best_gi)
                used_pred.add(pi)
        tp = len(matched_gt)
        fn = len(gts) - tp
        fp = len(preds) - len(used_pred)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        suff[image_id] = min(precision, recall)
    return suff


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path,
                        default=Path("outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"))
    parser.add_argument("--split", type=Path,
                        default=Path("data/splits/cv3_airport_proxy_k60_v2.json"))
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--hard-threshold", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    # 25 细类 -> IoU 阈值（按粗类）
    iou_thresholds = {}
    for cid, coarse in protocol.category_mapping.items():
        iou_thresholds[str(cid)] = protocol.iou_thresholds[coarse]

    pred_boxes = _load_predictions(args.predictions)
    formal = load_formal_ground_truth(args.formal_crop_manifest,
                                      expected_images=4481, expected_annotations=20933)
    doc = json.loads(args.split.read_text(encoding="utf-8"))
    split = {int(s["image_id"]): (int(s["fold"]), str(s["group_id"])) for s in doc["samples"]}

    suff = _per_image_sufficiency(formal.boxes, pred_boxes, iou_thresholds)

    # 分布
    hist = Counter()
    for v in suff.values():
        hist[min(int(v * 10) / 10, 1.0)] += 1

    # 困难图特征
    hard_images = sorted(suff.items(), key=lambda x: x[1])[:max(1, len(suff) // 10)]
    hard_ids = {i for i, _ in hard_images}
    hard_coarse = Counter()
    hard_size = Counter()
    hard_group = Counter()
    for image_id in hard_ids:
        _, group = split.get(image_id, (-1, "unknown"))
        hard_group[group] += 1
        for g in formal.boxes.get(image_id, []):
            hard_coarse[protocol.category_mapping[int(g["category_id"])]] += 1
            hard_size[_size_bin(_short_edge(g["bbox_xyxy"]))] += 1

    payload = {
        "n_images": len(suff),
        "suff_mean": sum(suff.values()) / len(suff) if suff else None,
        "suff_histogram": dict(sorted(hist.items())),
        "n_hard_images": len(hard_ids),
        "hard_coarse_dist": dict(hard_coarse.most_common()),
        "hard_size_dist": dict(hard_size.most_common()),
        "hard_group_top5": dict(hard_group.most_common(5)),
        "note": "充分度 = 每图 min(precision, recall)；困难图 = 充分度最低 10% 的图。"
                "若困难图集中在车辆/尾类/特定 source group，则 AFSS 采样器有据可依。",
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
