#!/usr/bin/env python3
"""统一实验评估入口（所有 YOLO 变体 / 创新方案复用）。

材料 19 第 6 节「每个实验的统一输出」的可执行落地：输入任意一个实验的
低阈值 OOF 预测（list of {image_id, category_id, score, bbox_xyxy}），产出
可横向对比的统一指标 JSON，并落盘错误 cases 供诊断接口分析。

输出指标：
  - pooled Recall/FDR（官方细类匹配口径）
  - macro Recall/FDR（V1.6：大类内细类简单平均，即 4/20/1）
  - fold0/1/2 分层 Recall/FDR
  - 分层错误：FP_BG / FP_CLS / FP_DUP / FP_LOC / FN_MISS / FN_CLS / FN_LOC
  - 专项：车辆 Recall/FDR + 车辆 tiny/small 计数、HM/LQS/TU-160/F-22 各自 TP/FP/FN
  - 尺寸分布（tiny/small/medium/large，按 GT short_edge）

错误 cases 落盘为 `<output>.cases.json`，供 analyze_experiment_errors.py
按类/尺寸/source-group 定位问题。

纯 CPU。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from rsdet.analysis.oof_detection import (
    decompose_official_errors,
    load_formal_ground_truth,
)
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

FINE_NAMES = [
    "HM", "LQS", "QHS", "MS",
    "A1_SU-35", "A2_C-130", "A3_C-17", "A4_C-5", "A5_F-16", "A6_TU-160",
    "A7_E-3", "A8_B-52", "A9_P-3C", "A10_B-1B", "A11_E-8", "A12_TU-22",
    "A13_F-15", "A14_KC-135", "A15_F-22", "A16_FA-18", "A17_TU-95",
    "A18_KC-10", "A19_SU-34", "A20_SU-24", "FSC",
]
# 少数样本 / 结构性短板专项（材料 19 明确点名）
FOCUS_CLASSES = {"HM": 0, "LQS": 1, "A6_TU-160": 9, "A15_F-22": 18}


def _short_edge(box: Sequence[float]) -> float:
    return min(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))


def _size_bin(short_edge: float) -> str:
    if short_edge < 32:
        return "tiny"
    if short_edge < 96:
        return "small"
    if short_edge < 256:
        return "medium"
    return "large"


def _load_predictions(path: Path) -> dict[int, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("predictions 必须是 list")
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in payload:
        grouped[int(item["image_id"])].append(
            {
                "bbox_xyxy": [float(v) for v in item["bbox_xyxy"]],
                "score": float(item["score"]),
                "category_id": int(item["category_id"]),
            }
        )
    return dict(grouped)


def _load_split(path: Path) -> dict[int, tuple[int, str]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    return {
        int(s["image_id"]): (int(s["fold"]), str(s["group_id"]))
        for s in doc["samples"]
    }


def _subset(boxes: dict[int, list], ids: set[int]) -> dict[int, list]:
    return {i: boxes[i] for i in ids if i in boxes}


def _pooled_eval(formal_boxes, pred_boxes, protocol) -> dict[str, Any]:
    from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
    overall, _trace = evaluate_predictions_with_trace(
        formal_boxes, pred_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    return {
        "recall": overall.recall,
        "fdr": overall.fdr,
        "tp": overall.details["tp"],
        "fp": overall.details["fp"],
        "fn": overall.details["fn"],
    }


def _build_cases_image_map(cases: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    out: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        out[int(c["image_id"])].append(c)
    return dict(out)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path,
                        default=Path("outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv"))
    parser.add_argument("--split", type=Path,
                        default=Path("data/splits/cv3_airport_proxy_k60_v2.json"))
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--model-key", type=str, default="EXPERIMENT")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    pred_boxes = _load_predictions(args.predictions)
    formal = load_formal_ground_truth(args.formal_crop_manifest,
                                      expected_images=4481, expected_annotations=20933)
    split = _load_split(args.split)

    # pooled + 分层错误 + cases（一次性，守恒）
    metrics, cases, _partner = decompose_official_errors(
        formal, pred_boxes, threshold=args.threshold,
        protocol=protocol, model_key=args.model_key, include_cases=True,
    )
    ranking = evaluate_ranking_metrics(
        formal.boxes, pred_boxes,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )

    # fold 分层
    folds: dict[int, dict[str, Any]] = {}
    for fold in (0, 1, 2):
        ids = {i for i, (f, _g) in split.items() if f == fold}
        fold_gt = _subset(formal.boxes, ids)
        fold_pred = _subset(pred_boxes, ids)
        folds[fold] = _pooled_eval(fold_gt, fold_pred, protocol)

    # 专项：焦点类 + 车辆 tiny/small
    focus: dict[str, dict[str, Any]] = {}
    case_gt_by_fine: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        if c["case_side"] == "ground_truth":
            case_gt_by_fine[int(c["category_id"])].append(c)
    for name, cid in FOCUS_CLASSES.items():
        # TP = GT 总数 - FN（FN 从 cases 的 ground_truth side 统计）
        gt_total = sum(1 for _g in formal.objects.values() if _g.category_id == cid)
        fn = sum(1 for c in case_gt_by_fine[cid] if c["reason"].startswith("FN"))
        # FP = 预测为该类但未匹配（从 cases 的 prediction side 统计）
        fp = sum(1 for c in cases
                 if c["case_side"] == "prediction"
                 and int(c["category_id"]) == cid)
        tp = gt_total - fn
        recall = tp / gt_total if gt_total else 0.0
        fdr = fp / (tp + fp) if (tp + fp) else 0.0
        focus[name] = {"gt": gt_total, "tp": tp, "fp": fp, "fn": fn,
                       "recall": recall, "fdr": fdr}

    vehicle_gt = [g for g in formal.objects.values() if g.category_id == 24]
    vehicle_tiny_small = sum(1 for g in vehicle_gt if _size_bin(_short_edge(g.bbox_xyxy)) in ("tiny", "small"))

    # 尺寸分布（GT 维度）
    size_counter = Counter(_size_bin(_short_edge(g.bbox_xyxy)) for g in formal.objects.values())

    payload = {
        "model_key": args.model_key,
        "threshold": args.threshold,
        "n_predictions": sum(len(v) for v in pred_boxes.values()),
        "pooled": _pooled_eval(formal.boxes, pred_boxes, protocol),
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
        "per_fold": folds,
        "error_breakdown": {
            "fp": dict(metrics.get("fp_counts", {})),
            "fn": dict(metrics.get("fn_counts", {})),
        },
        "focus_classes": focus,
        "vehicle_tiny_small_gt": vehicle_tiny_small,
        "vehicle_total_gt": len(vehicle_gt),
        "gt_size_distribution": dict(size_counter),
        "macro_note": "macro 为官方 V1.6 口径：大类内细类简单平均（4/20/1）",
    }

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases_path = output.with_suffix(".cases.json")
    cases_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\ncases 落盘: {cases_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
