"""N0-1：cross-fit 全局阈值基线。

问题背景
--------
M1 正式 OOF 的候选阈值（0.051）是在整份 OOF 预测上"选择又回评"得到的，
属于同 OOF 探索性阈值（``exploratory_only``），带有自证偏差。真实比赛场景
不允许在测试集上调阈值。

本模块实现严格 cross-fit：

1. 对每个 held-out fold，只使用**另外两折**的 OOF 预测在固定网格上扫描全局
   score 阈值，按内部目标（Recall >= recall_min 且 FDR <= fdr_max 的最优
   工作点）选出该次选择的最佳阈值；
2. 把该阈值原样应用到 held-out 折的预测上，计算官方指标（pooled 与官方
   macro 双口径）；
3. 合并三份 held-out 评估结果，得到无偏的正式 M1 baseline；
4. 记录三折阈值离散度、fold 达标情况与内部目标（FDR <= 0.17）达成情况。

约束
----
- 评估走 ``evaluate_predictions_with_trace``，与官方协议完全同源；
- 任何 fold 的预测都不能参与决定自身阈值；
- 阈值选择过程与 M1 审计报告中的同 OOF 探索一致（同一网格、同一内部目标），
  便于对照偏差量。
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rsdet.evaluation.official_metric import (
    OverallMetrics,
    evaluate_predictions_with_trace,
)
from rsdet.evaluation.protocol import EvaluationProtocol


def load_cv3_aggregate(
    aggregate_dir: str | Path,
    *,
    candidate_floor: float,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]], dict[int, int]]:
    """加载 OOF 聚合：元数据、预测、以及 image_id -> fold 映射。

    Returns:
        (metadata, predictions, image_folds)。
        predictions: ``{image_id: [COCO 预测记录]}``。
        image_folds: ``{image_id: fold_index}``。
    """
    root = Path(aggregate_dir).expanduser().resolve()
    metadata_path = root / "oof_metadata.json"
    predictions_path = root / "predictions_oof_low.json"
    images_path = root / "oof_images.csv"
    for path in (metadata_path, predictions_path, images_path):
        if not path.is_file():
            raise FileNotFoundError(f"OOF aggregate 缺少 {path.name}: {path}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    predictions_list = json.loads(predictions_path.read_text(encoding="utf-8"))
    if not isinstance(predictions_list, list):
        raise ValueError("predictions_oof_low.json 顶层必须是列表")

    image_folds: dict[int, int] = {}
    with images_path.open("r", encoding="utf-8", newline="") as handle:
        import csv

        reader = csv.DictReader(handle)
        for row in reader:
            image_id = int(row["image_id"])
            image_folds[image_id] = int(row["fold"])

    predictions: dict[int, list[dict[str, Any]]] = {}
    for record in predictions_list:
        image_id = int(record["image_id"])
        if image_id not in image_folds:
            raise ValueError(f"预测包含未知 image_id={image_id}（不在 oof_images 中）")
        normalized_record = dict(record)
        bbox = [float(value) for value in record["bbox"]]
        if len(bbox) != 4:
            raise ValueError(f"预测 bbox 必须是 4 个数值: {record}")
        normalized_record["bbox_xyxy"] = [
            bbox[0],
            bbox[1],
            bbox[0] + bbox[2],
            bbox[1] + bbox[3],
        ]
        normalized_record.pop("bbox", None)
        predictions.setdefault(image_id, []).append(normalized_record)

    floor = float(metadata.get("low_score_threshold", -1.0))
    if not math.isclose(floor, candidate_floor, abs_tol=1e-12):
        raise ValueError(
            f"OOF 低阈值 {floor} 与 candidate_floor {candidate_floor} 不一致"
        )
    return metadata, predictions, image_folds


def load_gt_from_formal_crop_manifest(
    manifest_path: str | Path,
    *,
    expected_images: int,
    expected_annotations: int,
) -> dict[int, list[dict[str, Any]]]:
    """从 formal crop manifest（``crop_policy=tight``）恢复官方 GT。

    返回 ``{image_id: [{bbox_xyxy, category_id}, ...]}``，兼容
    :func:`evaluate_predictions` 的输入。仅取 ``crop_policy=tight`` 行保证
    每个对象只有一条官方 GT。
    """
    import csv

    manifest = Path(manifest_path).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"formal crop manifest 不存在: {manifest}")

    boxes: dict[int, list[dict[str, Any]]] = {}
    annotation_uids: set[str] = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "crop_policy",
            "annotation_uid",
            "formal_image_id",
            "gt_x0",
            "gt_y0",
            "gt_x1",
            "gt_y1",
            "class_id",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"manifest 缺少列: {sorted(missing)}")
        for row in reader:
            if row["crop_policy"] != "tight":
                continue
            uid = row["annotation_uid"]
            if uid in annotation_uids:
                raise ValueError(f"formal manifest 出现重复 tight annotation: {uid}")
            annotation_uids.add(uid)
            image_id = int(row["formal_image_id"])
            boxes.setdefault(image_id, []).append(
                {
                    "bbox_xyxy": [
                        float(row["gt_x0"]),
                        float(row["gt_y0"]),
                        float(row["gt_x1"]),
                        float(row["gt_y1"]),
                    ],
                    "category_id": int(row["class_id"]),
                }
            )

    if len(annotation_uids) != expected_annotations:
        raise ValueError(
            f"formal GT 对象数 {len(annotation_uids)} != expected {expected_annotations}"
        )
    if len(boxes) != expected_images:
        raise ValueError(
            f"formal GT 图像数 {len(boxes)} != expected {expected_images}"
        )
    return boxes


def split_by_fold(
    predictions: Mapping[int, list[dict[str, Any]]],
    image_folds: Mapping[int, int],
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """按 fold 拆分预测：``{fold: {image_id: [预测]}}``。"""
    folded: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for image_id, records in predictions.items():
        fold = image_folds[image_id]
        folded.setdefault(fold, {})[image_id] = list(records)
    return folded


def split_gt_by_fold(
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    image_folds: Mapping[int, int],
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """按 fold 拆分 GT：``{fold: {image_id: [GT]}}``。"""
    folded: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for image_id, records in gt_boxes.items():
        fold = image_folds[image_id]
        folded.setdefault(fold, {})[image_id] = list(records)
    return folded


def _merge_folds(
    folded: Mapping[int, Mapping[int, list[dict[str, Any]]]],
    fold_indices: list[int],
) -> dict[int, list[dict[str, Any]]]:
    """把多个 fold 的 ``{image_id: [记录]}`` 合并为一个字典。"""
    merged: dict[int, list[dict[str, Any]]] = {}
    for fold in fold_indices:
        for image_id, records in folded[fold].items():
            merged[image_id] = list(records)
    return merged


def _filter_by_score(
    predictions: Mapping[int, list[dict[str, Any]]],
    threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    """保留 score >= threshold 的预测。"""
    filtered: dict[int, list[dict[str, Any]]] = {}
    for image_id, records in predictions.items():
        kept = [
            record for record in records if float(record["score"]) >= threshold
        ]
        if kept:
            filtered[image_id] = kept
    return filtered


def evaluate_workpoint(
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    *,
    threshold: float,
    protocol: EvaluationProtocol,
) -> OverallMetrics:
    """在给定阈值下评估官方指标（pooled 口径）。"""
    filtered = _filter_by_score(predictions, threshold)
    metrics, _ = evaluate_predictions_with_trace(
        dict(gt_boxes),
        filtered,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    return metrics


def scan_global_threshold(
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    *,
    protocol: EvaluationProtocol,
    threshold_start: float = 0.001,
    threshold_stop: float = 1.0,
    threshold_step: float = 0.01,
    internal_recall_min: float = 0.85,
    internal_fdr_max: float = 0.20,
) -> tuple[float, OverallMetrics, list[dict[str, Any]]]:
    """在网格上扫描全局阈值，返回最优工作点及其曲线。

    最优定义（与 M1 同 OOF 探索一致）：在满足
    ``Recall >= internal_recall_min`` 且 ``FDR <= internal_fdr_max`` 的
    阈值中，选择 Recall 最大者；若没有同时满足的阈值，选择 Recall 最大者。

    Returns:
        (best_threshold, best_metrics, curve)。
        curve 每项: ``{threshold, recall, fdr, tp, fp, fn, gate_passed}``。
    """
    curve: list[dict[str, Any]] = []
    best_threshold = threshold_start
    best_metrics: OverallMetrics | None = None
    best_recall = -math.inf

    threshold = threshold_start
    while threshold <= threshold_stop + 1e-12:
        metrics = evaluate_workpoint(
            gt_boxes,
            predictions,
            threshold=threshold,
            protocol=protocol,
        )
        gate_passed = (
            metrics.recall >= internal_recall_min
            and metrics.fdr <= internal_fdr_max
        )
        curve.append(
            {
                "threshold": threshold,
                "recall": metrics.recall,
                "fdr": metrics.fdr,
                "tp": int(metrics.details.get("tp", 0)),
                "fp": int(metrics.details.get("fp", 0)),
                "fn": int(metrics.details.get("fn", 0)),
                "gate_passed": gate_passed,
            }
        )
        candidate_rank = (1.0 if gate_passed else 0.0, metrics.recall)
        best_rank = (
            1.0 if (
                best_metrics is not None
                and best_metrics.recall >= internal_recall_min
                and best_metrics.fdr <= internal_fdr_max
            ) else 0.0,
            best_recall,
        )
        if candidate_rank > best_rank:
            best_threshold = threshold
            best_metrics = metrics
            best_recall = metrics.recall
        threshold += threshold_step

    assert best_metrics is not None
    return best_threshold, best_metrics, curve


def run_crossfit(
    *,
    aggregate_dir: str | Path,
    formal_crop_manifest_path: str | Path,
    protocol: EvaluationProtocol,
    expected_images: int,
    expected_annotations: int,
    candidate_floor: float = 0.001,
    threshold_start: float = 0.001,
    threshold_stop: float = 1.0,
    threshold_step: float = 0.01,
    internal_recall_min: float = 0.85,
    internal_fdr_max: float = 0.20,
    internal_fdr_strict: float = 0.17,
) -> dict[str, Any]:
    """执行严格 cross-fit 阈值基线。

    Returns:
        结果字典（含 per-fold 与合并工作点、阈值离散度、达标情况），可直接
        序列化为 JSON。
    """
    metadata, predictions, image_folds = load_cv3_aggregate(
        aggregate_dir,
        candidate_floor=candidate_floor,
    )
    gt_boxes = load_gt_from_formal_crop_manifest(
        formal_crop_manifest_path,
        expected_images=expected_images,
        expected_annotations=expected_annotations,
    )

    fold_indices = sorted(image_folds.values())
    fold_set = set(fold_indices)
    gt_images = set(gt_boxes)
    pred_images = set(predictions)
    if gt_images != pred_images:
        raise ValueError(
            "GT 与预测的 image_id 集合不一致: "
            f"GT-only={len(gt_images - pred_images)}, pred-only={len(pred_images - gt_images)}"
        )
    if len(fold_set) != 3:
        raise ValueError(f"cross-fit 需要 3 折，实际 {len(fold_set)} 折")

    folded_pred = split_by_fold(predictions, image_folds)
    folded_gt = split_gt_by_fold(gt_boxes, image_folds)

    per_fold_results: list[dict[str, Any]] = []
    heldout_gt: dict[int, list[dict[str, Any]]] = {}
    heldout_pred: dict[int, list[dict[str, Any]]] = {}

    for held_out in sorted(fold_set):
        selection_folds = sorted(fold_set - {held_out})
        selection_gt = _merge_folds(folded_gt, selection_folds)
        selection_pred = _merge_folds(folded_pred, selection_folds)
        threshold, selection_metrics, curve = scan_global_threshold(
            selection_gt,
            selection_pred,
            protocol=protocol,
            threshold_start=threshold_start,
            threshold_stop=threshold_stop,
            threshold_step=threshold_step,
            internal_recall_min=internal_recall_min,
            internal_fdr_max=internal_fdr_max,
        )

        hold_gt = _merge_folds(folded_gt, [held_out])
        hold_pred = _merge_folds(folded_pred, [held_out])
        hold_metrics = evaluate_workpoint(
            hold_gt,
            hold_pred,
            threshold=threshold,
            protocol=protocol,
        )
        for image_id, records in hold_gt.items():
            heldout_gt[image_id] = list(records)
        for image_id, records in hold_pred.items():
            heldout_pred[image_id] = list(records)

        per_fold_results.append(
            {
                "held_out_fold": held_out,
                "selection_folds": selection_folds,
                "selected_threshold": threshold,
                "selection_curve": curve,
                "selection_recall": selection_metrics.recall,
                "selection_fdr": selection_metrics.fdr,
                "held_out_recall": hold_metrics.recall,
                "held_out_fdr": hold_metrics.fdr,
                "held_out_tp": int(hold_metrics.details.get("tp", 0)),
                "held_out_fp": int(hold_metrics.details.get("fp", 0)),
                "held_out_fn": int(hold_metrics.details.get("fn", 0)),
                "held_out_gate_passed": (
                    hold_metrics.recall >= internal_recall_min
                    and hold_metrics.fdr <= internal_fdr_max
                ),
                "held_out_internal_fdr_passed": (
                    hold_metrics.fdr <= internal_fdr_strict
                ),
            }
        )

    # 合并三份 held-out 预测，用逐折所选阈值分别评估后合并 TP/FP/FN。
    merged_metrics = evaluate_workpoint(
        heldout_gt,
        heldout_pred,
        threshold=min(
            float(result["selected_threshold"]) for result in per_fold_results
        ),
        protocol=protocol,
    )
    # 注意：合并评估不能使用单一阈值（每折阈值不同），这里直接用各自折的
    # held-out 指标做加权合并。
    total_tp = sum(result["held_out_tp"] for result in per_fold_results)
    total_fp = sum(result["held_out_fp"] for result in per_fold_results)
    total_fn = sum(result["held_out_fn"] for result in per_fold_results)
    merged_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) else 0.0
    merged_fdr = total_fp / (total_tp + total_fp) if (total_tp + total_fp) else 0.0

    thresholds = [result["selected_threshold"] for result in per_fold_results]
    threshold_mean = sum(thresholds) / len(thresholds)
    threshold_spread = max(thresholds) - min(thresholds)
    threshold_std = math.sqrt(
        sum((t - threshold_mean) ** 2 for t in thresholds) / len(thresholds)
    )

    gate_folds = [result["held_out_gate_passed"] for result in per_fold_results]
    strict_folds = [
        result["held_out_internal_fdr_passed"] for result in per_fold_results
    ]

    return {
        "analysis": "N0-1_crossfit_threshold_baseline",
        "model_key": metadata.get("model_key"),
        "seed": metadata.get("seed"),
        "cv3_manifest_sha256": metadata.get("source_manifest_sha256"),
        "candidate_floor": candidate_floor,
        "threshold_grid": {
            "start": threshold_start,
            "stop": threshold_stop,
            "step": threshold_step,
        },
        "internal_target": {
            "recall_min": internal_recall_min,
            "fdr_max": internal_fdr_max,
            "fdr_strict": internal_fdr_strict,
        },
        "image_count": len(gt_images),
        "ground_truth_objects": sum(map(len, gt_boxes.values())),
        "per_fold": per_fold_results,
        "merged_held_out": {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "recall": merged_recall,
            "fdr": merged_fdr,
            "official_gate_passed": (
                merged_recall >= internal_recall_min
                and merged_fdr <= internal_fdr_max
            ),
            "internal_fdr_passed": merged_fdr <= internal_fdr_strict,
        },
        "threshold_dispersion": {
            "mean": threshold_mean,
            "std": threshold_std,
            "spread": threshold_spread,
            "values": thresholds,
        },
        "fold_gate_pass_count": sum(1 for p in gate_folds if p),
        "fold_internal_fdr_pass_count": sum(1 for p in strict_folds if p),
        "note": (
            "merged held-out 指标由三折各自阈值下的 TP/FP/FN 直接汇总；"
            "非在同一阈值下评估。"
        ),
        "merged_metrics_single_min_threshold": {
            "recall": merged_metrics.recall,
            "fdr": merged_metrics.fdr,
            "tp": int(merged_metrics.details.get("tp", 0)),
            "fp": int(merged_metrics.details.get("fp", 0)),
            "fn": int(merged_metrics.details.get("fn", 0)),
            "threshold_used": min(thresholds),
            "note": "参考用：用最小阈值合并所有 held-out 预测的单阈值评估",
        },
    }
