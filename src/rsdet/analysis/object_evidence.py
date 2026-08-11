"""N0-3：Pred-OOF 对象证据 manifest。

目的
----
把 M1 三折 OOF 的全部候选（55,548 个）转换为统一的、可追溯的对象证据层，
成为 P03（对象分类）、P05（背景拒识）、困难门控、最终对象学生的共同唯一输入。

每个候选记录：
- 身份：proposal_uid、image_id、fold、source_group、checkpoint_sha256
- 预测：细类、分数、框
- 官方匹配状态：TP / FP（细类约束）
- 错误类型（FP 时）：FP_DUP / FP_CLS / FP_LOC / FP_BG
- 几何证据：GT 匹配信息、oracle 匹配信息、尺寸档、边界风险
- crop 校验和（可选，由外部传入）

三种视图：
- ``oracle_positive``：不看预测细类、按 IoU 找最佳可定位 GT（oracle 匹配），
  候选确实覆盖了某个目标的"真实上限"证据。
- ``deployable_positive``：正式候选链（细类匹配）中实际保留的 TP 对象。
- ``hard_negative``：FP_BG 与高风险未归因候选（背景拒识的训练素材）。

约束
----
- 官方匹配用 ``evaluate_predictions_with_trace``，与评估完全同源；
- oracle 匹配复用 N0-2 的几何匹配逻辑（同大类 + IoU）；
- manifest 是**不可变输入**：生成后不得修改，后续模块按版本引用。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from rsdet.analysis.decoupled_errors import (
    compute_oracle_localization,
)
from rsdet.evaluation.official_metric import (
    evaluate_predictions_with_trace,
)
from rsdet.evaluation.protocol import EvaluationProtocol

# FP 错误类型（与 oof_detection.decompose_official_errors 保持一致）。
FP_DUP = "FP_DUP"
FP_CLS = "FP_CLS"
FP_LOC = "FP_LOC"
FP_BG = "FP_BG"


def _build_gt_lookup(
    gt_boxes: Mapping[int, list[dict[str, Any]]],
) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (image_id, index): record
        for image_id, records in gt_boxes.items()
        for index, record in enumerate(records)
    }


def _size_bin(bbox_xyxy: list[float]) -> str:
    x0, y0, x1, y1 = (float(value) for value in bbox_xyxy)
    short_edge = min(x1 - x0, y1 - y0)
    if short_edge < 16:
        return "tiny_lt16"
    if short_edge < 32:
        return "small_16_32"
    if short_edge < 64:
        return "medium_32_64"
    return "large_ge64"


def _classify_fp(
    record: dict[str, Any],
    gt_lookup: dict[tuple[int, int], dict[str, Any]],
    protocol: EvaluationProtocol,
    *,
    image_id: int,
) -> str:
    """对一条未匹配预测归类 FP 类型（与 oof_detection 语义对齐）。

    优先级：
    1. 与任一 GT 细类相同且 IoU >= 阈值但已被其他预测抢占 → FP_DUP；
    2. 与任一 GT 的框重叠（IoU >= 阈值）但细类不同 → FP_CLS；
    3. 与任一 GT 框有部分重叠但 IoU < 阈值 → FP_LOC；
    4. 其余 → FP_BG。
    """
    pred_box = [float(value) for value in record["bbox_xyxy"]]
    pred_category = int(record["category_id"])

    best_iou = 0.0
    best_category: int | None = None
    best_gt_key: tuple[int, int] | None = None
    for gt_key, gt_record in gt_lookup.items():
        gt_image_id, _ = gt_key
        if gt_image_id != image_id:
            continue
        gt_box = [float(value) for value in gt_record["bbox_xyxy"]]
        iou = _iou(pred_box, gt_box)
        if iou > best_iou:
            best_iou = iou
            best_category = int(gt_record["category_id"])
            best_gt_key = gt_key

    if best_gt_key is None or best_iou == 0.0:
        return FP_BG
    gt_coarse = protocol.category_mapping.get(best_category)
    threshold = protocol.iou_thresholds.get(gt_coarse, 0.5)
    if best_iou >= threshold:
        if best_category == pred_category:
            # 同细类且够 IoU，说明 GT 已被别的预测匹配（重框）。
            return FP_DUP
        return FP_CLS
    # 有重叠但低于阈值。
    overlap = best_iou > 0.0
    return FP_LOC if overlap else FP_BG


def _iou(box_a: list[float], box_b: list[float]) -> float:
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _score_of(record: dict[str, Any]) -> float:
    return float(record["score"])


def build_object_evidence_manifest(
    *,
    gt_boxes: Mapping[int, list[dict[str, Any]]],
    predictions: Mapping[int, list[dict[str, Any]]],
    protocol: EvaluationProtocol,
    threshold: float,
    image_folds: Mapping[int, int],
    checkpoint_sha256: Mapping[int, str],
    image_groups: Mapping[int, str] | None = None,
    proposal_uids: Mapping[int, list[str]] | None = None,
) -> dict[str, Any]:
    """构建统一对象证据 manifest。

    Args:
        gt_boxes: ``{image_id: [{bbox_xyxy, category_id, ...}]}``。
        predictions: ``{image_id: [{bbox_xyxy, category_id, score}]}``。
        threshold: 工作点 score 阈值。
        image_folds: ``{image_id: fold}``。
        checkpoint_sha256: ``{image_id: checkpoint_sha256}``（每折一个）。
        image_groups: ``{image_id: source_group}``（可选）。
        proposal_uids: ``{image_id: [proposal_uid, ...]}``，与预测顺序一一对应
            （可选；缺省时用 ``m1-f{fold}-i{image_id}-p{index}`` 生成）。

    Returns:
        manifest 字典：``records``、``views``、``summary``、``meta``。
    """
    # 官方匹配（细类约束）。
    metrics, trace = evaluate_predictions_with_trace(
        dict(gt_boxes),
        {
            image_id: [
                {**record, "source_prediction_index": source_index}
                for source_index, record in enumerate(records)
                if _score_of(record) >= threshold
            ]
            for image_id, records in predictions.items()
        },
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )

    # oracle 几何匹配（同大类）。
    oracle_metrics, _ = compute_oracle_localization(
        gt_boxes,
        predictions,
        protocol=protocol,
        threshold=threshold,
    )
    oracle_match_by_prediction: dict[tuple[int, int], dict[str, Any]] = {}
    for match in oracle_metrics.matches:
        oracle_match_by_prediction[(int(match["image_id"]), int(match["prediction_index"]))] = match

    gt_lookup = _build_gt_lookup(gt_boxes)

    # prediction_index 保留原始低阈值 OOF 列表的位置，避免过滤后错位。
    matched_pred_keys: set[tuple[int, int]] = set()
    for match in trace.matches:
        matched_pred_keys.add((match.image_id, match.prediction_index))

    records: list[dict[str, Any]] = []
    for image_id, rec_list in predictions.items():
        fold = image_folds[image_id]
        ckpt = checkpoint_sha256.get(image_id, "")
        group = image_groups.get(image_id, "") if image_groups else ""
        for source_index, record in enumerate(rec_list):
            if _score_of(record) < threshold:
                continue
            key = (image_id, source_index)
            matched = key in matched_pred_keys

            category_id = int(record["category_id"])
            box = [float(value) for value in record["bbox_xyxy"]]
            uid = None
            if proposal_uids is not None and image_id in proposal_uids:
                if source_index < len(proposal_uids[image_id]):
                    uid = proposal_uids[image_id][source_index]
            if uid is None:
                uid = f"m1-f{fold}-i{image_id}-p{source_index}"

            gt_key: tuple[int, int] | None = None
            matched_iou: float | None = None
            gt_category: int | None = None
            error_type: str | None = None
            if matched:
                match = None
                for m in trace.matches:
                    if m.image_id == key[0] and m.prediction_index == key[1]:
                        match = m
                        break
                if match is not None:
                    gt_key = (match.image_id, match.ground_truth_index)
                    matched_iou = match.iou
                    gt_category = gt_lookup[gt_key]["category_id"]
                    error_type = "TP"
            else:
                # 未匹配 → 归类 FP。
                error_type = _classify_fp(
                    record,
                    gt_lookup,
                    protocol,
                    image_id=image_id,
                )

            oracle_hit = False
            oracle_iou: float | None = None
            oracle_gt_key: tuple[int, int] | None = None
            oracle_gt_category: int | None = None
            oracle_info = oracle_match_by_prediction.get(key)
            if oracle_info is not None:
                oracle_hit = True
                oracle_iou = float(oracle_info["iou"])
                oracle_gt_key = (int(oracle_info["image_id"]), int(oracle_info["index"]))
                oracle_gt_category = int(oracle_info["gt_category_id"])

            records.append(
                {
                    "proposal_uid": uid,
                    "image_id": image_id,
                    "fold": fold,
                    "source_group": group,
                    "checkpoint_sha256": ckpt,
                    "category_id": category_id,
                    "source_prediction_index": source_index,
                    "score": _score_of(record),
                    "bbox_xyxy": [round(value, 4) for value in box],
                    "size_bin": _size_bin(box),
                    "official_status": error_type,
                    "matched_gt_key": (f"{gt_key[0]}:{gt_key[1]}" if gt_key else None),
                    "matched_gt_category": gt_category,
                    "matched_iou": (round(matched_iou, 4) if matched_iou is not None else None),
                    "oracle_hit": oracle_hit,
                    "oracle_gt_key": (
                        f"{oracle_gt_key[0]}:{oracle_gt_key[1]}"
                        if oracle_gt_key is not None
                        else None
                    ),
                    "oracle_iou": (round(oracle_iou, 4) if oracle_iou is not None else None),
                    "oracle_gt_category": oracle_gt_category,
                }
            )

    views = _build_views(records)
    summary = _build_summary(records, views, metrics)
    meta = {
        "manifest_version": "pred_oof_evidence_v2",
        "oracle_assignment_policy": "candidate_specific_same_coarse_greedy_v2",
        "fp_subtype_policy": "nearest_overlap_diagnostic_v1",
        "fp_subtypes_are_official": False,
        "threshold": threshold,
        "official_recall": metrics.recall,
        "official_fdr": metrics.fdr,
        "official_tp": int(metrics.details["tp"]),
        "official_fp": int(metrics.details["fp"]),
        "official_fn": int(metrics.details["fn"]),
    }
    return {
        "meta": meta,
        "records": records,
        "views": views,
        "summary": summary,
    }


def _build_views(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """构建三种视图。"""
    oracle_positive: list[dict[str, Any]] = []
    deployable_positive: list[dict[str, Any]] = []
    hard_negative: list[dict[str, Any]] = []
    for record in records:
        if record["official_status"] == "TP":
            deployable_positive.append(record)
        if record["oracle_hit"]:
            oracle_positive.append(record)
        if record["official_status"] in (FP_BG, FP_DUP, FP_CLS, FP_LOC):
            if record["official_status"] == FP_BG or not record["oracle_hit"]:
                hard_negative.append(record)
    return {
        "oracle_positive": oracle_positive,
        "deployable_positive": deployable_positive,
        "hard_negative": hard_negative,
    }


def _build_summary(
    records: list[dict[str, Any]],
    views: dict[str, list[dict[str, Any]]],
    metrics: Any,
) -> dict[str, Any]:
    """汇总计数与分类统计。"""
    status_counts: Counter[str] = Counter(record["official_status"] for record in records)
    fp_types = {name: status_counts.get(name, 0) for name in (FP_DUP, FP_CLS, FP_LOC, FP_BG)}
    total_fp = sum(fp_types.values())
    return {
        "total_candidates": len(records),
        "official_tp": status_counts.get("TP", 0),
        "official_fp": total_fp,
        "fp_by_type": fp_types,
        "view_counts": {
            "oracle_positive": len(views["oracle_positive"]),
            "deployable_positive": len(views["deployable_positive"]),
            "hard_negative": len(views["hard_negative"]),
        },
        "official_recall": metrics.recall,
        "official_fdr": metrics.fdr,
    }


def write_manifest_json(manifest: dict[str, Any], path: Path) -> None:
    """原子写入 manifest JSON。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def manifest_sha256(path: Path) -> str:
    """计算 manifest 文件的 SHA256。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
