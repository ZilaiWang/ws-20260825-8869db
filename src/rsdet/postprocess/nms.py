"""Deterministic, framework-independent non-maximum suppression."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


def _validated_box(box: Sequence[float], *, name: str) -> tuple[float, float, float, float]:
    if len(box) != 4:
        raise ValueError(f"{name} must contain exactly four coordinates")
    try:
        values = tuple(float(value) for value in box)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} coordinates must be numeric") from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} coordinates must be finite")
    x1, y1, x2, y2 = values
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"{name} must satisfy x2 > x1 and y2 > y1")
    return x1, y1, x2, y2


def _iou_validated(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection_width = max(0.0, x2 - x1)
    intersection_height = max(0.0, y2 - y1)
    intersection = intersection_width * intersection_height
    if intersection == 0.0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return intersection / union


def nms(
    boxes: Sequence[Sequence[float]],
    scores: Sequence[float],
    iou_threshold: float,
) -> list[int]:
    """Return the indices retained by greedy NMS.

    Results are ordered by descending score. Equal-score boxes retain their
    original order, making repeated runs deterministic. A lower-ranked box is
    suppressed only when its IoU is strictly greater than ``iou_threshold``.
    """
    if len(boxes) != len(scores):
        raise ValueError("boxes and scores must have the same length")
    try:
        threshold = float(iou_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("iou_threshold must be numeric") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and within [0, 1]")

    validated_boxes = [
        _validated_box(box, name=f"boxes[{index}]") for index, box in enumerate(boxes)
    ]
    validated_scores: list[float] = []
    for index, score in enumerate(scores):
        try:
            numeric_score = float(score)
        except (TypeError, ValueError) as error:
            raise ValueError(f"scores[{index}] must be numeric") from error
        if not math.isfinite(numeric_score):
            raise ValueError(f"scores[{index}] must be finite")
        validated_scores.append(numeric_score)

    order = sorted(
        range(len(validated_boxes)),
        key=lambda index: (-validated_scores[index], index),
    )
    # 批量 IoU 的贪心 NMS（与旧纯 Python 实现语义完全一致：降序、严格
    # IoU > threshold 才抑制、同分保序），仅将逐框 Python IoU 换成
    # numpy vectorized 计算，避免高候选量（如 RT-DETR 低阈值）时 O(n^2)
    # 纯 Python 循环成为不可接受的瓶颈。
    if not order:
        return []
    boxes_arr = np.asarray(validated_boxes, dtype=np.float64)  # (n, 4) x1 y1 x2 y2
    order = np.asarray(order, dtype=np.int64)
    keep: list[int] = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        box = boxes_arr[current]
        rest_boxes = boxes_arr[rest]
        x1 = np.maximum(box[0], rest_boxes[:, 0])
        y1 = np.maximum(box[1], rest_boxes[:, 1])
        x2 = np.minimum(box[2], rest_boxes[:, 2])
        y2 = np.minimum(box[3], rest_boxes[:, 3])
        intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area_a = (box[2] - box[0]) * (box[3] - box[1])
        area_b = (rest_boxes[:, 2] - rest_boxes[:, 0]) * (rest_boxes[:, 3] - rest_boxes[:, 1])
        union = area_a + area_b - intersection
        iou = np.divide(
            intersection, union, out=np.zeros_like(union), where=union > 0.0
        )
        order = rest[iou <= threshold]
    return keep


def class_aware_nms_predictions(
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    iou_threshold: float,
    *,
    category_ids: Sequence[int] | None = None,
) -> dict[int, list[dict[str, Any]]]:
    """Apply deterministic NMS independently per image and fine category.

    This operation is intended for a *post-reranking* stage: an upstream model
    may have changed ``category_id`` after detector NMS, invalidating the
    detector's original class-aware suppression result.  Cross-category boxes
    are deliberately never compared.  When ``category_ids`` is provided,
    categories outside that allowlist bypass suppression exactly.  Records
    within a retained image are returned in descending score order, with their
    original position as the deterministic tie-breaker.

    The function copies output records and never mutates its input.  Empty
    image lists are retained so callers can preserve an explicit image ledger.
    """

    try:
        threshold = float(iou_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("iou_threshold must be numeric") from error
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and within [0, 1]")
    target_categories = (
        None if category_ids is None else frozenset(int(value) for value in category_ids)
    )

    output: dict[int, list[dict[str, Any]]] = {}
    for raw_image_id, raw_records in predictions.items():
        image_id = int(raw_image_id)
        records = list(raw_records)
        groups: dict[int, list[int]] = {}
        scores: list[float] = []
        boxes: list[tuple[float, float, float, float]] = []
        for index, record in enumerate(records):
            if "category_id" not in record or "score" not in record or "bbox_xyxy" not in record:
                raise ValueError(f"image_id={image_id} prediction[{index}] missing required field")
            try:
                category_id = int(record["category_id"])
                score = float(record["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"image_id={image_id} prediction[{index}] category/score invalid"
                ) from error
            if not math.isfinite(score):
                raise ValueError(f"image_id={image_id} prediction[{index}] score must be finite")
            box = _validated_box(
                record["bbox_xyxy"],
                name=f"predictions[{image_id}][{index}].bbox_xyxy",
            )
            groups.setdefault(category_id, []).append(index)
            scores.append(score)
            boxes.append(box)

        kept: list[int] = []
        for category_id in sorted(groups):
            indices = groups[category_id]
            if target_categories is not None and category_id not in target_categories:
                kept.extend(indices)
                continue
            local_keep = nms(
                [boxes[index] for index in indices],
                [scores[index] for index in indices],
                threshold,
            )
            kept.extend(indices[local_index] for local_index in local_keep)
        kept.sort(key=lambda index: (-scores[index], index))
        output[image_id] = [dict(records[index]) for index in kept]
    return output


def category_threshold_nms_predictions(
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    category_iou_thresholds: Mapping[int, float],
) -> dict[int, list[dict[str, Any]]]:
    """Apply deterministic NMS with a frozen IoU threshold per fine category.

    Categories absent from ``category_iou_thresholds`` bypass suppression.
    Threshold groups are applied in ascending category order.  Because each
    category belongs to exactly one group and NMS never compares categories,
    this is equivalent to a single per-category pass while reusing the audited
    :func:`class_aware_nms_predictions` implementation.
    """

    normalized: dict[int, float] = {}
    for raw_category, raw_threshold in category_iou_thresholds.items():
        category = int(raw_category)
        if category in normalized:
            raise ValueError(f"duplicate category threshold: {category}")
        try:
            threshold = float(raw_threshold)
        except (TypeError, ValueError) as error:
            raise ValueError(f"threshold for category {category} must be numeric") from error
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold for category {category} must be finite and within [0, 1]")
        normalized[category] = threshold

    result = {
        int(image_id): [dict(record) for record in records]
        for image_id, records in predictions.items()
    }
    by_threshold: dict[float, list[int]] = {}
    for category, threshold in normalized.items():
        by_threshold.setdefault(threshold, []).append(category)
    for threshold in sorted(by_threshold, reverse=True):
        result = class_aware_nms_predictions(
            result,
            threshold,
            category_ids=sorted(by_threshold[threshold]),
        )
    return result


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for two valid, positive-area ``xyxy`` boxes."""
    return _iou_validated(
        _validated_box(box_a, name="box_a"),
        _validated_box(box_b, name="box_b"),
    )


__all__ = [
    "category_threshold_nms_predictions",
    "class_aware_nms_predictions",
    "compute_iou",
    "nms",
]
