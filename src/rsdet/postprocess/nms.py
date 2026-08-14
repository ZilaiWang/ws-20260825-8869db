"""Deterministic, framework-independent non-maximum suppression."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


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
    keep: list[int] = []
    while order:
        current = order[0]
        keep.append(current)
        current_box = validated_boxes[current]
        order = [
            index
            for index in order[1:]
            if _iou_validated(current_box, validated_boxes[index]) <= threshold
        ]
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
            if (
                "category_id" not in record
                or "score" not in record
                or "bbox_xyxy" not in record
            ):
                raise ValueError(
                    f"image_id={image_id} prediction[{index}] missing required field"
                )
            try:
                category_id = int(record["category_id"])
                score = float(record["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"image_id={image_id} prediction[{index}] category/score invalid"
                ) from error
            if not math.isfinite(score):
                raise ValueError(
                    f"image_id={image_id} prediction[{index}] score must be finite"
                )
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


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for two valid, positive-area ``xyxy`` boxes."""
    return _iou_validated(
        _validated_box(box_a, name="box_a"),
        _validated_box(box_b, name="box_b"),
    )


__all__ = ["class_aware_nms_predictions", "compute_iou", "nms"]
