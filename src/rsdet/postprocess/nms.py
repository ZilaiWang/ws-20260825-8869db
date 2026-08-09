"""Deterministic, framework-independent non-maximum suppression."""

from __future__ import annotations

import math
from collections.abc import Sequence


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


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute IoU for two valid, positive-area ``xyxy`` boxes."""
    return _iou_validated(
        _validated_box(box_a, name="box_a"),
        _validated_box(box_b, name="box_b"),
    )


__all__ = ["compute_iou", "nms"]
