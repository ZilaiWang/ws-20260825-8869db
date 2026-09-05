"""Conservative cross-fine duplicate suppression for Ship predictions only."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from rsdet.pipeline.sparse_recenter import overlap


def _xyxy(row: Mapping[str, Any]) -> list[float]:
    box = [float(value) for value in row["bbox"]]
    if len(box) != 4:
        raise ValueError("bbox must contain x, y, width, height")
    x, y, width, height = box
    if width <= 0.0 or height <= 0.0:
        raise ValueError("bbox must have positive area")
    return [x, y, x + width, y + height]


def normalized_center_distance(first: list[float], second: list[float]) -> float:
    first_center = ((first[0] + first[2]) / 2.0, (first[1] + first[3]) / 2.0)
    second_center = ((second[0] + second[2]) / 2.0, (second[1] + second[3]) / 2.0)
    distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    smaller_side = min(
        max(first[2] - first[0], first[3] - first[1]),
        max(second[2] - second[0], second[3] - second[1]),
    )
    return distance / smaller_side


def suppress_strict_ship_cross_fine(
    rows: Iterable[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.75,
    ios_threshold: float = 0.90,
    center_distance_threshold: float = 0.20,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Keep the highest-score real proposal in only near-certain Ship duplicates."""
    for value, name in (
        (iou_threshold, "iou_threshold"),
        (ios_threshold, "ios_threshold"),
        (center_distance_threshold, "center_distance_threshold"),
    ):
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    ordered = sorted(
        (dict(row) for row in rows),
        key=lambda row: (-float(row["score"]), int(row["category_id"]), row["bbox"]),
    )
    kept: list[dict[str, Any]] = []
    suppressed = 0
    compared = 0
    for row in ordered:
        label = int(row["category_id"])
        if label not in {0, 1, 2, 3}:
            kept.append(row)
            continue
        box = _xyxy(row)
        duplicate = False
        for existing in kept:
            existing_label = int(existing["category_id"])
            if existing_label not in {0, 1, 2, 3} or existing_label == label:
                continue
            compared += 1
            existing_box = _xyxy(existing)
            iou, ios = overlap(box, existing_box)
            if (iou >= iou_threshold or ios >= ios_threshold) and normalized_center_distance(
                box, existing_box
            ) <= center_distance_threshold:
                duplicate = True
                break
        if duplicate:
            suppressed += 1
        else:
            kept.append(row)
    return kept, {
        "input_count": len(ordered),
        "output_count": len(kept),
        "suppressed_count": suppressed,
        "cross_fine_comparisons": compared,
    }


__all__ = ["normalized_center_distance", "suppress_strict_ship_cross_fine"]
