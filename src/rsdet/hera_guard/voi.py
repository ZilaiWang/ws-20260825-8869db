"""Deterministic value-of-information routing and sparse recenter windows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class RouteDecision:
    candidate_index: int
    priority: float
    reasons: tuple[str, ...]


def _binary_entropy(probability: float) -> float:
    value = min(max(float(probability), 1e-8), 1.0 - 1e-8)
    return -(value * math.log(value) + (1.0 - value) * math.log(1.0 - value)) / math.log(2.0)


def voi_priority(
    row: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    decision_threshold: float,
) -> tuple[float, tuple[str, ...]]:
    """Rank uncertainty, disagreement, smallness and boundary truncation."""

    score = float(row["score"])
    x, y, width, height = (float(value) for value in row["bbox"])
    if width <= 0 or height <= 0 or image_width <= 0 or image_height <= 0:
        raise ValueError("invalid proposal/image geometry")
    threshold_distance = abs(score - decision_threshold)
    near_threshold = math.exp(-threshold_distance / 0.08)
    foreground = float(row.get("foreground_probability", score))
    disagreement = abs(score - foreground)
    smallness = min(1.0, 32.0 / max(1.0, min(width, height)))
    edge_distance = min(x, y, image_width - (x + width), image_height - (y + height))
    boundary = max(0.0, 1.0 - edge_distance / max(8.0, 0.25 * min(width, height)))
    entropy = _binary_entropy(score)
    priority = (
        0.30 * near_threshold
        + 0.20 * entropy
        + 0.20 * min(1.0, disagreement * 2.0)
        + 0.15 * smallness
        + 0.15 * boundary
    )
    reasons = []
    if near_threshold >= 0.5:
        reasons.append("near_threshold")
    if disagreement >= 0.20:
        reasons.append("evidence_disagreement")
    if smallness >= 0.75:
        reasons.append("tiny_object")
    if boundary >= 0.5:
        reasons.append("boundary_object")
    return priority, tuple(reasons)


def select_voi_budget(
    records: Sequence[Mapping[str, Any]],
    image_sizes: Mapping[int, tuple[int, int]],
    *,
    budget: int,
    decision_threshold: float,
) -> list[RouteDecision]:
    if budget < 0:
        raise ValueError("budget must be non-negative")
    rows = []
    for index, record in enumerate(records):
        width, height = image_sizes[int(record["image_id"])]
        priority, reasons = voi_priority(
            record,
            image_width=width,
            image_height=height,
            decision_threshold=decision_threshold,
        )
        rows.append(RouteDecision(index, priority, reasons))
    rows.sort(key=lambda item: (-item.priority, item.candidate_index))
    return rows[: min(budget, len(rows))]


def recenter_windows(
    bbox_xywh: Sequence[float],
    *,
    context_ratio: float = 1.75,
    shift_fraction: float = 0.18,
) -> list[tuple[float, float, float, float]]:
    """Return center plus four deterministic sparse translations in xyxy."""

    x, y, width, height = (float(value) for value in bbox_xywh)
    if width <= 0 or height <= 0 or context_ratio <= 0 or shift_fraction < 0:
        raise ValueError("invalid recenter geometry")
    side = max(width, height) * context_ratio
    cx, cy = x + width / 2.0, y + height / 2.0
    shifts = ((0.0, 0.0), (-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0))
    result = []
    for dx, dy in shifts:
        center_x = cx + dx * shift_fraction * side
        center_y = cy + dy * shift_fraction * side
        result.append(
            (
                center_x - side / 2.0,
                center_y - side / 2.0,
                center_x + side / 2.0,
                center_y + side / 2.0,
            )
        )
    return result


__all__ = [
    "RouteDecision",
    "recenter_windows",
    "select_voi_budget",
    "voi_priority",
]
