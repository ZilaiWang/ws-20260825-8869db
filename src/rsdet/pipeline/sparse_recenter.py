"""Deterministic, budgeted re-centering requests for boundary-risk clusters."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class RecenterCandidate:
    box: Box
    local_box: Box
    score: float
    label: int
    tile_id: int
    tile_width: int
    tile_height: int
    internal_edges: tuple[bool, bool, bool, bool]


@dataclass(frozen=True)
class RecenterRequest:
    box: Box
    label: int
    priority: float
    reasons: tuple[str, ...]
    max_score: float
    member_count: int


@dataclass(frozen=True)
class RecenterWindow:
    x_start: int
    y_start: int
    width: int
    height: int
    query_box: Box
    label: int
    priority: float
    reasons: tuple[str, ...]

    @property
    def box(self) -> Box:
        return (
            float(self.x_start),
            float(self.y_start),
            float(self.x_start + self.width),
            float(self.y_start + self.height),
        )


def area(box: Sequence[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def overlap(first: Sequence[float], second: Sequence[float]) -> tuple[float, float]:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = area(first)
    second_area = area(second)
    union = first_area + second_area - intersection
    smaller = min(first_area, second_area)
    return (
        intersection / union if union > 0.0 else 0.0,
        intersection / smaller if smaller > 0.0 else 0.0,
    )


def union_box(boxes: Sequence[Sequence[float]]) -> Box:
    if not boxes:
        raise ValueError("boxes must be non-empty")
    return (
        min(float(box[0]) for box in boxes),
        min(float(box[1]) for box in boxes),
        max(float(box[2]) for box in boxes),
        max(float(box[3]) for box in boxes),
    )


def internal_context_margin(candidate: RecenterCandidate) -> float:
    """Smallest available context on avoidable (non-image) tile edges in pixels."""
    x1, y1, x2, y2 = candidate.local_box
    left, top, right, bottom = candidate.internal_edges
    values = []
    if left:
        values.append(x1)
    if top:
        values.append(y1)
    if right:
        values.append(float(candidate.tile_width) - x2)
    if bottom:
        values.append(float(candidate.tile_height) - y2)
    return min(values) if values else math.inf


def cluster_same_fine(
    candidates: Iterable[RecenterCandidate],
    *,
    merge_iou: float = 0.50,
    merge_ios: float = 0.75,
) -> tuple[tuple[RecenterCandidate, ...], ...]:
    if not 0.0 < merge_iou <= 1.0 or not 0.0 < merge_ios <= 1.0:
        raise ValueError("merge thresholds must be in (0, 1]")
    ordered = sorted(
        candidates,
        key=lambda row: (-row.score, row.label, row.box, row.tile_id),
    )
    assigned = [False] * len(ordered)
    clusters = []
    for anchor_index, anchor in enumerate(ordered):
        if assigned[anchor_index]:
            continue
        assigned[anchor_index] = True
        members = [anchor]
        for index in range(anchor_index + 1, len(ordered)):
            if assigned[index]:
                continue
            candidate = ordered[index]
            if candidate.label != anchor.label or candidate.tile_id == anchor.tile_id:
                continue
            iou, ios = overlap(anchor.box, candidate.box)
            if iou >= merge_iou or ios >= merge_ios:
                assigned[index] = True
                members.append(candidate)
        clusters.append(tuple(members))
    return tuple(clusters)


def boundary_risk_requests(
    clusters: Iterable[Sequence[RecenterCandidate]],
    *,
    rescue_floor: float,
    output_threshold: float,
    overlap_size: int,
    selected_labels: frozenset[int] = frozenset({0, 1, 2, 3, 24}),
) -> tuple[RecenterRequest, ...]:
    """Rank only deterministic risk signals frozen before E3 evaluation."""
    if not 0.0 <= rescue_floor <= output_threshold <= 1.0:
        raise ValueError("require 0 <= rescue_floor <= output_threshold <= 1")
    if overlap_size <= 0:
        raise ValueError("overlap_size must be positive")
    requests = []
    for cluster in clusters:
        if not cluster:
            continue
        label = int(cluster[0].label)
        if label not in selected_labels or any(row.label != label for row in cluster):
            continue
        maximum = max(row.score for row in cluster)
        if maximum < rescue_floor:
            continue
        merged = union_box([row.box for row in cluster])
        max_side = max(merged[2] - merged[0], merged[3] - merged[1])
        required_context = max(32.0, 0.25 * max_side)
        best_context = max(internal_context_margin(row) for row in cluster)
        score_spread = maximum - min(row.score for row in cluster)
        reasons = []
        if maximum < output_threshold:
            reasons.append("risk_band")
        if best_context < required_context:
            reasons.append("poor_context")
        if max_side > overlap_size:
            reasons.append("larger_than_overlap")
        if len(cluster) > 1 and score_spread >= 0.20:
            reasons.append("cross_tile_score_spread")
        if not reasons:
            continue
        band = max(output_threshold - rescue_floor, 1e-9)
        closeness = 1.0 - min(abs(output_threshold - maximum) / band, 1.0)
        context_severity = (
            0.0
            if math.isinf(best_context)
            else max(0.0, required_context - best_context) / required_context
        )
        priority = (
            2.0 * closeness
            + context_severity
            + (0.5 if max_side > overlap_size else 0.0)
            + (0.25 if score_spread >= 0.20 else 0.0)
        )
        requests.append(
            RecenterRequest(
                box=merged,
                label=label,
                priority=priority,
                reasons=tuple(reasons),
                max_score=maximum,
                member_count=len(cluster),
            )
        )
    return tuple(sorted(requests, key=lambda row: (-row.priority, row.label, row.box)))


def centered_window(
    request: RecenterRequest,
    *,
    image_width: int,
    image_height: int,
    window_size: int = 1024,
) -> RecenterWindow:
    if image_width <= 0 or image_height <= 0 or window_size <= 0:
        raise ValueError("image and window dimensions must be positive")
    width = min(window_size, image_width)
    height = min(window_size, image_height)
    x1, y1, x2, y2 = request.box
    x_start = int(round((x1 + x2 - width) / 2.0))
    y_start = int(round((y1 + y2 - height) / 2.0))
    x_start = min(max(x_start, 0), image_width - width)
    y_start = min(max(y_start, 0), image_height - height)
    return RecenterWindow(
        x_start,
        y_start,
        width,
        height,
        request.box,
        request.label,
        request.priority,
        request.reasons,
    )


def select_windows(
    requests: Iterable[RecenterRequest],
    *,
    image_width: int,
    image_height: int,
    window_size: int = 1024,
    max_windows: int = 8,
    dedup_iou: float = 0.60,
) -> tuple[RecenterWindow, ...]:
    if max_windows <= 0:
        raise ValueError("max_windows must be positive")
    if not 0.0 <= dedup_iou <= 1.0:
        raise ValueError("dedup_iou must be in [0, 1]")
    selected = []
    for request in sorted(requests, key=lambda row: (-row.priority, row.label, row.box)):
        window = centered_window(
            request,
            image_width=image_width,
            image_height=image_height,
            window_size=window_size,
        )
        if any(overlap(window.box, existing.box)[0] >= dedup_iou for existing in selected):
            continue
        selected.append(window)
        if len(selected) >= max_windows:
            break
    return tuple(selected)


__all__ = [
    "RecenterCandidate",
    "RecenterRequest",
    "RecenterWindow",
    "boundary_risk_requests",
    "centered_window",
    "cluster_same_fine",
    "internal_context_margin",
    "overlap",
    "select_windows",
    "union_box",
]
