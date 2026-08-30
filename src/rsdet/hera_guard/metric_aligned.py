"""Metric-aligned proposal roles for HERA-Guard V3.

The official evaluator is the sole authority for the canonical TP assigned to
each object.  Geometry-only associations are retained separately so that a
duplicate, a wrong-fine proposal and a localization/background proposal never
share the same training target by accident.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from rsdet.analysis.oer_labels import build_official_proposal_labels
from rsdet.evaluation.official_metric import compute_iou

CANONICAL = "canonical_positive"
DUPLICATE = "duplicate_negative"
CROSS_FINE = "cross_fine_negative"
CROSS_COARSE = "cross_coarse_negative"
LOCALIZATION = "localization_negative"
BACKGROUND = "background_negative"


@dataclass(frozen=True)
class MetricAlignedRole:
    candidate_id: int
    image_id: int
    predicted_category_id: int
    predicted_coarse: str
    role: str
    target: int
    object_group_id: str
    support_gt_index: int | None
    support_category_id: int | None
    support_coarse: str | None
    support_iou: float
    official_match_iou: float


@dataclass(frozen=True)
class MetricAlignedRoleResult:
    roles: tuple[MetricAlignedRole, ...]
    role_counts: dict[str, int]
    object_group_count: int
    max_group_size: int
    official_tp: int
    official_fp: int
    cross_coarse_foreground_pollution: int


def normalize_prediction(raw: Mapping[str, Any], *, candidate_id: int) -> dict[str, Any]:
    """Normalize a COCO or internal prediction without changing its score."""

    if "bbox_xyxy" in raw:
        box = [float(value) for value in raw["bbox_xyxy"]]
    else:
        x, y, width, height = (float(value) for value in raw["bbox"])
        box = [x, y, x + width, y + height]
    if (
        len(box) != 4
        or not all(math.isfinite(value) for value in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"invalid prediction box: {box}")
    score = float(raw["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"invalid prediction score: {score}")
    return {
        **dict(raw),
        "candidate_id": int(candidate_id),
        "source_prediction_index": int(candidate_id),
        "image_id": int(raw["image_id"]),
        "category_id": int(raw["category_id"]),
        "score": score,
        "bbox_xyxy": box,
    }


def _best_supports(
    *,
    image_id: int,
    box: Sequence[float],
    predicted_category_id: int,
    predicted_coarse: str,
    gt_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    category_mapping: Mapping[int, str],
) -> dict[str, tuple[int, int, str, float] | None]:
    """Find same-fine, same-coarse and unconstrained support independently.

    A closer wrong-class object must not hide a valid same-fine association.
    Deterministic ties always select the first GT row.
    """

    best: dict[str, tuple[float, int, int, str] | None] = {
        "same_fine": None,
        "same_coarse": None,
        "any": None,
    }
    for gt_index, gt in enumerate(gt_boxes.get(image_id, ())):
        category_id = int(gt["category_id"])
        coarse = str(category_mapping[category_id])
        iou = compute_iou(list(box), [float(value) for value in gt["bbox_xyxy"]])
        candidate = (iou, -gt_index, category_id, coarse)
        keys = ["any"]
        if coarse == predicted_coarse:
            keys.append("same_coarse")
        if category_id == predicted_category_id:
            keys.append("same_fine")
        for key in keys:
            if best[key] is None or candidate[:2] > best[key][:2]:
                best[key] = candidate

    output: dict[str, tuple[int, int, str, float] | None] = {}
    for key, value in best.items():
        if value is None:
            output[key] = None
        else:
            iou, negative_index, category_id, coarse = value
            output[key] = (-negative_index, category_id, coarse, iou)
    return output


def build_metric_aligned_roles(
    *,
    gt_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions: Iterable[Mapping[str, Any]],
    category_mapping: Mapping[int, str],
    iou_thresholds: Mapping[str, float],
    background_iou: float = 0.05,
) -> MetricAlignedRoleResult:
    """Assign one canonical positive and explicit negative roles.

    ``canonical_positive`` is copied from the exact prediction-first official
    trace.  Geometry-only support then explains why every remaining proposal is
    a duplicate, wrong-fine/coarse, localization, or clear-background sample.
    """

    if not 0.0 <= background_iou < min(float(value) for value in iou_thresholds.values()):
        raise ValueError("background_iou must lie below every official IoU threshold")
    normalized = [
        normalize_prediction(raw, candidate_id=index)
        for index, raw in enumerate(predictions)
    ]
    official = build_official_proposal_labels(
        gt_boxes={int(key): list(value) for key, value in gt_boxes.items()},
        predictions=normalized,
        category_mapping={int(key): str(value) for key, value in category_mapping.items()},
        iou_thresholds={str(key): float(value) for key, value in iou_thresholds.items()},
    )
    rows: list[MetricAlignedRole] = []
    group_sizes: Counter[str] = Counter()
    cross_coarse_pollution = 0
    for item in normalized:
        candidate_id = int(item["candidate_id"])
        image_id = int(item["image_id"])
        predicted_category = int(item["category_id"])
        predicted_coarse = str(category_mapping[predicted_category])
        official_label = official.labels[candidate_id]
        supports = _best_supports(
            image_id=image_id,
            box=item["bbox_xyxy"],
            predicted_category_id=predicted_category,
            predicted_coarse=predicted_coarse,
            gt_boxes=gt_boxes,
            category_mapping=category_mapping,
        )
        support_index: int | None = None
        support_category: int | None = None
        support_coarse: str | None = None
        support_iou = 0.0
        qualifies = False
        object_group_id = ""
        if official_label.is_valid:
            role = CANONICAL
            support_index = int(official_label.matched_gt_index)
            matched_gt = gt_boxes[image_id][support_index]
            support_category = int(matched_gt["category_id"])
            support_coarse = str(category_mapping[support_category])
            support_iou = float(official_label.matched_iou)
            qualifies = True
        else:
            same_fine = supports["same_fine"]
            same_coarse = supports["same_coarse"]
            any_support = supports["any"]
            if (
                same_fine is not None
                and same_fine[3] >= float(iou_thresholds[predicted_coarse])
            ):
                support_index, support_category, support_coarse, support_iou = same_fine
                qualifies = True
                role = DUPLICATE
            elif (
                same_coarse is not None
                and same_coarse[3] >= float(iou_thresholds[predicted_coarse])
            ):
                support_index, support_category, support_coarse, support_iou = same_coarse
                qualifies = True
                role = CROSS_FINE
            elif (
                any_support is not None
                and any_support[3] >= float(iou_thresholds[any_support[2]])
            ):
                support_index, support_category, support_coarse, support_iou = any_support
                qualifies = True
                role = CROSS_COARSE
                cross_coarse_pollution += 1
            else:
                if any_support is not None:
                    support_index, support_category, support_coarse, support_iou = any_support
                role = LOCALIZATION if support_iou > background_iou else BACKGROUND
        if qualifies:
            object_group_id = f"image{image_id}:gt{support_index}"
            group_sizes[object_group_id] += 1
        rows.append(
            MetricAlignedRole(
                candidate_id=candidate_id,
                image_id=image_id,
                predicted_category_id=predicted_category,
                predicted_coarse=predicted_coarse,
                role=role,
                target=int(role == CANONICAL),
                object_group_id=object_group_id,
                support_gt_index=support_index,
                support_category_id=support_category,
                support_coarse=support_coarse,
                support_iou=float(support_iou),
                official_match_iou=float(official_label.matched_iou),
            )
        )
    counts = Counter(row.role for row in rows)
    return MetricAlignedRoleResult(
        roles=tuple(rows),
        role_counts=dict(sorted(counts.items())),
        object_group_count=len(group_sizes),
        max_group_size=max(group_sizes.values(), default=0),
        official_tp=int(official.metrics.details["tp"]),
        official_fp=int(official.metrics.details["fp"]),
        cross_coarse_foreground_pollution=cross_coarse_pollution,
    )


def group_candidate_ids(roles: Iterable[MetricAlignedRole]) -> dict[str, list[int]]:
    """Return deterministic non-empty object groups."""

    groups: dict[str, list[int]] = defaultdict(list)
    for row in roles:
        if row.object_group_id:
            groups[row.object_group_id].append(row.candidate_id)
    return {key: sorted(value) for key, value in sorted(groups.items())}


__all__ = [
    "BACKGROUND",
    "CANONICAL",
    "CROSS_COARSE",
    "CROSS_FINE",
    "DUPLICATE",
    "LOCALIZATION",
    "MetricAlignedRole",
    "MetricAlignedRoleResult",
    "build_metric_aligned_roles",
    "group_candidate_ids",
    "normalize_prediction",
]
