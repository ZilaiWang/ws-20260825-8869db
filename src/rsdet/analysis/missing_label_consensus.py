"""Build conservative cross-detector candidates for missing-label review.

This module never edits annotations and never turns an unreviewed proposal into
an ignored region.  It only ranks primary-detector boxes that are geometrically
supported by a second detector and do not overlap any existing annotation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from rsdet.evaluation.official_metric import compute_iou


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    """Convert a finite positive COCO box to ``xyxy`` coordinates."""

    values = [float(value) for value in box]
    if (
        len(values) != 4
        or not all(math.isfinite(value) for value in values)
        or values[2] <= 0.0
        or values[3] <= 0.0
    ):
        raise ValueError(f"invalid COCO bbox={values}")
    return [values[0], values[1], values[0] + values[2], values[1] + values[3]]


def build_missing_label_candidates(
    primary: Sequence[Mapping[str, Any]],
    specialist: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    category_id: int,
    support_iou_min: float,
    existing_gt_iou_max: float,
    product_min: float,
    primary_score_min: float = 0.0,
    specialist_score_min: float = 0.0,
    dedup_iou: float = 0.50,
) -> list[dict[str, Any]]:
    """Return deterministic review candidates for one fine category.

    Specialist detections are evidence only.  Every returned geometry is an
    original primary-detector box.  Existing GT overlap is checked against all
    categories, not only ``category_id``.
    """

    for name, value in {
        "support_iou_min": support_iou_min,
        "existing_gt_iou_max": existing_gt_iou_max,
        "product_min": product_min,
        "primary_score_min": primary_score_min,
        "specialist_score_min": specialist_score_min,
        "dedup_iou": dedup_iou,
    }.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1]")

    specialist_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for order, raw in enumerate(specialist):
        if int(raw["category_id"]) != category_id:
            continue
        score = _validated_score(raw)
        if score < specialist_score_min:
            continue
        specialist_index[int(raw["image_id"])].append(
            {
                "order": order,
                "score": score,
                "bbox_xyxy": xywh_to_xyxy(raw["bbox"]),
            }
        )

    gt_index: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in annotations:
        gt_index[int(raw["image_id"])].append(
            {
                "category_id": int(raw["category_id"]),
                "bbox_xyxy": xywh_to_xyxy(raw["bbox"]),
            }
        )

    candidates: list[dict[str, Any]] = []
    for primary_order, raw in enumerate(primary):
        if int(raw["category_id"]) != category_id:
            continue
        primary_score = _validated_score(raw)
        if primary_score < primary_score_min:
            continue
        image_id = int(raw["image_id"])
        primary_box = xywh_to_xyxy(raw["bbox"])

        best_support: tuple[float, float, int, list[float]] | None = None
        for support in specialist_index.get(image_id, ()):
            overlap = compute_iou(primary_box, support["bbox_xyxy"])
            if overlap < support_iou_min:
                continue
            rank = (float(support["score"]), overlap, -int(support["order"]))
            if best_support is None or rank > best_support[:3]:
                best_support = (*rank, list(support["bbox_xyxy"]))
        if best_support is None:
            continue
        support_score, support_iou, neg_support_order, support_box = best_support
        product = primary_score * support_score
        if product < product_min:
            continue

        maximum_gt_iou = 0.0
        nearest_gt_category: int | None = None
        for gt in gt_index.get(image_id, ()):
            overlap = compute_iou(primary_box, gt["bbox_xyxy"])
            if overlap > maximum_gt_iou:
                maximum_gt_iou = overlap
                nearest_gt_category = int(gt["category_id"])
        if maximum_gt_iou >= existing_gt_iou_max:
            continue

        candidates.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox_xyxy": primary_box,
                "primary_score": primary_score,
                "support_score": support_score,
                "support_iou": support_iou,
                "agreement_product": product,
                "support_bbox_xyxy": support_box,
                "maximum_gt_iou": maximum_gt_iou,
                "nearest_gt_category_id": nearest_gt_category,
                "primary_order": primary_order,
                "specialist_order": -neg_support_order,
            }
        )

    candidates.sort(
        key=lambda row: (
            -float(row["agreement_product"]),
            -float(row["support_iou"]),
            -float(row["primary_score"]),
            int(row["image_id"]),
            int(row["primary_order"]),
        )
    )
    return _greedy_deduplicate(candidates, iou_threshold=dedup_iou)


def _greedy_deduplicate(
    candidates: Sequence[Mapping[str, Any]], *, iou_threshold: float
) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    by_image: dict[int, list[list[float]]] = defaultdict(list)
    for raw in candidates:
        row = dict(raw)
        image_id = int(row["image_id"])
        box = [float(value) for value in row["bbox_xyxy"]]
        if any(compute_iou(box, other) >= iou_threshold for other in by_image[image_id]):
            continue
        by_image[image_id].append(box)
        kept.append(row)
    return kept


def _validated_score(raw: Mapping[str, Any]) -> float:
    score = float(raw["score"])
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"invalid score={score}")
    return score


__all__ = ["build_missing_label_candidates", "xywh_to_xyxy"]
