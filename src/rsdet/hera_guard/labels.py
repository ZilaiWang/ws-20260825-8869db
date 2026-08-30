"""Proposal-domain labels that are independent of the detector's fine class.

Official TP labels deliberately require the predicted fine category to match.
PAV objectness/fine supervision has a different purpose: determine whether the
proposal geometry covers any real object and, if so, which object it covers.
This module keeps those two contracts explicit instead of overloading one bit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rsdet.evaluation.official_metric import compute_iou


@dataclass(frozen=True)
class ProposalObjectLabel:
    candidate_id: int
    is_object: bool
    matched_gt_index: int | None
    matched_category_id: int | None
    matched_coarse: str | None
    matched_iou: float


def build_proposal_object_labels(
    *,
    gt_boxes: Mapping[int, list[Mapping[str, Any]]],
    predictions: Iterable[Mapping[str, Any]],
    category_mapping: Mapping[int, str],
    iou_thresholds: Mapping[str, float],
) -> dict[int, ProposalObjectLabel]:
    """Label each proposal by its best qualifying GT, ignoring predicted class.

    Matching is intentionally not one-to-one: duplicated proposals both cover
    the object visually and are separated later by official TP/active-FP risk
    labels.  Ties are resolved by the original GT index for reproducibility.
    """

    labels: dict[int, ProposalObjectLabel] = {}
    for order, raw in enumerate(predictions):
        candidate_id = int(raw.get("source_prediction_index", order))
        if candidate_id in labels:
            raise ValueError(f"duplicate candidate ID: {candidate_id}")
        image_id = int(raw["image_id"])
        box = [float(value) for value in raw["bbox_xyxy"]]
        best: tuple[float, int, int, str] | None = None
        for gt_index, gt in enumerate(gt_boxes.get(image_id, [])):
            category_id = int(gt["category_id"])
            if category_id not in category_mapping:
                raise ValueError(f"GT category lacks coarse mapping: {category_id}")
            coarse = str(category_mapping[category_id])
            if coarse not in iou_thresholds:
                raise ValueError(f"coarse category lacks IoU threshold: {coarse}")
            iou = compute_iou(box, [float(value) for value in gt["bbox_xyxy"]])
            if not math.isfinite(iou) or iou < float(iou_thresholds[coarse]):
                continue
            candidate = (iou, -gt_index, category_id, coarse)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
        if best is None:
            labels[candidate_id] = ProposalObjectLabel(
                candidate_id=candidate_id,
                is_object=False,
                matched_gt_index=None,
                matched_category_id=None,
                matched_coarse=None,
                matched_iou=0.0,
            )
            continue
        iou, negative_index, category_id, coarse = best
        labels[candidate_id] = ProposalObjectLabel(
            candidate_id=candidate_id,
            is_object=True,
            matched_gt_index=-negative_index,
            matched_category_id=category_id,
            matched_coarse=coarse,
            matched_iou=iou,
        )
    return labels


__all__ = ["ProposalObjectLabel", "build_proposal_object_labels"]
