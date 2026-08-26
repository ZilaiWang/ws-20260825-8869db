"""Official proposal labels shared by OER, SCOPE-Audit, and HERA-Guard.

The label builder intentionally delegates all one-to-one assignment to
``evaluate_predictions_with_trace``.  Callers may add geometric/oracle
diagnostics, but ``is_valid`` and ``matched_gt_index`` must come from this
official prediction-first trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from rsdet.evaluation.official_metric import (
    OfficialEvaluationTrace,
    OverallMetrics,
    evaluate_predictions_with_trace,
)

LABEL_CONTRACT_VERSION = "official_prediction_first_v1"


@dataclass(frozen=True)
class OfficialProposalLabel:
    candidate_id: int
    image_id: int
    is_valid: bool
    matched_gt_index: int | None
    matched_iou: float


@dataclass(frozen=True)
class OfficialProposalLabelResult:
    labels: dict[int, OfficialProposalLabel]
    metrics: OverallMetrics
    trace: OfficialEvaluationTrace
    contract_version: str = LABEL_CONTRACT_VERSION


def build_official_proposal_labels(
    *,
    gt_boxes: dict[int, list[dict[str, Any]]],
    predictions: Iterable[Mapping[str, Any]],
    category_mapping: dict[int, str],
    iou_thresholds: dict[str, float],
) -> OfficialProposalLabelResult:
    """Return one label for every globally unique candidate ID."""

    pred_by_image: dict[int, list[dict[str, Any]]] = {
        int(image_id): [] for image_id in gt_boxes
    }
    candidate_image: dict[int, int] = {}
    for order, raw in enumerate(predictions):
        candidate_id = int(raw.get("source_prediction_index", raw.get("candidate_id", order)))
        if candidate_id in candidate_image:
            raise ValueError(f"candidate ID must be globally unique: {candidate_id}")
        image_id = int(raw["image_id"])
        candidate_image[candidate_id] = image_id
        pred_by_image.setdefault(image_id, []).append(
            {
                "category_id": int(raw["category_id"]),
                "score": float(raw["score"]),
                "bbox_xyxy": [float(value) for value in raw["bbox_xyxy"]],
                "source_prediction_index": candidate_id,
            }
        )

    # Freeze stable tie order independently of caller iteration order.
    for records in pred_by_image.values():
        records.sort(key=lambda row: (-row["score"], row["source_prediction_index"]))

    metrics, trace = evaluate_predictions_with_trace(
        gt_boxes,
        pred_by_image,
        class_names=list(dict.fromkeys(category_mapping.values())),
        category_mapping=category_mapping,
        iou_thresholds=iou_thresholds,
    )
    matched = {int(item.prediction_index): item for item in trace.matches}
    labels: dict[int, OfficialProposalLabel] = {}
    for candidate_id, image_id in candidate_image.items():
        item = matched.get(candidate_id)
        labels[candidate_id] = OfficialProposalLabel(
            candidate_id=candidate_id,
            image_id=image_id,
            is_valid=item is not None,
            matched_gt_index=None if item is None else int(item.ground_truth_index),
            matched_iou=0.0 if item is None else float(item.iou),
        )

    if sum(label.is_valid for label in labels.values()) != int(metrics.details["tp"]):
        raise AssertionError("proposal labels and official TP count diverged")
    return OfficialProposalLabelResult(labels=labels, metrics=metrics, trace=trace)


__all__ = [
    "LABEL_CONTRACT_VERSION",
    "OfficialProposalLabel",
    "OfficialProposalLabelResult",
    "build_official_proposal_labels",
]
