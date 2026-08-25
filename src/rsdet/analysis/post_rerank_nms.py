"""Leakage-safe analysis for class-aware NMS after object reranking.

Detector NMS is no longer authoritative after a downstream object head changes
fine categories.  This module evaluates a second, deterministic NMS in the
final category space.  IoU thresholds are selected on the other two OOF folds
and applied once to the held-out fold.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from rsdet.analysis.crossfit_thresholds import (
    evaluate_ranking_workpoint,
    evaluate_workpoint,
)
from rsdet.evaluation.protocol import EvaluationProtocol
from rsdet.postprocess.nms import class_aware_nms_predictions

POST_RERANK_NMS_CONTRACT_VERSION = "r1_post_rerank_nms_v1"


def _prediction_count(predictions: Mapping[int, Sequence[Mapping[str, Any]]]) -> int:
    return sum(len(records) for records in predictions.values())


def _evaluate_predictions(
    gt_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    normalized_gt = {
        int(image_id): [dict(record) for record in records]
        for image_id, records in gt_boxes.items()
    }
    pooled = evaluate_workpoint(
        normalized_gt,
        predictions,
        threshold=0.0,
        protocol=protocol,
    )
    ranking = evaluate_ranking_workpoint(
        normalized_gt,
        predictions,
        threshold=0.0,
        protocol=protocol,
        require_complete_taxonomy=True,
    )
    return {
        "prediction_count": _prediction_count(predictions),
        "pooled_recall": pooled.recall,
        "pooled_fdr": pooled.fdr,
        "tp": int(pooled.details["tp"]),
        "fp": int(pooled.details["fp"]),
        "fn": int(pooled.details["fn"]),
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
    }


def _subset(
    records: Mapping[int, Sequence[Mapping[str, Any]]],
    image_folds: Mapping[int, int],
    folds: set[int],
) -> dict[int, list[dict[str, Any]]]:
    return {
        int(image_id): [dict(record) for record in records.get(int(image_id), ())]
        for image_id, fold in image_folds.items()
        if int(fold) in folds
    }


def build_nms_curve(
    gt_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    iou_thresholds: Sequence[float],
    category_ids: Sequence[int] | None,
    protocol: EvaluationProtocol,
) -> list[dict[str, Any]]:
    """Evaluate the fixed NMS grid with official pooled and macro metrics."""

    thresholds = sorted({float(value) for value in iou_thresholds}, reverse=True)
    if not thresholds or thresholds[0] != 1.0:
        raise ValueError("NMS grid must include 1.0 as the no-suppression reference")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("NMS thresholds must be finite and within [0, 1]")

    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        suppressed = class_aware_nms_predictions(
            predictions,
            threshold,
            category_ids=category_ids,
        )
        rows.append(
            {
                "iou_threshold": threshold,
                **_evaluate_predictions(gt_boxes, suppressed, protocol),
            }
        )
    return rows


def select_nms_workpoint(
    curve: Sequence[Mapping[str, Any]],
    *,
    maximum_pooled_recall_drop: float,
) -> dict[str, Any]:
    """Select minimum macro FDR under a bounded pooled-Recall sacrifice."""

    if maximum_pooled_recall_drop < 0.0:
        raise ValueError("maximum_pooled_recall_drop must be nonnegative")
    references = [row for row in curve if float(row["iou_threshold"]) == 1.0]
    if len(references) != 1:
        raise ValueError("curve must contain exactly one no-suppression row")
    baseline_recall = float(references[0]["pooled_recall"])
    eligible = [
        row
        for row in curve
        if baseline_recall - float(row["pooled_recall"])
        <= maximum_pooled_recall_drop + 1e-15
    ]
    if not eligible:
        raise RuntimeError("no NMS candidate satisfies the recall constraint")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["macro_fdr"]),
            -float(row["pooled_recall"]),
            -float(row["iou_threshold"]),
        ),
    )
    return {
        "policy": "min_macro_fdr_subject_to_bounded_pooled_recall_drop",
        "maximum_pooled_recall_drop": maximum_pooled_recall_drop,
        "baseline_pooled_recall": baseline_recall,
        "selected_iou_threshold": float(selected["iou_threshold"]),
        "selected": dict(selected),
        "eligible_count": len(eligible),
    }


def run_crossfit_post_rerank_nms(
    gt_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    image_folds: Mapping[int, int],
    *,
    iou_thresholds: Sequence[float],
    maximum_pooled_recall_drop: float,
    category_ids: Sequence[int] | None,
    protocol: EvaluationProtocol,
) -> tuple[dict[str, Any], dict[int, list[dict[str, Any]]]]:
    """Select NMS IoU on two folds and merge the three held-out applications."""

    normalized_folds = {int(image_id): int(fold) for image_id, fold in image_folds.items()}
    if set(normalized_folds.values()) != {0, 1, 2}:
        raise ValueError("image_folds must cover exactly folds 0, 1, and 2")
    if set(int(image_id) for image_id in gt_boxes) != set(normalized_folds):
        raise ValueError("ground-truth image ledger and image_folds differ")

    merged: dict[int, list[dict[str, Any]]] = {}
    per_fold: list[dict[str, Any]] = []
    for held_out_fold in (0, 1, 2):
        selection_folds = {0, 1, 2} - {held_out_fold}
        selection_gt = _subset(gt_boxes, normalized_folds, selection_folds)
        selection_predictions = _subset(predictions, normalized_folds, selection_folds)
        curve = build_nms_curve(
            selection_gt,
            selection_predictions,
            iou_thresholds=iou_thresholds,
            category_ids=category_ids,
            protocol=protocol,
        )
        selection = select_nms_workpoint(
            curve,
            maximum_pooled_recall_drop=maximum_pooled_recall_drop,
        )
        held_out_predictions = _subset(predictions, normalized_folds, {held_out_fold})
        held_out_gt = _subset(gt_boxes, normalized_folds, {held_out_fold})
        held_out_suppressed = class_aware_nms_predictions(
            held_out_predictions,
            selection["selected_iou_threshold"],
            category_ids=category_ids,
        )
        overlap = set(merged) & set(held_out_suppressed)
        if overlap:
            raise RuntimeError(f"held-out fold image overlap: {sorted(overlap)[:3]}")
        merged.update(held_out_suppressed)
        per_fold.append(
            {
                "held_out_fold": held_out_fold,
                "selection_folds": sorted(selection_folds),
                "selection": selection,
                "selection_curve": curve,
                "held_out_images": len(held_out_suppressed),
                "held_out_predictions_before": _prediction_count(held_out_predictions),
                "held_out_predictions_after": _prediction_count(held_out_suppressed),
                "held_out_before": _evaluate_predictions(
                    held_out_gt,
                    held_out_predictions,
                    protocol,
                ),
                "held_out_after": _evaluate_predictions(
                    held_out_gt,
                    held_out_suppressed,
                    protocol,
                ),
            }
        )
    if set(merged) != set(normalized_folds):
        raise RuntimeError("cross-fit NMS output does not cover the full image ledger")
    return (
        {
            "contract_version": POST_RERANK_NMS_CONTRACT_VERSION,
            "status": "complete",
            "scientific_scope": "iterative_oof_development",
            "per_fold": per_fold,
            "selected_iou_threshold_by_fold": {
                str(item["held_out_fold"]): item["selection"]["selected_iou_threshold"]
                for item in per_fold
            },
        },
        merged,
    )


__all__ = [
    "POST_RERANK_NMS_CONTRACT_VERSION",
    "build_nms_curve",
    "run_crossfit_post_rerank_nms",
    "select_nms_workpoint",
]
