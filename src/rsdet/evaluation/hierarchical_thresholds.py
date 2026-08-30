"""Leakage-safe hierarchical fine-class score calibration.

The official ranking averages metrics over the 25 fine classes, while the
hard admission gate is pooled over all classes.  This module fits one global
anchor, three coarse anchors and 25 fine-class thresholds on training folds
only.  Fine thresholds are shrunk toward their coarse anchor according to the
amount of ground-truth evidence, preventing tiny tail classes from receiving
unstable, extreme thresholds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, TypeVar

from rsdet.analysis.oof_detection import build_threshold_curve
from rsdet.evaluation.protocol import EvaluationProtocol

T = TypeVar("T")


@dataclass(frozen=True)
class ThresholdCurves:
    """Reusable score curves fitted on one training partition."""

    global_curve: list[dict[str, Any]]
    coarse_curves: dict[str, list[dict[str, Any]]]
    fine_curves: dict[int, list[dict[str, Any]]]
    fine_gt_counts: dict[int, int]


def select_threshold(points: list[dict[str, Any]], target_fdr: float) -> dict[str, Any]:
    """Select maximum-recall point under a fixed FDR, conservatively on ties."""

    if not points:
        raise ValueError("threshold curve cannot be empty")
    if not 0.0 <= target_fdr <= 1.0:
        raise ValueError("target_fdr must be in [0, 1]")
    feasible = [point for point in points if float(point["overall_fdr"]) <= target_fdr]
    if feasible:
        return max(
            feasible,
            key=lambda point: (
                float(point["overall_recall"]),
                -float(point["overall_fdr"]),
                float(point["threshold"]),
            ),
        )
    return min(
        points,
        key=lambda point: (
            float(point["overall_fdr"]),
            -float(point["threshold"]),
        ),
    )


def shrink_threshold(
    raw_threshold: float,
    anchor_threshold: float,
    *,
    evidence: int,
    prior_strength: float,
    minimum_evidence: int,
) -> tuple[float, float]:
    """Shrink in logit space and return ``(threshold, fine_weight)``."""

    if evidence < 0 or prior_strength < 0 or minimum_evidence < 0:
        raise ValueError("evidence and prior controls must be non-negative")
    if not 0.0 < raw_threshold < 1.0 or not 0.0 < anchor_threshold < 1.0:
        raise ValueError("thresholds must be strictly inside (0, 1)")
    weight = (
        0.0
        if evidence < minimum_evidence
        else evidence / (evidence + prior_strength)
        if evidence + prior_strength > 0
        else 1.0
    )

    def logit(value: float) -> float:
        return math.log(value / (1.0 - value))

    mixed = weight * logit(raw_threshold) + (1.0 - weight) * logit(anchor_threshold)
    return 1.0 / (1.0 + math.exp(-mixed)), weight


def _scoped(mapping: dict[int, list[T]], image_ids: set[int]) -> dict[int, list[T]]:
    return {image_id: list(mapping.get(image_id, [])) for image_id in sorted(image_ids)}


def _category_subset(
    mapping: dict[int, list[dict[str, Any]]], category_ids: set[int]
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [item for item in items if int(item["category_id"]) in category_ids]
        for image_id, items in mapping.items()
    }


def _subprotocol(
    protocol: EvaluationProtocol,
    *,
    class_name: str,
    category_ids: set[int],
) -> EvaluationProtocol:
    return EvaluationProtocol(
        contract_version=protocol.contract_version,
        eval_version=protocol.eval_version,
        ranking_version=protocol.ranking_version,
        class_names=[class_name],
        category_mapping={category_id: class_name for category_id in sorted(category_ids)},
        iou_thresholds={class_name: protocol.iou_thresholds[class_name]},
        recall_min=protocol.recall_min,
        fdr_max=protocol.fdr_max,
        latency_max_seconds=protocol.latency_max_seconds,
    )


def build_hierarchical_curves(
    gt: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    thresholds: list[float],
    protocol: EvaluationProtocol,
) -> ThresholdCurves:
    """Build global/coarse/fine curves once for one training partition."""

    train_gt = _scoped(gt, image_ids)
    train_pred = _scoped(predictions, image_ids)
    global_curve, _ = build_threshold_curve(
        train_gt, train_pred, thresholds=thresholds, protocol=protocol
    )
    coarse_curves: dict[str, list[dict[str, Any]]] = {}
    fine_curves: dict[int, list[dict[str, Any]]] = {}
    fine_gt_counts: dict[int, int] = {}
    for coarse_name in protocol.class_names:
        category_ids = {
            category_id
            for category_id, mapped_name in protocol.category_mapping.items()
            if mapped_name == coarse_name
        }
        coarse_protocol = _subprotocol(
            protocol, class_name=coarse_name, category_ids=category_ids
        )
        coarse_curves[coarse_name], _ = build_threshold_curve(
            _category_subset(train_gt, category_ids),
            _category_subset(train_pred, category_ids),
            thresholds=thresholds,
            protocol=coarse_protocol,
        )
        for category_id in sorted(category_ids):
            fine_protocol = _subprotocol(
                protocol, class_name=coarse_name, category_ids={category_id}
            )
            fine_gt = _category_subset(train_gt, {category_id})
            fine_pred = _category_subset(train_pred, {category_id})
            fine_gt_counts[category_id] = sum(len(items) for items in fine_gt.values())
            fine_curves[category_id], _ = build_threshold_curve(
                fine_gt,
                fine_pred,
                thresholds=thresholds,
                protocol=fine_protocol,
            )
    return ThresholdCurves(
        global_curve=global_curve,
        coarse_curves=coarse_curves,
        fine_curves=fine_curves,
        fine_gt_counts=fine_gt_counts,
    )


def fit_hierarchical_thresholds(
    curves: ThresholdCurves,
    *,
    protocol: EvaluationProtocol,
    target_fdr: float,
    prior_strength: float = 50.0,
    minimum_evidence: int = 10,
) -> dict[str, Any]:
    """Fit hierarchical thresholds from precomputed training-only curves."""

    global_point = select_threshold(curves.global_curve, target_fdr)
    coarse_points = {
        name: select_threshold(curve, target_fdr)
        for name, curve in curves.coarse_curves.items()
    }
    fine_thresholds: dict[int, float] = {}
    fine_audit: dict[int, dict[str, Any]] = {}
    for category_id, curve in curves.fine_curves.items():
        coarse_name = protocol.category_mapping[category_id]
        raw_point = select_threshold(curve, target_fdr)
        raw = float(raw_point["threshold"])
        anchor = float(coarse_points[coarse_name]["threshold"])
        threshold, weight = shrink_threshold(
            raw,
            anchor,
            evidence=curves.fine_gt_counts[category_id],
            prior_strength=prior_strength,
            minimum_evidence=minimum_evidence,
        )
        fine_thresholds[category_id] = threshold
        fine_audit[category_id] = {
            "coarse_class": coarse_name,
            "gt_evidence": curves.fine_gt_counts[category_id],
            "raw_threshold": raw,
            "coarse_anchor": anchor,
            "fine_weight": weight,
            "selected_threshold": threshold,
            "raw_train_recall": float(raw_point["overall_recall"]),
            "raw_train_fdr": float(raw_point["overall_fdr"]),
        }
    return {
        "global_threshold": float(global_point["threshold"]),
        "coarse_thresholds": {
            name: float(point["threshold"]) for name, point in coarse_points.items()
        },
        "fine_thresholds": fine_thresholds,
        "fine_audit": fine_audit,
    }


def filter_by_thresholds(
    predictions: dict[int, list[dict[str, Any]]],
    thresholds: dict[int, float],
) -> dict[int, list[dict[str, Any]]]:
    """Apply category-specific thresholds while preserving image coverage."""

    missing = {
        int(item["category_id"])
        for items in predictions.values()
        for item in items
        if int(item["category_id"]) not in thresholds
    }
    if missing:
        raise ValueError(f"missing thresholds for category ids: {sorted(missing)}")
    return {
        image_id: [
            item
            for item in items
            if float(item["score"]) >= thresholds[int(item["category_id"])]
        ]
        for image_id, items in predictions.items()
    }
