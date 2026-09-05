"""Asymmetric Pareto rescue/reject with immutable primary proposals.

The primary detector owns every emitted box, score and fine label.  Auxiliary
detectors can only provide boolean support for a declared primary proposal;
auxiliary-only detections are never emitted.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from rsdet.contracts import Prediction
from rsdet.submission.agreement import _iou


@dataclass(frozen=True)
class AprrConfig:
    primary_threshold: float
    ship_support_threshold: float
    vehicle_support_threshold: float = 0.546
    vehicle_protect_threshold: float = 0.60
    ship_candidate_floor: float = 0.001
    ship_support_iou: float = 0.50
    vehicle_support_iou: float = 0.35
    ship_rescue_labels: frozenset[int] = frozenset({0, 1, 2})
    vehicle_label: int = 24

    def __post_init__(self) -> None:
        probabilities = (
            self.primary_threshold,
            self.ship_support_threshold,
            self.vehicle_support_threshold,
            self.vehicle_protect_threshold,
            self.ship_candidate_floor,
            self.ship_support_iou,
            self.vehicle_support_iou,
        )
        if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in probabilities):
            raise ValueError("APRR thresholds must be finite and within [0, 1]")
        if self.ship_candidate_floor > self.primary_threshold:
            raise ValueError("ship_candidate_floor cannot exceed primary_threshold")
        if self.vehicle_protect_threshold < self.primary_threshold:
            raise ValueError("vehicle_protect_threshold cannot be below primary_threshold")
        if not self.ship_rescue_labels or not self.ship_rescue_labels <= frozenset(range(4)):
            raise ValueError("ship_rescue_labels must be a non-empty subset of 0..3")
        if self.vehicle_label != 24:
            raise ValueError("the frozen competition Vehicle label is 24")


def _validate(prediction: Prediction, name: str) -> None:
    if not (
        len(prediction.boxes_xyxy) == len(prediction.scores) == len(prediction.labels)
    ):
        raise ValueError(f"{name} arrays must have equal lengths")
    for box, score, label in zip(
        prediction.boxes_xyxy,
        prediction.scores,
        prediction.labels,
        strict=True,
    ):
        if len(box) != 4 or not all(math.isfinite(float(value)) for value in box):
            raise ValueError(f"{name} boxes must contain four finite values")
        if float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]):
            raise ValueError(f"{name} boxes must have positive area")
        if not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
            raise ValueError(f"{name} scores must be finite and within [0, 1]")
        if isinstance(label, bool) or int(label) != label or not 0 <= int(label) < 25:
            raise ValueError(f"{name} labels must be integer category ids 0..24")


def _has_support(
    box: Sequence[float],
    label: int,
    auxiliary: Prediction,
    *,
    minimum_score: float,
    minimum_iou: float,
) -> bool:
    return any(
        int(auxiliary_label) == label
        and float(auxiliary_score) >= minimum_score
        and _iou(box, auxiliary_box) >= minimum_iou
        for auxiliary_box, auxiliary_score, auxiliary_label in zip(
            auxiliary.boxes_xyxy,
            auxiliary.scores,
            auxiliary.labels,
            strict=True,
        )
    )


def apply_aprr(
    primary: Prediction,
    ship_support: Prediction,
    vehicle_support: Prediction,
    *,
    config: AprrConfig,
) -> tuple[Prediction, dict[str, int]]:
    """Apply the frozen APRR proposal policy to one image.

    Rows are returned in primary order.  This guarantees that every output is
    an exact member of the primary ledger and that no expert score can alter
    prediction-first ranking.
    """

    _validate(primary, "primary")
    _validate(ship_support, "ship_support")
    _validate(vehicle_support, "vehicle_support")
    if not (
        primary.image_id == ship_support.image_id == vehicle_support.image_id
    ):
        raise ValueError("APRR prediction image ids must match")

    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    kept_labels: list[int] = []
    stats = {
        "primary_candidates": len(primary.scores),
        "core_kept": 0,
        "ship_rescued": 0,
        "vehicle_risk_supported": 0,
        "vehicle_risk_rejected": 0,
        "below_policy_floor": 0,
    }
    for box, raw_score, raw_label in zip(
        primary.boxes_xyxy,
        primary.scores,
        primary.labels,
        strict=True,
    ):
        score = float(raw_score)
        label = int(raw_label)
        keep = False
        reason = "below_policy_floor"
        if label == config.vehicle_label:
            if score >= config.vehicle_protect_threshold:
                keep = True
                reason = "core_kept"
            elif score >= config.primary_threshold:
                if _has_support(
                    box,
                    label,
                    vehicle_support,
                    minimum_score=config.vehicle_support_threshold,
                    minimum_iou=config.vehicle_support_iou,
                ):
                    keep = True
                    reason = "vehicle_risk_supported"
                else:
                    reason = "vehicle_risk_rejected"
        elif score >= config.primary_threshold:
            keep = True
            reason = "core_kept"
        elif (
            label in config.ship_rescue_labels
            and score >= config.ship_candidate_floor
            and _has_support(
                box,
                label,
                ship_support,
                minimum_score=config.ship_support_threshold,
                minimum_iou=config.ship_support_iou,
            )
        ):
            keep = True
            reason = "ship_rescued"
        stats[reason] += 1
        if keep:
            kept_boxes.append([float(value) for value in box])
            kept_scores.append(score)
            kept_labels.append(label)
    return (
        Prediction(
            image_id=primary.image_id,
            boxes_xyxy=kept_boxes,
            scores=kept_scores,
            labels=kept_labels,
        ),
        stats,
    )


__all__ = ["AprrConfig", "apply_aprr"]
