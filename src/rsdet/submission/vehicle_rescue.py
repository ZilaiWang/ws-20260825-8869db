"""Selective vehicle reject-and-rescue using a geometry-preserving specialist."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rsdet.contracts import InferenceSample, Prediction
from rsdet.models.base import BaseDetector
from rsdet.submission.agreement import _iou


@dataclass(frozen=True)
class VehicleRescueConfig:
    vehicle_label: int = 24
    core_threshold: float = 0.22
    candidate_floor: float = 0.01
    support_iou: float = 0.35
    rescue_product_threshold: float = 0.059
    promoted_score: float = 0.220001

    def __post_init__(self) -> None:
        values = (
            self.core_threshold,
            self.candidate_floor,
            self.support_iou,
            self.rescue_product_threshold,
            self.promoted_score,
        )
        if any(not 0.0 <= float(value) <= 1.0 for value in values):
            raise ValueError("vehicle rescue thresholds must be in [0, 1]")
        if self.candidate_floor > self.core_threshold:
            raise ValueError("candidate_floor cannot exceed core_threshold")
        if self.promoted_score < self.core_threshold:
            raise ValueError("promoted_score must pass core_threshold")


def apply_vehicle_reject_rescue(
    primary: Prediction,
    specialist: Prediction,
    *,
    config: VehicleRescueConfig,
) -> Prediction:
    """Keep core vehicles, reject unsupported tails and rescue supported tails.

    Ship/aircraft rows and primary geometry/fine labels are bitwise preserved.
    The specialist can only decide whether an existing primary vehicle row is
    retained; it never contributes a new box or category.
    """

    if primary.image_id != specialist.image_id:
        raise ValueError("primary and specialist prediction image IDs differ")
    support_rows = [
        (box, float(score))
        for box, score, label in zip(
            specialist.boxes_xyxy,
            specialist.scores,
            specialist.labels,
            strict=True,
        )
        if int(label) == config.vehicle_label
    ]
    boxes: list[list[float]] = []
    scores: list[float] = []
    labels: list[int] = []
    for box, raw_score, raw_label in zip(
        primary.boxes_xyxy,
        primary.scores,
        primary.labels,
        strict=True,
    ):
        score = float(raw_score)
        label = int(raw_label)
        output_score = score
        keep = True
        if label == config.vehicle_label and score < config.core_threshold:
            keep = score >= config.candidate_floor
            if keep:
                support = max(
                    (
                        specialist_score
                        for specialist_box, specialist_score in support_rows
                        if _iou(list(box), specialist_box) >= config.support_iou
                    ),
                    default=0.0,
                )
                keep = score * support >= config.rescue_product_threshold
                if keep:
                    output_score = max(score, config.promoted_score)
        if keep:
            boxes.append(list(box))
            scores.append(output_score)
            labels.append(label)
    return Prediction(primary.image_id, boxes, scores, labels)


class SelectiveVehicleRescueDetector(BaseDetector):
    """Run the specialist only for tiles containing tail vehicle proposals."""

    def __init__(
        self,
        primary: BaseDetector,
        specialist: BaseDetector,
        *,
        config: VehicleRescueConfig,
    ) -> None:
        self.primary = primary
        self.specialist = specialist
        self.config = config
        self.primary_tile_count = 0
        self.specialist_tile_count = 0

    def load(self, checkpoint_path: str) -> None:
        del checkpoint_path

    def to(self, device: str) -> None:
        self.primary.to(device)
        self.specialist.to(device)

    def eval(self) -> None:
        self.primary.eval()
        self.specialist.eval()

    def predict(self, batch: Sequence[InferenceSample]) -> list[Prediction]:
        primary_predictions = self.primary.predict(batch)
        if len(primary_predictions) != len(batch):
            raise RuntimeError("primary detector returned an invalid batch length")
        self.primary_tile_count += len(batch)
        selected_indices = [
            index
            for index, prediction in enumerate(primary_predictions)
            if any(
                int(label) == self.config.vehicle_label
                and self.config.candidate_floor <= float(score) < self.config.core_threshold
                for score, label in zip(
                    prediction.scores, prediction.labels, strict=True
                )
            )
        ]
        if not selected_indices:
            return primary_predictions
        specialist_predictions = self.specialist.predict(
            [batch[index] for index in selected_indices]
        )
        if len(specialist_predictions) != len(selected_indices):
            raise RuntimeError("specialist detector returned an invalid batch length")
        self.specialist_tile_count += len(selected_indices)
        output = list(primary_predictions)
        for index, specialist_prediction in zip(
            selected_indices, specialist_predictions, strict=True
        ):
            output[index] = apply_vehicle_reject_rescue(
                primary_predictions[index],
                specialist_prediction,
                config=self.config,
            )
        return output


__all__ = [
    "SelectiveVehicleRescueDetector",
    "VehicleRescueConfig",
    "apply_vehicle_reject_rescue",
]
