"""Class-disjoint routing after two independent large-image pipelines."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import Any

from rsdet.contracts import Prediction
from rsdet.models.base import BaseDetector
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline
from rsdet.postprocess.nms import compute_iou

OFFICIAL_LABELS = frozenset(range(25))


def _labels(values: Iterable[int], name: str) -> frozenset[int]:
    result = frozenset(int(value) for value in values)
    if not result or result - OFFICIAL_LABELS:
        raise ValueError(f"{name} must be a non-empty subset of labels 0..24")
    return result


@dataclass(frozen=True)
class ResolutionLabelRoute:
    """Assign every official label to exactly one detector branch."""

    primary_labels: frozenset[int]
    expert_labels: frozenset[int]
    primary_threshold: float
    expert_threshold: float

    def __post_init__(self) -> None:
        primary = _labels(self.primary_labels, "primary_labels")
        expert = _labels(self.expert_labels, "expert_labels")
        if primary & expert:
            raise ValueError("resolution route label ownership must be disjoint")
        if primary | expert != OFFICIAL_LABELS:
            raise ValueError("resolution route must cover all official labels 0..24")
        for name, threshold in (
            ("primary_threshold", self.primary_threshold),
            ("expert_threshold", self.expert_threshold),
        ):
            if not 0.0 <= float(threshold) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        object.__setattr__(self, "primary_labels", primary)
        object.__setattr__(self, "expert_labels", expert)
        object.__setattr__(self, "primary_threshold", float(self.primary_threshold))
        object.__setattr__(self, "expert_threshold", float(self.expert_threshold))


@dataclass(frozen=True)
class PrimaryLabelRescue:
    """Append high-confidence primary boxes for labels owned by the expert.

    Expert-owned boxes always have priority.  A primary candidate is appended
    only when it does not overlap an expert box or an earlier accepted rescue
    box of the same fine category.  This makes the operation a true fallback:
    it cannot replace expert geometry merely because scores from the two
    independently trained branches are not calibrated to one another.
    """

    labels: frozenset[int]
    threshold: float
    dedup_iou: float

    def __post_init__(self) -> None:
        labels = _labels(self.labels, "primary rescue labels")
        threshold = float(self.threshold)
        dedup_iou = float(self.dedup_iou)
        if not isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("primary rescue threshold must be finite in [0, 1]")
        if not isfinite(dedup_iou) or not 0.0 < dedup_iou <= 1.0:
            raise ValueError("primary rescue dedup_iou must be finite in (0, 1]")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "dedup_iou", dedup_iou)


def _select(
    prediction: Prediction, labels: frozenset[int], threshold: float
) -> list[tuple[list[float], float, int]]:
    count = len(prediction.boxes_xyxy)
    if len(prediction.scores) != count or len(prediction.labels) != count:
        raise ValueError("prediction arrays have inconsistent lengths")
    invalid = {int(label) for label in prediction.labels} - OFFICIAL_LABELS
    if invalid:
        raise ValueError(f"prediction contains invalid labels: {sorted(invalid)}")
    return [
        (
            [float(value) for value in prediction.boxes_xyxy[index]],
            float(prediction.scores[index]),
            int(prediction.labels[index]),
        )
        for index in range(count)
        if int(prediction.labels[index]) in labels and float(prediction.scores[index]) >= threshold
    ]


def compose_routed_predictions(
    primary: Prediction,
    expert: Prediction,
    *,
    route: ResolutionLabelRoute,
    primary_rescue: PrimaryLabelRescue | None = None,
) -> Prediction:
    """Filter branch-owned labels, optionally append primary fallback boxes."""

    if primary.image_id != expert.image_id:
        raise ValueError("cannot compose predictions with different image ids")
    primary_rows = _select(primary, route.primary_labels, route.primary_threshold)
    expert_rows = _select(expert, route.expert_labels, route.expert_threshold)
    rows = [*primary_rows, *expert_rows]
    if primary_rescue is not None:
        if not primary_rescue.labels <= route.expert_labels:
            raise ValueError("primary rescue labels must be owned by the expert branch")
        rescue_candidates = _select(
            primary,
            primary_rescue.labels,
            primary_rescue.threshold,
        )
        rescue_candidates.sort(key=lambda row: (-row[1], row[2], tuple(row[0])))
        protected = [row for row in expert_rows if row[2] in primary_rescue.labels]
        for candidate in rescue_candidates:
            if any(
                old[2] == candidate[2]
                and compute_iou(candidate[0], old[0]) >= primary_rescue.dedup_iou
                for old in protected
            ):
                continue
            rows.append(candidate)
            protected.append(candidate)
    rows.sort(key=lambda row: (-row[1], row[2], tuple(row[0])))
    return Prediction(
        image_id=primary.image_id,
        boxes_xyxy=[row[0] for row in rows],
        scores=[row[1] for row in rows],
        labels=[row[2] for row in rows],
    )


class DualPipelineResolutionRuntime:
    """Run two safe-fusion pipelines before class-disjoint composition."""

    def __init__(
        self,
        primary: BaseDetector,
        expert: BaseDetector,
        *,
        primary_pipeline: PipelineConfig,
        expert_pipeline: PipelineConfig,
        route: ResolutionLabelRoute,
        primary_rescue: PrimaryLabelRescue | None = None,
    ) -> None:
        self.primary = primary
        self.expert = expert
        self.primary_pipeline = primary_pipeline
        self.expert_pipeline = expert_pipeline
        self.route = route
        self.primary_rescue = primary_rescue

    def to(self, device: str) -> None:
        self.primary.to(device)
        self.expert.to(device)

    def eval(self) -> None:
        self.primary.eval()
        self.expert.eval()

    def predict_image(
        self, image: Any, *, parent_image_id: int = 0
    ) -> tuple[Prediction, dict[str, Any]]:
        primary, primary_timing = run_pipeline(
            image,
            self.primary,
            config=self.primary_pipeline,
            parent_image_id=parent_image_id,
        )
        expert, expert_timing = run_pipeline(
            image,
            self.expert,
            config=self.expert_pipeline,
            parent_image_id=parent_image_id,
        )
        return compose_routed_predictions(
            primary,
            expert,
            route=self.route,
            primary_rescue=self.primary_rescue,
        ), {
            "primary": primary_timing.to_dict(),
            "expert": expert_timing.to_dict(),
            "pipeline_seconds_sum": primary_timing.pipeline_s + expert_timing.pipeline_s,
        }


__all__ = [
    "DualPipelineResolutionRuntime",
    "PrimaryLabelRescue",
    "ResolutionLabelRoute",
    "compose_routed_predictions",
]
