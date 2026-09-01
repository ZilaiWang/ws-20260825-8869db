"""Shared global/coarse/fine score-threshold deployment contract."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral

from rsdet.contracts import Prediction
from rsdet.data.xh_dataset import FINE_NAMES, coarse_name


def validate_threshold(value: float, name: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")
    return numeric


def normalize_fine_thresholds(
    thresholds: Mapping[int | str, float] | None,
    *,
    require_complete: bool = False,
) -> dict[int, float] | None:
    if thresholds is None:
        return None
    normalized: dict[int, float] = {}
    for raw_label, value in thresholds.items():
        if isinstance(raw_label, bool):
            raise ValueError("fine threshold category ids must be integers")
        if isinstance(raw_label, str):
            if not raw_label.isdigit() or str(int(raw_label)) != raw_label:
                raise ValueError("fine threshold category ids must be canonical integers")
            label = int(raw_label)
        elif isinstance(raw_label, Integral):
            label = int(raw_label)
        else:
            raise ValueError("fine threshold category ids must be integers")
        if label in normalized:
            raise ValueError(f"duplicate fine threshold category id: {label}")
        if not 0 <= label < len(FINE_NAMES):
            raise ValueError(f"fine threshold category id out of range: {label}")
        normalized[label] = validate_threshold(value, f"score_threshold_by_fine.{label}")
    if require_complete and set(normalized) != set(range(len(FINE_NAMES))):
        missing = sorted(set(range(len(FINE_NAMES))) - set(normalized))
        extra = sorted(set(normalized) - set(range(len(FINE_NAMES))))
        raise ValueError(
            "formal score_threshold_by_fine must cover all fine classes: "
            f"missing={missing}, extra={extra}"
        )
    return normalized


def effective_threshold(
    label: int,
    *,
    global_threshold: float,
    coarse_thresholds: Mapping[str, float] | None = None,
    fine_thresholds: Mapping[int, float] | None = None,
) -> float:
    """Return the frozen ``fine > coarse > global`` threshold."""

    label = int(label)
    if fine_thresholds is not None and label in fine_thresholds:
        return float(fine_thresholds[label])
    if coarse_thresholds is not None:
        return float(coarse_thresholds[coarse_name(label)])
    return float(global_threshold)


def filter_prediction_by_thresholds(
    prediction: Prediction,
    *,
    global_threshold: float,
    coarse_thresholds: Mapping[str, float] | None = None,
    fine_thresholds: Mapping[int, float] | None = None,
) -> Prediction:
    keep = [
        index
        for index, (score, label) in enumerate(
            zip(prediction.scores, prediction.labels, strict=True)
        )
        if float(score)
        >= effective_threshold(
            int(label),
            global_threshold=global_threshold,
            coarse_thresholds=coarse_thresholds,
            fine_thresholds=fine_thresholds,
        )
    ]
    return Prediction(
        prediction.image_id,
        [prediction.boxes_xyxy[index] for index in keep],
        [prediction.scores[index] for index in keep],
        [prediction.labels[index] for index in keep],
    )


__all__ = [
    "effective_threshold",
    "filter_prediction_by_thresholds",
    "normalize_fine_thresholds",
    "validate_threshold",
]
