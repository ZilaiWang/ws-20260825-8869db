"""Deployment-safe cross-detector agreement scoring.

The specialist never contributes geometry or labels.  It may only rescore an
existing primary proposal by same-fine geometric support.
"""

from __future__ import annotations

from collections.abc import Collection

from rsdet.contracts import Prediction


def _iou(first: list[float], second: list[float]) -> float:
    x0 = max(float(first[0]), float(second[0]))
    y0 = max(float(first[1]), float(second[1]))
    x1 = min(float(first[2]), float(second[2]))
    y1 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return 0.0 if union <= 0.0 else intersection / union


def apply_label_agreement(
    primary: Prediction,
    specialist: Prediction,
    *,
    labels: Collection[int],
    support_iou: float = 0.35,
) -> Prediction:
    """Multiply selected primary scores by best same-fine specialist support."""
    if primary.image_id != specialist.image_id:
        raise ValueError("primary and specialist prediction image IDs differ")
    if not 0.0 <= support_iou <= 1.0:
        raise ValueError("support_iou must be in [0, 1]")
    selected_labels = {int(label) for label in labels}
    specialist_rows: dict[int, list[tuple[list[float], float]]] = {}
    for box, score, label in zip(
        specialist.boxes_xyxy,
        specialist.scores,
        specialist.labels,
        strict=True,
    ):
        label = int(label)
        if label in selected_labels:
            specialist_rows.setdefault(label, []).append((box, float(score)))
    rescored: list[float] = []
    for box, score, label in zip(
        primary.boxes_xyxy,
        primary.scores,
        primary.labels,
        strict=True,
    ):
        score = float(score)
        label = int(label)
        if label not in selected_labels:
            rescored.append(score)
            continue
        support = max(
            (
                specialist_score
                for specialist_box, specialist_score in specialist_rows.get(label, ())
                if _iou(box, specialist_box) >= support_iou
            ),
            default=0.0,
        )
        rescored.append(score * support)
    return Prediction(
        image_id=primary.image_id,
        boxes_xyxy=[list(box) for box in primary.boxes_xyxy],
        scores=rescored,
        labels=[int(label) for label in primary.labels],
    )


def apply_vehicle_agreement(
    primary: Prediction,
    specialist: Prediction,
    *,
    vehicle_label: int = 24,
    support_iou: float = 0.35,
) -> Prediction:
    """Backward-compatible vehicle-only agreement route."""
    return apply_label_agreement(
        primary,
        specialist,
        labels=(vehicle_label,),
        support_iou=support_iou,
    )
