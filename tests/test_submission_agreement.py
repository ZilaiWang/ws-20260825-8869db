from __future__ import annotations

import pytest

from rsdet.contracts import Prediction
from rsdet.submission.agreement import apply_label_agreement, apply_vehicle_agreement


def test_vehicle_agreement_never_imports_specialist_geometry() -> None:
    primary = Prediction(
        image_id=7,
        boxes_xyxy=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        scores=[0.8, 0.6, 0.9],
        labels=[24, 24, 3],
    )
    specialist = Prediction(
        image_id=7,
        boxes_xyxy=[[1, 1, 11, 11], [100, 100, 110, 110], [40, 40, 50, 50]],
        scores=[0.5, 0.99, 1.0],
        labels=[24, 24, 4],
    )
    result = apply_vehicle_agreement(primary, specialist, support_iou=0.35)
    assert result.boxes_xyxy == primary.boxes_xyxy
    assert result.labels == primary.labels
    assert result.scores == pytest.approx([0.4, 0.0, 0.9])


def test_vehicle_agreement_requires_aligned_images() -> None:
    primary = Prediction(1, [], [], [])
    specialist = Prediction(2, [], [], [])
    with pytest.raises(ValueError, match="image IDs differ"):
        apply_vehicle_agreement(primary, specialist)


def test_label_agreement_rescores_only_explicit_fine_labels() -> None:
    primary = Prediction(
        image_id=9,
        boxes_xyxy=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        scores=[0.8, 0.6, 0.7],
        labels=[4, 5, 3],
    )
    specialist = Prediction(
        image_id=9,
        boxes_xyxy=[[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        scores=[0.5, 0.25, 0.1],
        labels=[4, 5, 3],
    )
    result = apply_label_agreement(primary, specialist, labels=(4, 5))
    assert result.scores == pytest.approx([0.4, 0.15, 0.7])
