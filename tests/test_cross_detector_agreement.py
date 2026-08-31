from __future__ import annotations

import pytest

from rsdet.analysis.cross_detector_agreement import (
    best_same_fine_support,
    marginal_false_detection_rate,
)


def _row(image: int, category: int, box: list[float], score: float) -> dict[str, object]:
    return {
        "image_id": image,
        "category_id": category,
        "bbox_xyxy": box,
        "score": score,
    }


def test_support_requires_same_image_fine_and_iou() -> None:
    primary = [_row(1, 24, [0, 0, 10, 10], 0.2)]
    specialist = [
        _row(1, 24, [0, 0, 10, 10], 0.7),
        _row(1, 24, [1, 1, 9, 9], 0.9),
        _row(1, 23, [0, 0, 10, 10], 1.0),
        _row(2, 24, [0, 0, 10, 10], 1.0),
    ]
    result = best_same_fine_support(primary, specialist, iou_threshold=0.35)
    assert result == [
        {
            "support_score": 0.9,
            "support_iou": pytest.approx(0.64),
            "agreement_product": pytest.approx(0.18),
        }
    ]


def test_support_is_zero_without_valid_geometry() -> None:
    result = best_same_fine_support(
        [_row(1, 24, [0, 0, 10, 10], 0.2)],
        [_row(1, 24, [9, 9, 19, 19], 0.9)],
        iou_threshold=0.35,
    )
    assert result[0] == {
        "support_score": 0.0,
        "support_iou": 0.0,
        "agreement_product": 0.0,
    }


def test_marginal_fdr() -> None:
    assert marginal_false_detection_rate(22, 6) == pytest.approx(6 / 28)
    assert marginal_false_detection_rate(0, 0) == 0.0
    with pytest.raises(ValueError):
        marginal_false_detection_rate(-1, 0)
