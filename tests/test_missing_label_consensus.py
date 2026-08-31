from __future__ import annotations

import pytest

from rsdet.analysis.missing_label_consensus import (
    build_missing_label_candidates,
    xywh_to_xyxy,
)


def _row(image: int, category: int, box: list[float], score: float) -> dict[str, object]:
    return {"image_id": image, "category_id": category, "bbox": box, "score": score}


def test_xywh_validation() -> None:
    assert xywh_to_xyxy([1, 2, 3, 4]) == [1, 2, 4, 6]
    with pytest.raises(ValueError):
        xywh_to_xyxy([0, 0, 0, 1])


def test_candidate_requires_support_and_no_existing_gt() -> None:
    primary = [
        _row(1, 24, [0, 0, 10, 10], 0.20),
        _row(1, 24, [20, 20, 10, 10], 0.90),
        _row(1, 23, [40, 40, 10, 10], 1.00),
    ]
    specialist = [
        _row(1, 24, [1, 1, 10, 10], 0.50),
        _row(1, 24, [20, 20, 10, 10], 0.90),
    ]
    annotations = [{"image_id": 1, "category_id": 3, "bbox": [20, 20, 10, 10]}]
    result = build_missing_label_candidates(
        primary,
        specialist,
        annotations,
        category_id=24,
        support_iou_min=0.35,
        existing_gt_iou_max=0.05,
        product_min=0.059,
    )
    assert len(result) == 1
    assert result[0]["agreement_product"] == pytest.approx(0.10)
    assert result[0]["support_iou"] == pytest.approx(81 / 119)
    assert result[0]["maximum_gt_iou"] == 0.0


def test_candidates_are_ranked_and_deduplicated() -> None:
    primary = [
        _row(1, 24, [0, 0, 10, 10], 0.5),
        _row(1, 24, [1, 1, 10, 10], 0.4),
        _row(2, 24, [0, 0, 10, 10], 0.3),
    ]
    specialist = [
        _row(1, 24, [0, 0, 10, 10], 0.8),
        _row(2, 24, [0, 0, 10, 10], 0.9),
    ]
    result = build_missing_label_candidates(
        primary,
        specialist,
        [],
        category_id=24,
        support_iou_min=0.35,
        existing_gt_iou_max=0.05,
        product_min=0.05,
        dedup_iou=0.5,
    )
    assert [(row["image_id"], row["agreement_product"]) for row in result] == [
        (1, pytest.approx(0.4)),
        (2, pytest.approx(0.27)),
    ]
