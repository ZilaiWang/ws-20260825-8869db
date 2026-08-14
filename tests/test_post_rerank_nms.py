from __future__ import annotations

import copy

import pytest

from rsdet.analysis.post_rerank_nms import select_nms_workpoint
from rsdet.postprocess.nms import class_aware_nms_predictions


def _prediction(
    category_id: int,
    score: float,
    box: list[float],
    *,
    uid: str,
) -> dict[str, object]:
    return {
        "category_id": category_id,
        "score": score,
        "bbox_xyxy": box,
        "uid": uid,
    }


def test_post_rerank_nms_suppresses_only_same_fine_class() -> None:
    predictions = {
        10: [
            _prediction(9, 0.90, [0, 0, 10, 10], uid="winner"),
            _prediction(9, 0.80, [1, 1, 11, 11], uid="duplicate"),
            _prediction(10, 0.70, [1, 1, 11, 11], uid="other-class"),
        ]
    }
    original = copy.deepcopy(predictions)

    result = class_aware_nms_predictions(predictions, 0.5)

    assert [record["uid"] for record in result[10]] == ["winner", "other-class"]
    assert predictions == original


def test_post_rerank_nms_is_deterministic_for_equal_scores() -> None:
    predictions = {
        1: [
            _prediction(4, 0.5, [0, 0, 10, 10], uid="first"),
            _prediction(4, 0.5, [0, 0, 10, 10], uid="second"),
        ],
        2: [],
    }

    first = class_aware_nms_predictions(predictions, 0.7)
    second = class_aware_nms_predictions(predictions, 0.7)

    assert first == second
    assert [record["uid"] for record in first[1]] == ["first"]
    assert first[2] == []


def test_post_rerank_nms_preserves_categories_outside_allowlist() -> None:
    predictions = {
        1: [
            _prediction(2, 0.9, [0, 0, 10, 10], uid="ship-a"),
            _prediction(2, 0.8, [0, 0, 10, 10], uid="ship-b"),
            _prediction(9, 0.7, [0, 0, 10, 10], uid="aircraft-a"),
            _prediction(9, 0.6, [0, 0, 10, 10], uid="aircraft-b"),
        ]
    }

    result = class_aware_nms_predictions(
        predictions,
        0.7,
        category_ids=range(4, 24),
    )

    assert [record["uid"] for record in result[1]] == [
        "ship-a",
        "ship-b",
        "aircraft-a",
    ]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_post_rerank_nms_rejects_invalid_threshold(threshold: float) -> None:
    with pytest.raises(ValueError):
        class_aware_nms_predictions({}, threshold)


def test_select_nms_workpoint_respects_recall_budget() -> None:
    curve = [
        {
            "iou_threshold": 1.0,
            "pooled_recall": 0.9300,
            "macro_fdr": 0.20,
        },
        {
            "iou_threshold": 0.7,
            "pooled_recall": 0.9295,
            "macro_fdr": 0.16,
        },
        {
            "iou_threshold": 0.6,
            "pooled_recall": 0.9288,
            "macro_fdr": 0.14,
        },
    ]

    result = select_nms_workpoint(curve, maximum_pooled_recall_drop=0.001)

    assert result["selected_iou_threshold"] == 0.7
    assert result["eligible_count"] == 2


def test_select_nms_workpoint_prefers_conservative_threshold_on_exact_tie() -> None:
    curve = [
        {
            "iou_threshold": 1.0,
            "pooled_recall": 0.93,
            "macro_fdr": 0.20,
        },
        {
            "iou_threshold": 0.8,
            "pooled_recall": 0.93,
            "macro_fdr": 0.15,
        },
        {
            "iou_threshold": 0.7,
            "pooled_recall": 0.93,
            "macro_fdr": 0.15,
        },
    ]

    result = select_nms_workpoint(curve, maximum_pooled_recall_drop=0.001)

    assert result["selected_iou_threshold"] == 0.8
