"""Official OER label contract tests."""

from rsdet.analysis.oer_labels import (
    LABEL_CONTRACT_VERSION,
    build_official_proposal_labels,
)


def test_oer_labels_use_prediction_first_official_trace() -> None:
    gt = {
        1: [
            {"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]},
            {"category_id": 5, "bbox_xyxy": [60.0, 0.0, 160.0, 100.0]},
        ]
    }
    predictions = [
        {
            "image_id": 1,
            "category_id": 5,
            "score": 0.90,
            "bbox_xyxy": [32.0, 0.0, 132.0, 100.0],
            "source_prediction_index": 10,
        },
        {
            "image_id": 1,
            "category_id": 5,
            "score": 0.80,
            "bbox_xyxy": [80.0, 0.0, 160.0, 100.0],
            "source_prediction_index": 11,
        },
    ]
    result = build_official_proposal_labels(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping={5: "aircraft"},
        iou_thresholds={"aircraft": 0.50},
    )
    assert result.contract_version == LABEL_CONTRACT_VERSION
    assert result.labels[10].is_valid
    assert result.labels[10].matched_gt_index == 1
    assert result.labels[10].matched_iou > 0.50
    assert not result.labels[11].is_valid
    assert result.labels[11].matched_gt_index is None
    assert result.metrics.details["tp"] == 1
    assert result.metrics.details["fp"] == 1


def test_oer_labels_reject_duplicate_global_candidate_ids() -> None:
    gt = {1: []}
    predictions = [
        {
            "image_id": 1,
            "category_id": 5,
            "score": 0.9,
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            "source_prediction_index": 7,
        },
        {
            "image_id": 1,
            "category_id": 5,
            "score": 0.8,
            "bbox_xyxy": [20.0, 20.0, 30.0, 30.0],
            "source_prediction_index": 7,
        },
    ]
    import pytest

    with pytest.raises(ValueError, match="globally unique"):
        build_official_proposal_labels(
            gt_boxes=gt,
            predictions=predictions,
            category_mapping={5: "aircraft"},
            iou_thresholds={"aircraft": 0.50},
        )
