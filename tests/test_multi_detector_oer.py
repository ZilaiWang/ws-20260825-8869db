import numpy as np

from rsdet.analysis.multi_detector_oer import (
    MULTI_DETECTOR_AGREEMENT_COLUMNS,
    box_iou,
    build_multi_detector_features,
    candidate_validity_labels,
    class_aware_nms_records,
    prediction_ledger,
)


def _records():
    return [
        {
            "image_id": 1,
            "category_id": 4,
            "bbox_xyxy": [0, 0, 10, 10],
            "score": 0.8,
            "model_key": "Y5",
        },
        {
            "image_id": 1,
            "category_id": 4,
            "bbox_xyxy": [1, 0, 11, 10],
            "score": 0.7,
            "model_key": "M3",
        },
        {
            "image_id": 1,
            "category_id": 5,
            "bbox_xyxy": [1, 0, 11, 10],
            "score": 0.6,
            "model_key": "M3",
        },
    ]


def test_box_iou_and_class_aware_nms() -> None:
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    records = _records() + [{**_records()[0], "score": 0.1}]
    kept = class_aware_nms_records(records, iou_threshold=0.5)
    assert len(kept) == 2


def test_features_expose_cross_model_fine_and_coarse_agreement() -> None:
    features, columns = build_multi_detector_features(
        _records(), category_mapping={4: "aircraft", 5: "aircraft"}
    )
    assert columns == MULTI_DETECTOR_AGREEMENT_COLUMNS
    fine_iou = columns.index("same_fine_best_iou")
    coarse_iou = columns.index("same_coarse_best_iou")
    assert features.shape == (3, len(columns))
    assert features[0, fine_iou] > 0.8
    assert features[2, fine_iou] == 0.0
    assert features[2, coarse_iou] > 0.8
    assert np.isfinite(features).all()


def test_candidate_validity_is_same_fine_and_uses_coarse_iou_threshold() -> None:
    records = _records()
    labels = candidate_validity_labels(
        records,
        gt_boxes={1: [{"category_id": 4, "bbox_xyxy": [0, 0, 10, 10]}]},
        category_mapping={4: "aircraft", 5: "aircraft"},
        iou_thresholds={"aircraft": 0.5},
    )
    assert labels.tolist() == [1, 1, 0]


def test_prediction_ledger_defers_trace_indices_until_after_filtering() -> None:
    records = [
        {"image_id": 2, "score": 0.5},
        {"image_id": 1, "score": 0.4},
        {"image_id": 2, "score": 0.3},
    ]
    ledger = prediction_ledger(records, (1, 2, 3))
    assert all("source_prediction_index" not in row for row in ledger[1])
    assert all("source_prediction_index" not in row for row in ledger[2])
    assert ledger[3] == []
