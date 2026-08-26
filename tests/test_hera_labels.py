from rsdet.hera_guard.labels import build_proposal_object_labels


def test_object_label_ignores_detector_fine_class_and_allows_duplicates() -> None:
    gt = {
        7: [
            {"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "category_id": 2},
        ]
    }
    predictions = [
        {
            "image_id": 7,
            "category_id": 3,
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            "source_prediction_index": 10,
        },
        {
            "image_id": 7,
            "category_id": 4,
            "bbox_xyxy": [1.0, 1.0, 9.0, 9.0],
            "source_prediction_index": 11,
        },
    ]
    labels = build_proposal_object_labels(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping={2: "aircraft", 3: "aircraft", 4: "aircraft"},
        iou_thresholds={"aircraft": 0.5},
    )
    assert labels[10].is_object and labels[10].matched_category_id == 2
    assert labels[11].is_object and labels[11].matched_category_id == 2


def test_object_label_uses_target_coarse_iou_threshold() -> None:
    gt = {1: [{"bbox_xyxy": [0.0, 0.0, 10.0, 10.0], "category_id": 24}]}
    predictions = [
        {
            "image_id": 1,
            "category_id": 2,
            "bbox_xyxy": [0.0, 0.0, 6.0, 6.0],
            "source_prediction_index": 0,
        }
    ]
    labels = build_proposal_object_labels(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping={2: "aircraft", 24: "vehicle"},
        iou_thresholds={"aircraft": 0.5, "vehicle": 0.35},
    )
    assert labels[0].is_object
    assert labels[0].matched_coarse == "vehicle"
