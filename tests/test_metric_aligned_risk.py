from rsdet.hera_guard.metric_aligned import (
    BACKGROUND,
    CANONICAL,
    CROSS_COARSE,
    CROSS_FINE,
    DUPLICATE,
    LOCALIZATION,
    build_metric_aligned_roles,
)


def test_metric_aligned_roles_follow_official_one_winner() -> None:
    gt = {
        1: [
            {"bbox_xyxy": [0, 0, 10, 10], "category_id": 0},
            {"bbox_xyxy": [20, 20, 30, 30], "category_id": 4},
        ]
    }
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 9, 9], "score": 0.8},
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.7},
        {"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.6},
        {"image_id": 1, "category_id": 4, "bbox": [26, 26, 10, 10], "score": 0.5},
        {"image_id": 1, "category_id": 4, "bbox": [40, 40, 10, 10], "score": 0.4},
    ]
    mapping = {0: "ship", 1: "ship", 4: "aircraft"}
    result = build_metric_aligned_roles(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping=mapping,
        iou_thresholds={"ship": 0.5, "aircraft": 0.5, "vehicle": 0.35},
    )
    assert [row.role for row in result.roles] == [
        CANONICAL,
        DUPLICATE,
        CROSS_FINE,
        CROSS_COARSE,
        LOCALIZATION,
        BACKGROUND,
    ]
    assert result.official_tp == 1
    assert result.cross_coarse_foreground_pollution == 1


def test_metric_aligned_tie_is_deterministic() -> None:
    gt = {1: [{"bbox_xyxy": [0, 0, 10, 10], "category_id": 24}]}
    predictions = [
        {"image_id": 1, "category_id": 24, "bbox_xyxy": [0, 0, 10, 10], "score": 0.5},
        {"image_id": 1, "category_id": 24, "bbox_xyxy": [0, 0, 10, 10], "score": 0.5},
    ]
    result = build_metric_aligned_roles(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping={24: "vehicle"},
        iou_thresholds={"ship": 0.5, "aircraft": 0.5, "vehicle": 0.35},
    )
    assert [row.role for row in result.roles] == [CANONICAL, DUPLICATE]
