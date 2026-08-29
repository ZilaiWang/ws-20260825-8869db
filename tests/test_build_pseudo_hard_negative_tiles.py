from scripts.build_pseudo_hard_negative_tiles import (
    crop_window,
    select_background_candidates,
    yolo_labels_for_crop,
)


def test_crop_window_clamps_to_image() -> None:
    assert crop_window([950, 950, 20, 20], image_width=1000, image_height=1000, tile_size=200) == (
        800,
        800,
        1000,
        1000,
    )


def test_yolo_labels_for_crop_keeps_centered_gt() -> None:
    rows = yolo_labels_for_crop([{"category_id": 4, "bbox": [20, 30, 20, 10]}], (0, 0, 100, 100))
    assert rows == ["4 0.30000000 0.35000000 0.20000000 0.10000000"]


def test_select_background_candidates_rejects_gt_and_nearby_centers() -> None:
    predictions = [
        {"bbox": [0, 0, 10, 10], "score": 0.9},
        {"bbox": [100, 100, 10, 10], "score": 0.8},
        {"bbox": [110, 100, 10, 10], "score": 0.7},
    ]
    gt = [{"bbox": [0, 0, 10, 10]}]
    selected = select_background_candidates(
        predictions, gt, count=2, max_any_iou=0.1, min_center_distance=30
    )
    assert [row["score"] for row in selected] == [0.8]
