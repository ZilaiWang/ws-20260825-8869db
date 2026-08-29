from scripts.merge_cv3_pseudo_vehicle_multiscale import merge_vehicle_predictions


def _pred(image_id: int, category_id: int, score: float, x: float) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "score": score,
        "bbox": [x, 0.0, 10.0, 10.0],
    }


def test_only_secondary_vehicle_is_added_and_duplicates_are_suppressed() -> None:
    primary = [_pred(1, 3, 0.9, 0.0), _pred(1, 24, 0.8, 20.0)]
    secondary = [
        _pred(1, 3, 0.99, 50.0),
        _pred(1, 24, 0.7, 20.5),
        _pred(1, 24, 0.6, 50.0),
    ]
    merged, summary = merge_vehicle_predictions(primary, secondary, nms_iou=0.5)

    assert sum(item["category_id"] == 3 for item in merged) == 1
    vehicles = [item for item in merged if item["category_id"] == 24]
    assert len(vehicles) == 2
    assert {item["multiscale_source"] for item in vehicles} == {
        "primary_1024",
        "secondary_800",
    }
    assert summary["kept_secondary_vehicle"] == 1


def test_score_scale_is_applied_before_nms() -> None:
    primary = [_pred(1, 24, 0.8, 20.0)]
    secondary = [_pred(1, 24, 0.9, 20.0)]
    merged, _ = merge_vehicle_predictions(
        primary, secondary, nms_iou=0.5, secondary_score_scale=0.5
    )
    assert len(merged) == 1
    assert merged[0]["multiscale_source"] == "primary_1024"
