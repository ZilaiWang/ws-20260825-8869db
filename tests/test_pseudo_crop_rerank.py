from PIL import Image

from rsdet.data.crop_classification import render_crop
from scripts.rerank_cv3_pseudo_with_crop import (
    _iou_xywh,
    _square_window,
    aircraft_same_class_nms,
)


def test_square_window_centers_rectangular_box() -> None:
    assert _square_window([10, 20, 40, 20]) == (10.0, 10.0, 50.0, 50.0)
    assert _square_window([10, 20, 20, 40]) == (0.0, 20.0, 40.0, 60.0)


def test_iou_xywh() -> None:
    assert _iou_xywh([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _iou_xywh([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0


def test_render_crop_rgb_fast_path_is_pixel_equivalent() -> None:
    image = Image.new("RGB", (16, 12), (10, 20, 30))
    direct = render_crop(image, (2, 1, 10, 9), 8)
    converted = render_crop(image.convert("RGBA"), (2, 1, 10, 9), 8)
    assert direct.mode == "RGB"
    assert direct.tobytes() == converted.tobytes()


def test_aircraft_nms_is_same_class_and_aircraft_only() -> None:
    predictions = [
        {"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 4, "bbox": [0, 0, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 5, "bbox": [0, 0, 10, 10], "score": 0.7},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.6},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.5},
    ]
    kept = aircraft_same_class_nms(predictions, 0.5)
    assert [item["score"] for item in kept] == [0.9, 0.7, 0.6, 0.5]
