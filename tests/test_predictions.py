"""统一推理输入、预测校验和 COCO 转换测试。"""

import json
from pathlib import Path

import pytest

from rsdet.contracts import InferenceSample, Prediction
from rsdet.engine.predictor import predict_batches
from rsdet.models.registry import build_model
from rsdet.predictions import (
    predictions_to_coco_records,
    validate_coco_prediction_records,
    validate_prediction,
    write_coco_predictions,
)


def test_dummy_detector_preserves_input_image_ids() -> None:
    detector = build_model("dummy", {})
    samples = [
        InferenceSample(image_id=101, image=None, width=640, height=480),
        InferenceSample(image_id=205, image=None, width=320, height=320),
    ]

    outputs = predict_batches(detector, samples, batch_size=2, allowed_category_ids=range(25))

    assert [output.image_id for output in outputs] == [101, 205]
def test_prediction_to_coco_records() -> None:
    prediction = Prediction(
        image_id=7,
        boxes_xyxy=[[10.0, 20.0, 50.0, 80.0]],
        scores=[0.75],
        labels=[24],
    )

    records = predictions_to_coco_records([prediction], allowed_category_ids=range(25))

    assert records == [
        {
            "image_id": 7,
            "category_id": 24,
            "bbox": [10.0, 20.0, 40.0, 60.0],
            "score": 0.75,
        }
    ]


def test_write_coco_predictions_uses_standard_top_level_list(tmp_path: Path) -> None:
    output_path = tmp_path / "predictions.json"
    write_coco_predictions(
        output_path,
        [Prediction(image_id=1, boxes_xyxy=[], scores=[], labels=[])],
        allowed_category_ids=range(25),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize(
    "prediction, message",
    [
        (Prediction(1, [[0, 0, 10, 10]], [], [0]), "长度必须一致"),
        (Prediction(1, [[0, 0, 10, 10]], [1.2], [0]), r"必须在 \[0, 1\]"),
        (Prediction(1, [[0, 0, 10, 10]], [0.5], [25]), "不在允许类别"),
        (Prediction(1, [[-1, 0, 10, 10]], [0.5], [0]), "不能为负数"),
    ],
)
def test_invalid_predictions_are_rejected(prediction: Prediction, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        validate_prediction(prediction, allowed_category_ids=range(25))


def test_prediction_image_id_must_match_input() -> None:
    prediction = Prediction(2, [], [], [])
    with pytest.raises(ValueError, match="与输入 image_id=1 不一致"):
        validate_prediction(prediction, expected_image_id=1)


def test_coco_records_can_be_checked_against_gt_image_size() -> None:
    records = [{"image_id": 3, "category_id": 0, "bbox": [10, 20, 30, 40], "score": 0.8}]
    summary = validate_coco_prediction_records(
        records,
        allowed_category_ids=range(25),
        image_sizes={3: (100, 100)},
    )
    assert summary == {
        "detections": 1,
        "images_with_predictions": 1,
        "categories_with_predictions": 1,
    }


def test_coco_record_outside_gt_image_is_rejected() -> None:
    records = [{"image_id": 3, "category_id": 0, "bbox": [90, 90, 20, 20], "score": 0.8}]
    with pytest.raises(ValueError, match="超出图像宽度"):
        validate_coco_prediction_records(
            records,
            allowed_category_ids=range(25),
            image_sizes={3: (100, 100)},
        )
