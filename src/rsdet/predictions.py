"""统一预测校验和 COCO detection JSON 转换。

该模块不依赖 PyTorch、Ultralytics 或其他模型框架。模型成员只需输出标准
``Prediction`` 或标准 COCO detection JSON，即可接入公共评测和大图流程。
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

from rsdet.contracts import Prediction


def _as_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{field_name} 必须是整数，当前为 {value!r}")
    return int(value)


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field_name} 必须是数值，当前为 {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"{field_name} 必须是有限数，当前为 {value!r}")
    return result


def _validate_xyxy(box: Sequence[Any], *, width: int | None, height: int | None) -> list[float]:
    if len(box) != 4:
        raise ValueError(f"xyxy bbox 必须包含 4 个数值，当前为 {box!r}")
    x1, y1, x2, y2 = [
        _finite_float(value, f"bbox[{index}]") for index, value in enumerate(box)
    ]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"xyxy bbox 必须满足 x2>x1 且 y2>y1，当前为 {[x1, y1, x2, y2]}")
    if x1 < 0.0 or y1 < 0.0:
        raise ValueError(f"bbox 左上角不能为负数: {[x1, y1, x2, y2]}")
    if width is not None and x2 > width:
        raise ValueError(f"bbox x 坐标超出图像宽度 {width}: {[x1, y1, x2, y2]}")
    if height is not None and (y1 < 0.0 or y2 > height):
        raise ValueError(f"bbox y 坐标超出图像高度 {height}: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def validate_prediction(
    prediction: Prediction,
    *,
    expected_image_id: int | None = None,
    allowed_category_ids: Iterable[int] | None = None,
    image_size: tuple[int, int] | None = None,
) -> None:
    """校验一个模型无关预测。

    ``image_size`` 使用 ``(width, height)``。省略时只检查坐标自身合法性，
    不检查是否越出图像边界。
    """
    if not isinstance(prediction, Prediction):
        raise TypeError(f"模型输出必须是 Prediction，当前为 {type(prediction).__name__}")
    image_id = _as_int(prediction.image_id, "image_id")
    if expected_image_id is not None:
        expected = _as_int(expected_image_id, "expected_image_id")
        if image_id != expected:
            raise ValueError(f"预测 image_id={image_id} 与输入 image_id={expected} 不一致")

    try:
        boxes = list(prediction.boxes_xyxy)
        scores = list(prediction.scores)
        labels = list(prediction.labels)
    except TypeError as error:
        raise ValueError("boxes_xyxy、scores、labels 必须是可迭代序列") from error
    if not (len(boxes) == len(scores) == len(labels)):
        raise ValueError(
            "boxes_xyxy、scores、labels 长度必须一致: "
            f"{len(boxes)}, {len(scores)}, {len(labels)}"
        )

    width = height = None
    if image_size is not None:
        if len(image_size) != 2:
            raise ValueError(f"image_size 必须是 (width, height)，当前为 {image_size}")
        width = _as_int(image_size[0], "image_width")
        height = _as_int(image_size[1], "image_height")
        if width <= 0 or height <= 0:
            raise ValueError(f"图像尺寸必须为正数，当前为 {image_size}")

    allowed = None if allowed_category_ids is None else {int(value) for value in allowed_category_ids}
    for index, (box, score_value, label_value) in enumerate(zip(boxes, scores, labels)):
        _validate_xyxy(box, width=width, height=height)
        score = _finite_float(score_value, f"scores[{index}]")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"scores[{index}] 必须在 [0, 1]，当前为 {score}")
        label = _as_int(label_value, f"labels[{index}]")
        if allowed is not None and label not in allowed:
            raise ValueError(f"labels[{index}]={label} 不在允许类别中")


def predictions_to_coco_records(
    predictions: Iterable[Prediction],
    *,
    allowed_category_ids: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    """把统一预测转换成标准 COCO detection 顶层列表。"""
    records: list[dict[str, Any]] = []
    allowed = None if allowed_category_ids is None else tuple(allowed_category_ids)
    for prediction in predictions:
        validate_prediction(prediction, allowed_category_ids=allowed)
        for box, score, label in zip(
            prediction.boxes_xyxy,
            prediction.scores,
            prediction.labels,
        ):
            x1, y1, x2, y2 = [float(value) for value in box]
            records.append(
                {
                    "image_id": int(prediction.image_id),
                    "category_id": int(label),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                }
            )
    return records


def write_coco_predictions(
    path: str | Path,
    predictions: Iterable[Prediction],
    *,
    allowed_category_ids: Iterable[int] | None = None,
) -> None:
    """校验并写出标准 COCO detection JSON。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = predictions_to_coco_records(
        predictions,
        allowed_category_ids=allowed_category_ids,
    )
    output_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_coco_prediction_records(path: str | Path) -> list[Mapping[str, Any]]:
    """读取 COCO detection 列表，也兼容含 ``annotations`` 的对象。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        records = data
    elif isinstance(data, Mapping) and isinstance(data.get("annotations"), list):
        records = data["annotations"]
    else:
        raise ValueError("预测文件必须是 COCO detection 列表或含 annotations 的对象")
    if not all(isinstance(record, Mapping) for record in records):
        raise ValueError("预测列表中的每一项都必须是对象")
    return records


def validate_coco_prediction_records(
    records: Sequence[Mapping[str, Any]],
    *,
    allowed_category_ids: Iterable[int],
    image_sizes: Mapping[int, tuple[int, int]] | None = None,
) -> dict[str, int]:
    """校验框架直接导出的 COCO detection 记录并返回简要统计。"""
    allowed = {int(value) for value in allowed_category_ids}
    image_ids: set[int] = set()
    category_ids: set[int] = set()
    for index, record in enumerate(records):
        missing = {"image_id", "category_id", "bbox", "score"} - set(record)
        if missing:
            raise ValueError(f"预测第 {index} 项缺少字段: {sorted(missing)}")

        image_id = _as_int(record["image_id"], f"records[{index}].image_id")
        category_id = _as_int(record["category_id"], f"records[{index}].category_id")
        if category_id not in allowed:
            raise ValueError(f"预测第 {index} 项 category_id={category_id} 不在允许类别中")
        score = _finite_float(record["score"], f"records[{index}].score")
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"预测第 {index} 项 score 必须在 [0, 1]，当前为 {score}")

        width = height = None
        if image_sizes is not None:
            if image_id not in image_sizes:
                raise ValueError(f"预测第 {index} 项 image_id={image_id} 不在图像清单中")
            image_size = image_sizes[image_id]
            if len(image_size) != 2:
                raise ValueError(f"image_sizes[{image_id}] 必须是 (width, height)")
            width = _as_int(image_size[0], f"image_sizes[{image_id}].width")
            height = _as_int(image_size[1], f"image_sizes[{image_id}].height")
            if width <= 0 or height <= 0:
                raise ValueError(f"image_id={image_id} 的图像尺寸必须为正数")

        bbox = record["bbox"]
        if not isinstance(bbox, Sequence) or isinstance(bbox, (str, bytes)) or len(bbox) != 4:
            raise ValueError(f"预测第 {index} 项 bbox 必须为 4 个数值")
        x, y, box_width, box_height = [
            _finite_float(value, f"records[{index}].bbox[{position}]")
            for position, value in enumerate(bbox)
        ]
        _validate_xyxy(
            [x, y, x + box_width, y + box_height],
            width=width,
            height=height,
        )
        image_ids.add(image_id)
        category_ids.add(category_id)

    return {
        "detections": len(records),
        "images_with_predictions": len(image_ids),
        "categories_with_predictions": len(category_ids),
    }
