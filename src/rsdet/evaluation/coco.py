"""评估脚本共用的 COCO 标注读取工具。"""

import json
import math
from pathlib import Path
from typing import Any


def load_coco_ground_truth(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    """读取 COCO ground truth，并按 ``image_id`` 分组。"""
    data = _load_json(Path(path))
    if not isinstance(data, dict) or not isinstance(data.get("annotations"), list):
        raise ValueError("GT 必须是包含 annotations 列表的 COCO JSON 对象")
    grouped = _group_annotations(data["annotations"], require_score=False)
    # Negative images are part of the evaluation universe.  Callers commonly
    # iterate over GT keys before filtering predictions; dropping an empty
    # image here would silently drop all of its false positives downstream.
    if "images" in data:
        if not isinstance(data["images"], list):
            raise ValueError("GT images 必须是列表")
        image_ids = [int(image["id"]) for image in data["images"]]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError("GT images 包含重复 image id")
        unknown = set(grouped) - set(image_ids)
        if unknown:
            raise ValueError(f"GT annotations 引用了 images 之外的 id: {sorted(unknown)[:10]}")
        return {image_id: grouped.get(image_id, []) for image_id in image_ids}
    # Legacy annotation-only inputs remain readable; their negative-image
    # coverage is unknown, not zero.
    return grouped


def load_coco_predictions(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    """读取标准 COCO detection list，也兼容含 annotations 的对象。"""
    data = _load_json(Path(path))
    if isinstance(data, list):
        annotations = data
    elif isinstance(data, dict) and isinstance(data.get("annotations"), list):
        annotations = data["annotations"]
    else:
        raise ValueError("预测文件必须是 COCO detection 列表或包含 annotations 的对象")
    return _group_annotations(annotations, require_score=True)


def _load_json(path: Path) -> Any:
    """读取 JSON 文件。"""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _xywh_to_xyxy(box: list[float]) -> list[float]:
    """把 COCO ``xywh`` 转成 ``xyxy``。"""
    if len(box) != 4 or box[2] < 0 or box[3] < 0:
        raise ValueError(f"非法 COCO bbox: {box}")
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def _group_annotations(
    annotations: list[dict[str, Any]],
    *,
    require_score: bool,
) -> dict[int, list[dict[str, Any]]]:
    """把 COCO 记录按 ``image_id`` 分组。"""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        record: dict[str, Any] = {
            "bbox_xyxy": _xywh_to_xyxy([float(value) for value in annotation["bbox"]]),
            "category_id": int(annotation["category_id"]),
        }
        if require_score:
            if "score" not in annotation:
                raise ValueError("预测记录缺少 score")
            score = float(annotation["score"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"预测 score 必须是 [0, 1] 内的有限数: {score}")
            record["score"] = score
        grouped.setdefault(image_id, []).append(record)
    return grouped
