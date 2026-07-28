"""跨切片融合：坐标恢复、边界裁剪、NMS 去重。

本模块独立于具体检测器，只依赖 numpy 和公共契约数据类型。
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from rsdet.contracts import Prediction, TileRecord
from rsdet.tiling.coordinates import clip_bbox, tile_to_full

logger = logging.getLogger(__name__)

# 25 细类 → IoU 阈值（匹配 project.yaml 的三大类映射）
DEFAULT_IOU_THRESHOLDS: Dict[int, float] = {}
for _cid in range(0, 4):
    DEFAULT_IOU_THRESHOLDS[_cid] = 0.50  # ship
for _cid in range(4, 24):
    DEFAULT_IOU_THRESHOLDS[_cid] = 0.50  # aircraft
DEFAULT_IOU_THRESHOLDS[24] = 0.35  # vehicle


def _compute_ious(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算单个 box 与 boxes[N,4] 的逐对 IoU（纯 numpy 向量化）。"""
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return np.where(union > 0.0, inter / union, 0.0)


def _nms_per_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """对单类别运行 score-descending NMS，返回保留的索引。"""
    if len(boxes) == 0:
        return np.array([], dtype=np.intp)
    order = np.argsort(-scores)
    keep: List[int] = []
    while len(order) > 0:
        current = int(order[0])
        keep.append(current)
        if len(order) == 1:
            break
        rest = order[1:]
        ious = _compute_ious(boxes[current], boxes[rest])
        order = rest[ious <= iou_threshold]
    return np.array(keep, dtype=np.intp)


def fuse_tile_predictions(
    tile_predictions: List[Prediction],
    tile_records: List[TileRecord],
    *,
    image_width: int,
    image_height: int,
    parent_image_id: int = 0,
    score_threshold: float = 0.0,
    iou_thresholds: Dict[int, float] | None = None,
) -> Prediction:
    """将多个切片的预测融合为单张原图预测。

    流程：
        1. 每框：tile 局部坐标 + tile offset → 全局 xyxy
        2. 每框：clip 到 [0, image_width] × [0, image_height]
        3. 每框：score < score_threshold → 丢弃
        4. 按细类分组 → 各类内 NMS（IoU 阈值按 iou_thresholds）
        5. 输出单个 Prediction（全局坐标，image_id = parent_image_id）

    Args:
        tile_predictions: 各切片推理结果，同序于 tile_records。
        tile_records: 切片坐标记录（含 x_offset / y_offset / parent_image_id）。
        image_width: 原图宽度（像素）。
        image_height: 原图高度（像素）。
        parent_image_id: 原图 ID，填入输出 Prediction。
        score_threshold: 融合前的最低置信度阈值。
        iou_thresholds: 细类 ID → NMS IoU 阈值；未指定时使用 DEFAULT_IOU_THRESHOLDS。

    Returns:
        融合后的单 Prediction，所有框均为全局像素坐标。
    """
    if len(tile_predictions) != len(tile_records):
        raise ValueError(
            f"tile_predictions 数量 ({len(tile_predictions)}) 与 "
            f"tile_records 数量 ({len(tile_records)}) 不一致"
        )

    if iou_thresholds is None:
        iou_thresholds = DEFAULT_IOU_THRESHOLDS

    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"图像尺寸必须 > 0: {image_width}x{image_height}")

    # ---------- 第一步：收集所有框到全局坐标系 ----------
    all_boxes_global: List[List[float]] = []
    all_scores: List[float] = []
    all_labels: List[int] = []

    for prediction, tile in zip(tile_predictions, tile_records):
        if not prediction.boxes_xyxy:
            continue

        for box_local, score, label in zip(
            prediction.boxes_xyxy, prediction.scores, prediction.labels
        ):
            # 分数过滤
            if score < score_threshold:
                continue

            # 局部 → 全局
            box_global = tile_to_full(box_local, tile.x_offset, tile.y_offset)

            # 边界裁剪
            box_global = clip_bbox(box_global, image_width, image_height)

            # 裁剪后面积为 0 则丢弃
            gw = box_global[2] - box_global[0]
            gh = box_global[3] - box_global[1]
            if gw <= 0.0 or gh <= 0.0:
                continue

            all_boxes_global.append(box_global)
            all_scores.append(float(score))
            all_labels.append(int(label))

    if not all_boxes_global:
        return Prediction(
            image_id=parent_image_id,
            boxes_xyxy=[],
            scores=[],
            labels=[],
        )

    # ---------- 第二步：按细类分组 NMS ----------
    boxes_arr = np.array(all_boxes_global, dtype=np.float64)
    scores_arr = np.array(all_scores, dtype=np.float64)
    labels_arr = np.array(all_labels, dtype=int)

    unique_labels = np.unique(labels_arr)
    keep_indices: List[int] = []

    for label in unique_labels:
        mask = labels_arr == label
        idx = np.where(mask)[0]
        iou_thr = iou_thresholds.get(int(label), 0.50)
        kept = _nms_per_class(boxes_arr[idx], scores_arr[idx], iou_threshold=iou_thr)
        keep_indices.extend(idx[kept].tolist())

    if not keep_indices:
        return Prediction(
            image_id=parent_image_id,
            boxes_xyxy=[],
            scores=[],
            labels=[],
        )

    keep_indices.sort()

    logger.debug(
        "融合: %d 切片 → %d 全局框 → NMS 后 %d 框",
        len(tile_predictions),
        len(all_boxes_global),
        len(keep_indices),
    )

    return Prediction(
        image_id=parent_image_id,
        boxes_xyxy=boxes_arr[keep_indices].tolist(),
        scores=scores_arr[keep_indices].tolist(),
        labels=labels_arr[keep_indices].tolist(),
    )
