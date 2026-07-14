"""非极大值抑制（NMS）。

当前提供基础接口，后续扩展类别感知 NMS 等。
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


def nms(boxes: List[List[float]], scores: List[float], iou_threshold: float) -> List[int]:
    """基础 NMS，返回保留的索引列表。

    Args:
        boxes: [[x1, y1, x2, y2], ...] xyxy 格式。
        scores: 每框的置信度。
        iou_threshold: IoU 阈值，高于此值的低分框被抑制。

    Returns:
        保留框的索引列表。

    TODO: 实现高效向量化版本。当前为基础实现。
    """
    raise NotImplementedError("NMS 尚未实现，待确定后处理策略")


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """计算两个 xyxy bbox 的 IoU。"""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_w = max(0.0, xb - xa)
    inter_h = max(0.0, yb - ya)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0
