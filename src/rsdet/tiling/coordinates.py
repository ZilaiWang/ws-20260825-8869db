"""坐标转换工具。

内部统一使用 xyxy 像素坐标。所有函数均进行输入合法性校验。
"""

import math
from collections.abc import Sequence


def _box_values(box: Sequence[float]) -> list[float]:
    """校验 bbox 长度和数值。"""
    if len(box) != 4:
        raise ValueError(f"bbox 必须包含 4 个数值: {box}")
    values = [float(value) for value in box]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"bbox 包含非有限数值: {box}")
    return values


def xyxy_to_xywh(box: Sequence[float]) -> list[float]:
    """[x1, y1, x2, y2] → [x, y, w, h]。

    Raises:
        ValueError: x2 < x1 或 y2 < y1。
    """
    x1, y1, x2, y2 = _box_values(box)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"非法 bbox: {box}")
    return [x1, y1, x2 - x1, y2 - y1]


def xywh_to_xyxy(box: Sequence[float]) -> list[float]:
    """[x, y, w, h] → [x1, y1, x2, y2]。

    Raises:
        ValueError: w < 0 或 h < 0。
    """
    x, y, w, h = _box_values(box)
    if w < 0 or h < 0:
        raise ValueError(f"非法 bbox: {box}")
    return [x, y, x + w, y + h]


def tile_to_full(
    box_tile: Sequence[float],
    x_offset: int,
    y_offset: int,
) -> list[float]:
    """tile 内坐标 → 原图坐标。

    Args:
        box_tile: tile 内的 [x1, y1, x2, y2]。
        x_offset: tile 左上角在原图的 x 偏移。
        y_offset: tile 左上角在原图的 y 偏移。

    Returns:
        原图像素坐标 [x1, y1, x2, y2]。
    """
    x1, y1, x2, y2 = _box_values(box_tile)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"非法 xyxy bbox: {box_tile}")
    return [x1 + x_offset, y1 + y_offset, x2 + x_offset, y2 + y_offset]


def full_to_tile(
    box_full: Sequence[float],
    x_offset: int,
    y_offset: int,
) -> list[float]:
    """原图坐标 → tile 内坐标。

    Args:
        box_full: 原图中的 [x1, y1, x2, y2]。
        x_offset: tile 左上角在原图的 x 偏移。
        y_offset: tile 左上角在原图的 y 偏移。

    Returns:
        tile 内坐标 [x1, y1, x2, y2]。
    """
    x1, y1, x2, y2 = _box_values(box_full)
    if x2 < x1 or y2 < y1:
        raise ValueError(f"非法 xyxy bbox: {box_full}")
    return [x1 - x_offset, y1 - y_offset, x2 - x_offset, y2 - y_offset]


def clip_bbox(
    box: Sequence[float],
    img_width: int,
    img_height: int,
) -> list[float]:
    """将 bbox 裁剪到图像边界内。

    Args:
        box: [x1, y1, x2, y2]。
        img_width: 图像宽度。
        img_height: 图像高度。

    Returns:
        裁剪后的 bbox。完全在图像外时返回位于最近边界的零面积框。

    Raises:
        ValueError: 图像尺寸非法。
    """
    if img_width <= 0 or img_height <= 0:
        raise ValueError(f"图像尺寸必须 > 0: {img_width}x{img_height}")
    raw_x1, raw_y1, raw_x2, raw_y2 = _box_values(box)
    if raw_x2 < raw_x1 or raw_y2 < raw_y1:
        raise ValueError(f"非法 xyxy bbox: {box}")
    x1 = max(0.0, min(raw_x1, img_width))
    y1 = max(0.0, min(raw_y1, img_height))
    x2 = max(0.0, min(raw_x2, img_width))
    y2 = max(0.0, min(raw_y2, img_height))
    return [x1, y1, x2, y2]
