"""大图滑窗切片器。

模型无关的切片坐标生成。当前不要求高性能图像 I/O。
"""

import logging

from rsdet.contracts import TileRecord

logger = logging.getLogger(__name__)


def generate_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
) -> list[TileRecord]:
    """对大图生成滑窗切片坐标。

    保证：
    - 图像边缘正确处理。
    - 最后一行/列完整覆盖。
    - tile_size 大于图像时返回单张全图切片。
    - 不重复生成完全相同的 tile。

    Args:
        image_width: 原图宽度（像素）。
        image_height: 原图高度（像素）。
        tile_size: 切片边长（像素）。
        overlap: 相邻切片重叠量（像素）。

    Returns:
        TileRecord 列表。

    Raises:
        ValueError: 输入参数非法。
    """
    if tile_size <= 0:
        raise ValueError(f"tile_size 必须 > 0，当前: {tile_size}")
    if overlap < 0:
        raise ValueError(f"overlap 不能为负，当前: {overlap}")
    if overlap >= tile_size:
        raise ValueError(f"overlap ({overlap}) 必须小于 tile_size ({tile_size})")
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"图像尺寸必须 > 0: {image_width}x{image_height}")

    stride = tile_size - overlap
    x_starts = _axis_starts(image_width, tile_size, stride)
    y_starts = _axis_starts(image_height, tile_size, stride)
    tiles: list[TileRecord] = []
    tile_id = 0

    for y in y_starts:
        actual_h = min(tile_size, image_height - y)
        for x in x_starts:
            actual_w = min(tile_size, image_width - x)
            tiles.append(
                TileRecord(
                    tile_id=tile_id,
                    parent_image_id=0,  # 调用方负责填充
                    x_offset=x,
                    y_offset=y,
                    width=actual_w,
                    height=actual_h,
                )
            )
            tile_id += 1

    logger.debug(
        "生成 %d 个切片: %dx%d, tile=%d, overlap=%d",
        len(tiles),
        image_width,
        image_height,
        tile_size,
        overlap,
    )
    return tiles


def _axis_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """生成单轴起点，并把最后一个完整 tile 对齐到图像边缘。"""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts
