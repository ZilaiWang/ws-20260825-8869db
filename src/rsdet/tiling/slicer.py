"""大图滑窗切片器。

模型无关的切片坐标生成。当前不要求高性能图像 I/O。
"""

import logging
from typing import List

from rsdet.contracts import TileRecord

logger = logging.getLogger(__name__)


def generate_tiles(
    image_width: int,
    image_height: int,
    tile_size: int,
    overlap: int,
) -> List[TileRecord]:
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
    tiles: List[TileRecord] = []
    tile_id = 0

    y = 0
    while y < image_height:
        x = 0
        actual_h = min(tile_size, image_height - y)
        while x < image_width:
            actual_w = min(tile_size, image_width - x)
            tiles.append(TileRecord(
                tile_id=tile_id,
                parent_image_id=0,  # 调用方负责填充
                x_offset=x,
                y_offset=y,
                width=actual_w,
                height=actual_h,
            ))
            tile_id += 1
            x += stride
            if x >= image_width:
                break
        y += stride
        if y >= image_height:
            break

    logger.debug(f"生成 {len(tiles)} 个切片: {image_width}x{image_height}, "
                 f"tile={tile_size}, overlap={overlap}")
    return tiles
