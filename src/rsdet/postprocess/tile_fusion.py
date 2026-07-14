"""跨切片融合。

TODO: 实现坐标恢复后的相邻切片框合并逻辑。
"""

from typing import List

from rsdet.contracts import Prediction


def fuse_tile_predictions(tile_predictions: List[Prediction]) -> List[Prediction]:
    """将多个切片预测合并为原图预测。

    Args:
        tile_predictions: 各切片的 Prediction 列表。

    Returns:
        融合后的单张原图 Prediction。

    TODO: 实现 NMS 合并、边界裁剪、重复去除。
    """
    raise NotImplementedError("tile_fusion 尚未实现")
