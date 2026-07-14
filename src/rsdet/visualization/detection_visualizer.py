"""检测结果可视化。

TODO: 基线模型确定后实现可视化功能。
"""

from typing import List

from rsdet.contracts import Prediction


def draw_predictions(
    image,
    predictions: Prediction,
    class_names: List[str],
    score_threshold: float = 0.3,
):
    """在图像上绘制检测框。

    Args:
        image: PIL Image 或 numpy 数组。
        predictions: Prediction 实例。
        class_names: 类别名列表。
        score_threshold: 低于此阈值的框不绘制。

    TODO: 实现绘制逻辑。
    """
    raise NotImplementedError("draw_predictions 尚未实现")
