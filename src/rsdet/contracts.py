"""稳定的数据契约定义。所有模块间数据交换必须使用此结构。

内部统一 bbox 格式：xyxy，像素坐标。
COCO 导出时转换为 xywh。
禁止在模块间混用 normalized 和 absolute 坐标。
"""

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class ImageRecord:
    """单张图像记录。

    Attributes:
        image_id: 稳定唯一图像 ID，禁止跨数据集重用。
        file_path: 图像文件路径。
        width: 图像宽度（像素）。
        height: 图像高度（像素）。
        metadata: 额外元信息。
    """

    image_id: int
    file_path: str
    width: int
    height: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationRecord:
    """单条标注记录。

    Attributes:
        annotation_id: 标注唯一 ID。
        image_id: 所属图像 ID，对应 ImageRecord.image_id。
        category_id: 类别 ID（整数）。
        bbox_xyxy: [x1, y1, x2, y2]，像素坐标。
        iscrowd: COCO iscrowd 标记。
        ignore: 是否在评估中忽略。
    """

    annotation_id: int
    image_id: int
    category_id: int
    bbox_xyxy: list  # [x1, y1, x2, y2]
    iscrowd: bool = False
    ignore: bool = False


@dataclass
class InferenceSample:
    """交给模型适配器的最小推理输入。

    ``image`` 的具体类型由模型适配器决定，可以是 numpy 数组、PIL 图像或
    框架张量。公共层只固定可追溯的图像 ID 和输入尺寸，避免强迫不同模型
    使用同一种预处理实现。

    Attributes:
        image_id: 稳定图像 ID。模型输出必须原样返回该 ID。
        image: 当前输入图像或 tile 的像素数据。
        width: 当前输入图像或 tile 的宽度。
        height: 当前输入图像或 tile 的高度。
        metadata: 模型或大图流水线需要的额外信息。
    """

    image_id: int
    image: Any
    width: int
    height: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """模型对单张图像的预测结果。

    Attributes:
        image_id: 对应 ImageRecord.image_id。
        boxes_xyxy: [N, 4] 数组，像素坐标 xyxy。
        scores: [N] 置信度数组。
        labels: [N] 类别标签数组。
    """

    image_id: int
    boxes_xyxy: list  # [[x1,y1,x2,y2], ...]
    scores: list  # [score, ...]
    labels: list  # [category_id, ...]


@dataclass
class TileRecord:
    """大图切片记录。

    Attributes:
        tile_id: 切片唯一 ID。
        parent_image_id: 原图 image_id。
        x_offset: 切片左上角在原图中的 x 偏移（像素）。
        y_offset: 切片左上角在原图中的 y 偏移（像素）。
        width: 切片宽度（像素）。
        height: 切片高度（像素）。
    """

    tile_id: int
    parent_image_id: int
    x_offset: int
    y_offset: int
    width: int
    height: int
