"""XH-202625 数据集统一加载器。

唯一内部约定：
- 图像：RGB、numpy.uint8、形状 [H, W, 3]
- 框：原图像素坐标 xyxy、numpy.float32、形状 [N, 4]
- 标签：官方细类 ID 0--24、numpy.int64、形状 [N]
- image_id：当前 split 内按文件主名排序后从 1 开始的稳定整数
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
from PIL import Image

FINE_NAMES: tuple[str, ...] = (
    "HM",
    "LQS",
    "QHS",
    "MS",
    "A1_SU-35",
    "A2_C-130",
    "A3_C-17",
    "A4_C-5",
    "A5_F-16",
    "A6_TU-160",
    "A7_E-3",
    "A8_B-52",
    "A9_P-3C",
    "A10_B-1B",
    "A11_E-8",
    "A12_TU-22",
    "A13_F-15",
    "A14_KC-135",
    "A15_F-22",
    "A16_FA-18",
    "A17_TU-95",
    "A18_KC-10",
    "A19_SU-34",
    "A20_SU-24",
    "FSC",
)

FINE_NAMES_CN: tuple[str, ...] = (
    "航母",
    "两栖舰",
    "驱护舰",
    "民船",
    "SU-35",
    "C-130",
    "C-17",
    "C-5",
    "F-16",
    "TU-160",
    "E-3",
    "B-52",
    "P-3C",
    "B-1B",
    "E-8",
    "TU-22",
    "F-15",
    "KC-135",
    "F-22",
    "FA-18",
    "TU-95",
    "KC-10",
    "SU-34",
    "SU-24",
    "发射车",
)

COARSE_NAMES: tuple[str, ...] = ("ship", "aircraft", "vehicle")
EVAL_IOU: dict[str, float] = {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35}
MODALITY: dict[str, str] = {
    "ship": "panchromatic",
    "aircraft": "rgb",
    "vehicle": "rgb",
}
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})

Sample = dict[str, Any]
Transform = Callable[[Sample], Sample]


def coarse_name(class_id: int) -> str:
    """把官方细类 ID 映射为评测大类。"""
    if 0 <= class_id <= 3:
        return "ship"
    if 4 <= class_id <= 23:
        return "aircraft"
    if class_id == 24:
        return "vehicle"
    raise ValueError(f"非法 class_id: {class_id}")


def yolo_to_xyxy(
    cx: float,
    cy: float,
    box_width: float,
    box_height: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """归一化 YOLO cxcywh 转原图像素 xyxy。"""
    x1 = (cx - box_width / 2.0) * image_width
    y1 = (cy - box_height / 2.0) * image_height
    x2 = (cx + box_width / 2.0) * image_width
    y2 = (cy + box_height / 2.0) * image_height
    return x1, y1, x2, y2


def xyxy_to_coco(box: Sequence[float]) -> list[float]:
    """像素 xyxy 转 COCO 像素 xywh。"""
    if len(box) != 4:
        raise ValueError(f"xyxy 必须有 4 个值，实际为 {len(box)}")
    x1, y1, x2, y2 = map(float, box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"非法 xyxy 框: {box}")
    return [x1, y1, x2 - x1, y2 - y1]


@dataclass(frozen=True)
class SampleRef:
    """不加载像素时即可确定的样本索引。"""

    image_id: int
    split: str
    stem: str
    image_path: Path
    label_path: Path | None


def _read_yolo_label(
    label_path: Path | None,
    image_width: int,
    image_height: int,
    clamp_boxes: bool,
) -> dict[str, np.ndarray]:
    if label_path is None:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
        }

    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
        }

    boxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: 标签行必须为 5 列")

        class_text, cx_text, cy_text, width_text, height_text = parts
        try:
            class_id = int(class_text)
            cx, cy, box_width, box_height = map(float, (cx_text, cy_text, width_text, height_text))
        except ValueError as error:
            raise ValueError(f"{label_path}:{line_number}: 非法数值") from error

        if not 0 <= class_id < len(FINE_NAMES):
            raise ValueError(f"{label_path}:{line_number}: 非法 class_id={class_id}")
        if not np.isfinite([cx, cy, box_width, box_height]).all():
            raise ValueError(f"{label_path}:{line_number}: 坐标必须为有限数")
        if box_width <= 0.0 or box_height <= 0.0:
            raise ValueError(f"{label_path}:{line_number}: 框宽高必须大于 0")

        box = np.asarray(
            yolo_to_xyxy(cx, cy, box_width, box_height, image_width, image_height),
            dtype=np.float64,
        )
        if clamp_boxes:
            box[[0, 2]] = np.clip(box[[0, 2]], 0.0, float(image_width))
            box[[1, 3]] = np.clip(box[[1, 3]], 0.0, float(image_height))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"{label_path}:{line_number}: 转换后得到空框")

        boxes.append(tuple(float(value) for value in box))
        labels.append(class_id)

    return {
        "boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        "labels": np.asarray(labels, dtype=np.int64),
    }


class XHDataset(Sequence[Sample]):
    """框架无关的统一数据集。

    Args:
        data_root: 含 ``images/``、``labels/`` 和 ``dataset.yaml`` 的目录。
        split: ``train`` 或 ``val``。
        load_images: False 时不返回像素，适合统计和 COCO 导出。
        require_labels: False 时允许无标签图像，适合正式测试集。
        clamp_boxes: 将浮点框限制到图像边界，模型输入建议保持 True。
        transform: 接收并返回完整 sample 字典的可选变换。

    返回格式见本文件顶部说明及 ``code/README.md``。
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        *,
        load_images: bool = True,
        require_labels: bool = True,
        clamp_boxes: bool = True,
        transform: Transform | None = None,
    ) -> None:
        self.data_root = Path(data_root).expanduser().resolve()
        self.split = split
        self.load_images = load_images
        self.require_labels = require_labels
        self.clamp_boxes = clamp_boxes
        self.transform = transform
        self.image_dir = self.data_root / "images" / split
        self.label_dir = self.data_root / "labels" / split

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"图像目录不存在: {self.image_dir}")
        if require_labels and not self.label_dir.is_dir():
            raise FileNotFoundError(f"标签目录不存在: {self.label_dir}")

        self.refs = self._build_index()

    def _build_index(self) -> tuple[SampleRef, ...]:
        image_by_stem: dict[str, Path] = {}
        for path in sorted(self.image_dir.iterdir(), key=lambda item: item.name):
            # Finder may leave AppleDouble resource-fork sidecars (``._name.jpg``)
            # when a dataset is copied from macOS to Linux.  They retain the image
            # suffix but are metadata, not decodable images or dataset samples.
            if path.name.startswith("._"):
                continue
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if path.stem in image_by_stem:
                raise ValueError(f"图像主名重复: {path.stem}")
            image_by_stem[path.stem] = path

        if not image_by_stem:
            return ()

        label_by_stem: dict[str, Path] = {}
        if self.label_dir.is_dir():
            for path in sorted(self.label_dir.glob("*.txt"), key=lambda item: item.name):
                if path.name.startswith("._"):
                    continue
                if path.stem in label_by_stem:
                    raise ValueError(f"标签主名重复: {path.stem}")
                label_by_stem[path.stem] = path

        if self.require_labels:
            missing = sorted(set(image_by_stem) - set(label_by_stem))
            orphan = sorted(set(label_by_stem) - set(image_by_stem))
            if missing or orphan:
                raise ValueError(f"图像标签配对失败: 缺失标签={missing[:5]}，无图标签={orphan[:5]}")

        refs: list[SampleRef] = []
        for image_id, stem in enumerate(sorted(image_by_stem), start=1):
            refs.append(
                SampleRef(
                    image_id=image_id,
                    split=self.split,
                    stem=stem,
                    image_path=image_by_stem[stem],
                    label_path=label_by_stem.get(stem),
                )
            )
        return tuple(refs)

    def __len__(self) -> int:
        return len(self.refs)

    def __getitem__(self, index: int) -> Sample:
        ref = self.refs[index]
        with Image.open(ref.image_path) as source:
            image_width, image_height = source.size
            file_format = source.format
            source_mode = source.mode
            image = (
                np.asarray(source.convert("RGB"), dtype=np.uint8).copy()
                if self.load_images
                else None
            )

        parsed = _read_yolo_label(
            ref.label_path,
            image_width,
            image_height,
            clamp_boxes=self.clamp_boxes,
        )
        boxes = parsed["boxes"]
        labels = parsed["labels"]
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        sample: Sample = {
            "image": image,
            "target": {
                "boxes": boxes,
                "labels": labels,
                "area": area.astype(np.float32, copy=False),
                "iscrowd": np.zeros(len(labels), dtype=np.int64),
                "image_id": ref.image_id,
                "orig_size": np.asarray([image_height, image_width], dtype=np.int64),
            },
            "meta": {
                "split": ref.split,
                "stem": ref.stem,
                "image_path": str(ref.image_path),
                "label_path": str(ref.label_path) if ref.label_path is not None else None,
                "width": image_width,
                "height": image_height,
                "file_format": file_format,
                "source_mode": source_mode,
            },
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __iter__(self) -> Iterator[Sample]:
        for index in range(len(self)):
            yield self[index]

    def class_counts(self) -> np.ndarray:
        """返回形状 [25] 的框数量；不加载图像像素。"""
        counts = np.zeros(len(FINE_NAMES), dtype=np.int64)
        for ref in self.refs:
            with Image.open(ref.image_path) as source:
                image_width, image_height = source.size
            labels = _read_yolo_label(
                ref.label_path,
                image_width,
                image_height,
                clamp_boxes=self.clamp_boxes,
            )["labels"]
            counts += np.bincount(labels, minlength=len(FINE_NAMES))
        return counts


def detection_collate(batch: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """变尺寸检测数据的 DataLoader collate_fn。"""
    images, targets = zip(*batch)
    return list(images), list(targets)


__all__ = [
    "COARSE_NAMES",
    "EVAL_IOU",
    "FINE_NAMES",
    "FINE_NAMES_CN",
    "MODALITY",
    "SampleRef",
    "XHDataset",
    "coarse_name",
    "detection_collate",
    "xyxy_to_coco",
    "yolo_to_xyxy",
]
