"""为少样本细粒度分支生成目标框裁剪。

裁剪仅在内存中产生，不修改或复制官方原始数据。训练/验证成员严格来自冻结的
split manifest，避免目标裁剪绕过图像级防泄漏划分。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from rsdet.data.imbalance import load_split_manifest
from rsdet.data.xh_dataset import FINE_NAMES, _read_yolo_label

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[no-redef]
        pass


@dataclass(frozen=True)
class ObjectCropRecord:
    """一个可追溯的 GT 目标裁剪索引。"""

    image_id: int
    image_path: Path
    box_xyxy: tuple[float, float, float, float]
    label: int


def as_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image if image.mode == "RGB" else image.convert("RGB")
    if isinstance(image, (str, Path)):
        with Image.open(image) as source:
            return source.convert("RGB")
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] not in {3, 4}:
        raise ValueError("目标裁剪输入必须是 RGB/RGBA HxWxC 图像")
    if array.dtype != np.uint8:
        if np.issubdtype(array.dtype, np.floating) and array.max(initial=0.0) <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array).convert("RGB")


def crop_and_resize(
    image: Any,
    box_xyxy: Sequence[float],
    *,
    output_size: int = 128,
    context_ratio: float = 0.15,
    fill: tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """带上下文扩框、方形 letterbox 后缩放为 RGB uint8 裁剪。"""
    if len(box_xyxy) != 4:
        raise ValueError("box_xyxy 必须包含 4 个数值")
    if output_size <= 0:
        raise ValueError("output_size 必须 > 0")
    if context_ratio < 0.0:
        raise ValueError("context_ratio 不能为负")

    source = as_rgb_image(image)
    width, height = source.size
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"非法目标框: {list(box_xyxy)}")
    box_width = x2 - x1
    box_height = y2 - y1
    x1 = max(0.0, x1 - box_width * context_ratio)
    y1 = max(0.0, y1 - box_height * context_ratio)
    x2 = min(float(width), x2 + box_width * context_ratio)
    y2 = min(float(height), y2 + box_height * context_ratio)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("目标框裁剪后为空")

    crop = source.crop((int(np.floor(x1)), int(np.floor(y1)), int(np.ceil(x2)), int(np.ceil(y2))))
    side = max(crop.size)
    canvas = Image.new("RGB", (side, side), fill)
    canvas.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
    resized = canvas.resize((output_size, output_size), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8).copy()


def crop_to_tensor(crop: np.ndarray):
    """RGB uint8 裁剪转为 [-1, 1] 的 CHW float32 Tensor。"""
    if not _TORCH_AVAILABLE:
        raise ImportError("torch 未安装，无法生成目标裁剪 Tensor")
    array = np.ascontiguousarray(crop.transpose(2, 0, 1))
    return torch.from_numpy(array).to(dtype=torch.float32).div_(127.5).sub_(1.0)


class ObjectCropDataset(Dataset):
    """冻结 split 中逐 GT 框的细粒度识别数据集。"""

    def __init__(
        self,
        data_root: str | Path,
        manifest_path: str | Path,
        split: str,
        *,
        output_size: int = 128,
        context_ratio: float = 0.15,
        box_jitter: float = 0.08,
        augment: bool = False,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch 未安装，无法使用 ObjectCropDataset")
        if split not in {"train", "val"}:
            raise ValueError("split 必须是 train 或 val")
        self.output_size = int(output_size)
        self.context_ratio = float(context_ratio)
        self.box_jitter = float(box_jitter)
        if not 0.0 <= self.box_jitter <= 0.5:
            raise ValueError("box_jitter 必须在 [0, 0.5]")
        self.augment = bool(augment)
        samples = load_split_manifest(manifest_path, data_root, num_classes=len(FINE_NAMES))
        self.records: tuple[ObjectCropRecord, ...] = self._build_records(
            [sample for sample in samples if sample.split == split]
        )
        if not self.records:
            raise ValueError(f"{split} split 中没有目标框")
        self.class_counts = np.bincount(
            [record.label for record in self.records], minlength=len(FINE_NAMES)
        ).astype(np.int64)

    @staticmethod
    def _build_records(samples: Sequence[Any]) -> tuple[ObjectCropRecord, ...]:
        records: list[ObjectCropRecord] = []
        for sample in samples:
            with Image.open(sample.image_path) as source:
                width, height = source.size
            annotations = _read_yolo_label(
                sample.label_path,
                width,
                height,
                clamp_boxes=True,
            )
            for box, label in zip(annotations["boxes"], annotations["labels"]):
                records.append(
                    ObjectCropRecord(
                        image_id=sample.image_id,
                        image_path=sample.image_path,
                        box_xyxy=tuple(float(value) for value in box),
                        label=int(label),
                    )
                )
        return tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def _augment(self, crop: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            crop = np.flip(crop, axis=1)
        if random.random() < 0.5:
            crop = np.flip(crop, axis=0)
        if random.random() < 0.75:
            crop = np.rot90(crop, random.randrange(4), axes=(0, 1))
        if random.random() < 0.5:
            gain = random.uniform(0.85, 1.15)
            bias = random.uniform(-8.0, 8.0)
            crop = np.clip(crop.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(crop)

    def __getitem__(self, index: int):
        record = self.records[index]
        box = record.box_xyxy
        if self.augment and self.box_jitter > 0.0:
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            shift_x = random.uniform(-self.box_jitter, self.box_jitter) * width
            shift_y = random.uniform(-self.box_jitter, self.box_jitter) * height
            scale = random.uniform(1.0 - self.box_jitter, 1.0 + self.box_jitter)
            center_x = (x1 + x2) / 2.0 + shift_x
            center_y = (y1 + y2) / 2.0 + shift_y
            half_width = width * scale / 2.0
            half_height = height * scale / 2.0
            box = (
                center_x - half_width,
                center_y - half_height,
                center_x + half_width,
                center_y + half_height,
            )
        crop = crop_and_resize(
            record.image_path,
            box,
            output_size=self.output_size,
            context_ratio=self.context_ratio,
        )
        if self.augment:
            crop = self._augment(crop)
        return crop_to_tensor(crop), record.label

    def sampling_weights(self, *, power: float = 0.5) -> list[float]:
        """返回温和逆频率实例权重；power=0.5 为平方根逆频率。"""
        if not 0.0 <= power <= 1.0:
            raise ValueError("power 必须在 [0, 1]")
        counts = np.maximum(self.class_counts, 1)
        class_weights = np.power(1.0 / counts.astype(np.float64), power)
        return [float(class_weights[record.label]) for record in self.records]


__all__ = [
    "ObjectCropDataset",
    "ObjectCropRecord",
    "as_rgb_image",
    "crop_and_resize",
    "crop_to_tensor",
]
