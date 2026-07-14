"""XH-202625 的轻量 PyTorch 检测数据适配器，不依赖 torchvision。

需要 PyTorch，不属于 requirements.txt 的轻量依赖范围。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

    class Dataset:  # type: ignore[no-redef]
        pass

from rsdet.data.xh_dataset import XHDataset

TorchTransform = Callable[
    [torch.Tensor, dict[str, Any]],
    tuple[torch.Tensor, dict[str, Any]],
] if _TORCH_AVAILABLE else Callable


class TorchDetectionDataset(Dataset):
    """返回 ``(image, target)``，可直接交给 PyTorch DataLoader。

    image 为 float32 ``[3, H, W]``，取值范围 ``[0, 1]``。
    target 中 boxes 为原图像素 xyxy，labels 为 0--24。
    """

    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        *,
        transforms: TorchTransform | None = None,
        require_labels: bool = True,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise ImportError("torch 未安装，无法使用 TorchDetectionDataset")
        self.core = XHDataset(
            data_root,
            split=split,
            load_images=True,
            require_labels=require_labels,
            clamp_boxes=True,
        )
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.core)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        sample = self.core[index]
        image_array = np.ascontiguousarray(sample["image"].transpose(2, 0, 1))
        image = torch.from_numpy(image_array).to(dtype=torch.float32).div_(255.0)

        source = sample["target"]
        target: dict[str, Any] = {
            "boxes": torch.from_numpy(source["boxes"].copy()).to(dtype=torch.float32),
            "labels": torch.from_numpy(source["labels"].copy()).to(dtype=torch.int64),
            "area": torch.from_numpy(source["area"].copy()).to(dtype=torch.float32),
            "iscrowd": torch.from_numpy(source["iscrowd"].copy()).to(dtype=torch.int64),
            "image_id": torch.tensor(source["image_id"], dtype=torch.int64),
            "orig_size": torch.from_numpy(source["orig_size"].copy()).to(dtype=torch.int64),
        }
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target


__all__ = ["TorchDetectionDataset"]
