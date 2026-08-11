"""P0-3 对象 crop 分类数据契约。

本模块不强依赖 PyTorch，因而在无 GPU/无 torch 的本地环境也能校验
manifest 选样、防泄漏划分与像素裁剪。PyTorch DataLoader 只需求对象实现
``__len__`` 和 ``__getitem__``，不要求继承 torch Dataset。
"""

from __future__ import annotations

import csv
import math
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from rsdet.data.xh_dataset import FINE_NAMES, coarse_name

REQUIRED_CROP_COLUMNS = {
    "manifest_version",
    "crop_id",
    "annotation_uid",
    "source_image_id",
    "source_relative_path",
    "source_width",
    "source_height",
    "source_checksum_sha256",
    "class_id",
    "class_name",
    "major_class",
    "crop_policy",
    "crop_x0",
    "crop_y0",
    "crop_x1",
    "crop_y1",
    "fold",
    "leakage_group_id",
    "color_mode",
    "outside_policy",
    "resize_semantics",
}
ALLOWED_POLICIES = frozenset({"tight", "context_1p25", "jitter_light"})
ALLOWED_SPLITS = frozenset({"train", "val"})


@dataclass(frozen=True)
class CropRecord:
    """一条可渲染的 crop 样本。"""

    manifest_version: str
    crop_id: str
    annotation_uid: str
    source_image_id: str
    source_relative_path: str
    source_width: int
    source_height: int
    source_checksum_sha256: str
    class_id: int
    class_name: str
    major_class: str
    crop_policy: str
    crop_xyxy: tuple[float, float, float, float]
    fold: int
    leakage_group_id: str


def _finite_float(value: str, field: str, line_number: int) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"manifest 第 {line_number} 行 {field} 非法: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"manifest 第 {line_number} 行 {field} 必须是有限数")
    return result


def _safe_relative_path(value: str, line_number: int) -> str:
    normalized = value.strip().replace("\\", "/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"manifest 第 {line_number} 行含不安全路径: {value!r}")
    return normalized


def load_crop_records(
    manifest_path: str | Path,
    *,
    crop_policy: str,
    held_out_fold: int,
    split: str,
    task: str = "all25",
) -> list[CropRecord]:
    """从 P0-2 manifest 读取一个 policy 和一个交叉验证分区。

    ``train`` 使用 ``fold != held_out_fold``，``val`` 使用该 held-out fold。
    划分依据是 P0-2 已修复的 ``leakage_group_id``，不重新按单张图分配。
    """

    manifest_path = Path(manifest_path).expanduser().resolve()
    if crop_policy not in ALLOWED_POLICIES:
        raise ValueError(f"未知 crop_policy={crop_policy!r}")
    if held_out_fold not in {0, 1, 2}:
        raise ValueError("held_out_fold 必须是 0、1 或 2")
    if split not in ALLOWED_SPLITS:
        raise ValueError(f"split 必须是 {sorted(ALLOWED_SPLITS)}")
    if task not in {"all25", "aircraft20"}:
        raise ValueError("task 必须是 all25 或 aircraft20")

    records: list[CropRecord] = []
    seen_annotations: set[str] = set()
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        missing = sorted(REQUIRED_CROP_COLUMNS - set(reader.fieldnames or ()))
        if missing:
            raise ValueError(f"{manifest_path} 缺少必需列: {', '.join(missing)}")
        for line_number, row in enumerate(reader, start=2):
            if row["crop_policy"].strip() != crop_policy:
                continue
            fold = int(row["fold"])
            if fold not in {0, 1, 2}:
                raise ValueError(f"manifest 第 {line_number} 行 fold 非法: {fold}")
            if (split == "val") != (fold == held_out_fold):
                continue

            class_id = int(row["class_id"])
            if not 0 <= class_id < len(FINE_NAMES):
                raise ValueError(f"manifest 第 {line_number} 行 class_id 非法: {class_id}")
            if task == "aircraft20" and not 4 <= class_id <= 23:
                continue
            expected_name = FINE_NAMES[class_id]
            expected_major = coarse_name(class_id)
            if row["class_name"].strip() != expected_name:
                raise ValueError(f"manifest 第 {line_number} 行 class_name 与 ID 不一致")
            if row["major_class"].strip() != expected_major:
                raise ValueError(f"manifest 第 {line_number} 行 major_class 与 ID 不一致")
            if row["color_mode"].strip() != "RGB":
                raise ValueError("P0-3 首轮只支持 P0-2 约定的 RGB 渲染")
            if row["outside_policy"].strip() != "pad":
                raise ValueError("P0-3 首轮只支持 outside_policy=pad")
            if row["resize_semantics"].strip() != "direct_square_resize":
                raise ValueError("P0-3 首轮只支持 direct_square_resize")

            annotation_uid = row["annotation_uid"].strip()
            if not annotation_uid or annotation_uid in seen_annotations:
                raise ValueError(
                    f"{crop_policy}/{split}/fold{held_out_fold} 中 annotation_uid 为空或重复"
                )
            crop_id = row["crop_id"].strip()
            leakage_group_id = row["leakage_group_id"].strip()
            checksum = row["source_checksum_sha256"].strip().lower()
            if not crop_id or not leakage_group_id or len(checksum) != 64:
                raise ValueError(f"manifest 第 {line_number} 行缺少稳定 ID 或合法 checksum")

            crop_xyxy = tuple(
                _finite_float(row[field], field, line_number)
                for field in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")
            )
            if crop_xyxy[2] <= crop_xyxy[0] or crop_xyxy[3] <= crop_xyxy[1]:
                raise ValueError(f"manifest 第 {line_number} 行 crop 为空")
            if not math.isclose(
                crop_xyxy[2] - crop_xyxy[0],
                crop_xyxy[3] - crop_xyxy[1],
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(f"manifest 第 {line_number} 行 crop 不是方形")

            source_width = int(float(row["source_width"]))
            source_height = int(float(row["source_height"]))
            if source_width <= 0 or source_height <= 0:
                raise ValueError(f"manifest 第 {line_number} 行原图尺寸非法")
            records.append(
                CropRecord(
                    manifest_version=row["manifest_version"].strip(),
                    crop_id=crop_id,
                    annotation_uid=annotation_uid,
                    source_image_id=row["source_image_id"].strip(),
                    source_relative_path=_safe_relative_path(
                        row["source_relative_path"], line_number
                    ),
                    source_width=source_width,
                    source_height=source_height,
                    source_checksum_sha256=checksum,
                    class_id=class_id,
                    class_name=expected_name,
                    major_class=expected_major,
                    crop_policy=crop_policy,
                    crop_xyxy=(
                        crop_xyxy[0],
                        crop_xyxy[1],
                        crop_xyxy[2],
                        crop_xyxy[3],
                    ),
                    fold=fold,
                    leakage_group_id=leakage_group_id,
                )
            )
            seen_annotations.add(annotation_uid)

    if not records:
        raise ValueError(
            f"{manifest_path} 未选出 {crop_policy}/{split}/fold{held_out_fold}/{task} 样本"
        )
    records.sort(key=lambda item: (item.annotation_uid, item.crop_id))
    return records


def validate_fold_isolation(
    train_records: Sequence[CropRecord], val_records: Sequence[CropRecord]
) -> None:
    """校验训练/验证间的对象、图像和泄漏组三层隔离。"""

    axes = (
        (
            "annotation_uid",
            {item.annotation_uid for item in train_records},
            {item.annotation_uid for item in val_records},
        ),
        (
            "source_image_id",
            {item.source_image_id for item in train_records},
            {item.source_image_id for item in val_records},
        ),
        (
            "leakage_group_id",
            {item.leakage_group_id for item in train_records},
            {item.leakage_group_id for item in val_records},
        ),
    )
    for name, train_values, val_values in axes:
        overlap = train_values & val_values
        if overlap:
            examples = sorted(overlap)[:3]
            raise ValueError(f"训练/验证 {name} 泄漏，例: {examples}")


def select_deterministic_subset(
    records: Sequence[CropRecord], max_samples: int | None
) -> list[CropRecord]:
    """用于 smoke test 的确定性、尽量分层子集。"""

    if max_samples is None or max_samples >= len(records):
        return list(records)
    if max_samples <= 0:
        raise ValueError("max_samples 必须大于 0")
    by_class: dict[int, list[CropRecord]] = {}
    for record in records:
        by_class.setdefault(record.class_id, []).append(record)
    selected: list[CropRecord] = []
    class_ids = sorted(by_class)
    cursor = 0
    while len(selected) < max_samples:
        made_progress = False
        for class_id in class_ids:
            values = by_class[class_id]
            if cursor < len(values):
                selected.append(values[cursor])
                made_progress = True
                if len(selected) == max_samples:
                    break
        if not made_progress:
            break
        cursor += 1
    return sorted(selected, key=lambda item: (item.annotation_uid, item.crop_id))


def render_crop(
    image: Image.Image,
    crop_xyxy: Sequence[float],
    resolution: int,
) -> Image.Image:
    """按 P0-2 连续浮点 EXTENT 约定直接渲染到方形输入。"""

    if resolution <= 0:
        raise ValueError("resolution 必须大于 0")
    if len(crop_xyxy) != 4:
        raise ValueError("crop_xyxy 必须有 4 个数")
    box = tuple(float(value) for value in crop_xyxy)
    if not all(math.isfinite(value) for value in box) or box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"非法 crop_xyxy={crop_xyxy}")
    rgb = image.convert("RGB")
    return rgb.transform(
        (resolution, resolution),
        Image.Transform.EXTENT,
        box,
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


class CropClassificationDataset:
    """运行时从原图渲染 crop，不物化重复图片。"""

    def __init__(
        self,
        records: Sequence[CropRecord],
        data_root: str | Path,
        resolution: int,
        *,
        transform: Callable[[Image.Image], Any] | None = None,
        image_cache_size: int = 8,
        verify_dimensions: bool = True,
    ) -> None:
        if not records:
            raise ValueError("records 不能为空")
        if resolution not in {224, 336}:
            raise ValueError("P0-3 首轮 resolution 只允许 224 或 336")
        if image_cache_size < 0:
            raise ValueError("image_cache_size 不得为负")
        self.records = tuple(records)
        self.data_root = Path(data_root).expanduser().resolve()
        self.resolution = resolution
        self.transform = transform
        self.image_cache_size = image_cache_size
        self.verify_dimensions = verify_dimensions
        self._cache: OrderedDict[Path, Image.Image] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_source(self, record: CropRecord) -> Path:
        path = (self.data_root / record.source_relative_path).resolve()
        try:
            path.relative_to(self.data_root)
        except ValueError as error:
            raise ValueError(f"原图路径逃逸 data_root: {path}") from error
        if not path.is_file():
            raise FileNotFoundError(f"原图不存在: {path}")
        return path

    def _open_image(self, path: Path, record: CropRecord) -> Image.Image:
        cached = self._cache.pop(path, None)
        if cached is not None:
            self._cache[path] = cached
            return cached
        with Image.open(path) as source:
            image = source.convert("RGB")
        if self.verify_dimensions and image.size != (record.source_width, record.source_height):
            raise ValueError(
                f"{path} 尺寸 {image.size} 与 manifest "
                f"{(record.source_width, record.source_height)} 不一致"
            )
        if self.image_cache_size:
            self._cache[path] = image
            while len(self._cache) > self.image_cache_size:
                self._cache.popitem(last=False)
        return image

    def __getitem__(self, index: int) -> tuple[Any, int, int]:
        record = self.records[index]
        source = self._open_image(self._resolve_source(record), record)
        crop = render_crop(source, record.crop_xyxy, self.resolution)
        value = self.transform(crop) if self.transform is not None else crop
        return value, record.class_id, index


def class_counts(records: Iterable[CropRecord]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for record in records:
        counts[record.class_id] = counts.get(record.class_id, 0) + 1
    return dict(sorted(counts.items()))
