"""P04 canonical crop、D4 视图和无 fold 对象集。"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image

from rsdet.data.crop_classification import (
    CropRecord,
    load_crop_records,
    render_crop,
)

D4_VIEW_IDS = (
    "r0",
    "r90",
    "r180",
    "r270",
    "flip_r0",
    "flip_r90",
    "flip_r180",
    "flip_r270",
)


def apply_d4_view(image: Image.Image, view_id: str) -> Image.Image:
    """对已渲染的 canonical crop 施加唯一的 D4 变换。

    ``flip`` 固定表示先水平翻转，再逆时针旋转。这 8 个变换与
    P03 的随机 1/4 旋转 + H/V flip 在分布上等价，但离线缓存中
    每个视图只保存一次。
    """

    if view_id not in D4_VIEW_IDS:
        raise ValueError(f"未知 D4 view_id={view_id!r}")
    value = image
    if view_id.startswith("flip_"):
        value = value.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        rotation = view_id.removeprefix("flip_")
    else:
        rotation = view_id
    transpose = {
        "r0": None,
        "r90": Image.Transpose.ROTATE_90,
        "r180": Image.Transpose.ROTATE_180,
        "r270": Image.Transpose.ROTATE_270,
    }[rotation]
    return value if transpose is None else value.transpose(transpose)


def canonical_image_sha256(image: Image.Image) -> str:
    """对 canonical RGB 像素与尺寸做稳定哈希。"""

    rgb = image.convert("RGB")
    digest = hashlib.sha256()
    digest.update(f"RGB|{rgb.width}|{rgb.height}|".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def preprocessing_fingerprint(payload: dict[str, object]) -> str:
    """对预处理合同生成可重现 fingerprint。"""

    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_all_crop_records(
    manifest_path: str | Path,
    *,
    crop_policy: str = "tight",
    task: str = "all25",
) -> list[CropRecord]:
    """读取一个 policy 的全部对象，不把当前 fold 写入特征语义。"""

    records: list[CropRecord] = []
    for fold in (0, 1, 2):
        records.extend(
            load_crop_records(
                manifest_path,
                crop_policy=crop_policy,
                held_out_fold=fold,
                split="val",
                task=task,
            )
        )
    records.sort(key=lambda item: (item.annotation_uid, item.crop_id))
    ids = [item.annotation_uid for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("全量 crop records 中 annotation_uid 重复")
    return records


def render_canonical_crop(
    record: CropRecord,
    data_root: str | Path,
    *,
    resolution: int = 224,
) -> Image.Image:
    """按 P03 的 EXTENT+bicubic 合同渲染单个 canonical crop。"""

    root = Path(data_root).expanduser().resolve()
    source_path = (root / record.source_relative_path).resolve()
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"原图路径逃逸 data_root: {source_path}") from error
    if not source_path.is_file():
        raise FileNotFoundError(f"原图不存在: {source_path}")
    with Image.open(source_path) as source:
        if source.size != (record.source_width, record.source_height):
            raise ValueError(
                f"{source_path} 尺寸 {source.size} 与 manifest "
                f"{(record.source_width, record.source_height)} 不一致"
            )
        return render_crop(source, record.crop_xyxy, resolution)


def load_annotation_allowlist(path: str | Path | None) -> set[str] | None:
    """读取每行一个 annotation_uid 的可选子集。"""

    if path is None:
        return None
    values = {
        line.strip()
        for line in Path(path).expanduser().resolve().read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError("annotation allowlist 为空")
    return values


def filter_records(
    records: Sequence[CropRecord],
    *,
    allowlist: set[str] | None = None,
    max_samples: int | None = None,
) -> list[CropRecord]:
    """按 allowlist 和类别轮询的确定性子集筛选。"""

    selected = [item for item in records if allowlist is None or item.annotation_uid in allowlist]
    if allowlist is not None:
        missing = allowlist - {item.annotation_uid for item in selected}
        if missing:
            raise ValueError(f"allowlist 有 {len(missing)} 个 UID 不在所选 policy 中")
    if max_samples is None or len(selected) <= max_samples:
        return selected
    if max_samples <= 0:
        raise ValueError("max_samples 必须大于 0")
    by_class: dict[int, list[CropRecord]] = {}
    for record in selected:
        by_class.setdefault(record.class_id, []).append(record)
    result: list[CropRecord] = []
    cursor = 0
    while len(result) < max_samples:
        progressed = False
        for class_id in sorted(by_class):
            values = by_class[class_id]
            if cursor < len(values):
                result.append(values[cursor])
                progressed = True
                if len(result) == max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(result, key=lambda item: (item.annotation_uid, item.crop_id))


def select_calibration_uids(
    manifest_path: str | Path,
    *,
    crop_policy: str = "tight",
    count: int = 256,
) -> list[str]:
    """为 P04-3 生成兼顾类别和风险因子的确定性子集。"""

    if count <= 0:
        raise ValueError("count 必须大于 0")
    rows: list[dict[str, str]] = []
    with (
        Path(manifest_path)
        .expanduser()
        .resolve()
        .open("r", encoding="utf-8-sig", newline="") as file
    ):
        reader = csv.DictReader(file)
        required = {
            "annotation_uid",
            "class_id",
            "crop_policy",
            "major_class",
            "source_edge_risk",
            "padding_fraction",
            "gt_coverage_fraction",
            "gt_short_edge",
        }
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"calibration 选样缺少列: {sorted(missing)}")
        for row in reader:
            if row["crop_policy"].strip() == crop_policy:
                rows.append(row)
    if count > len(rows):
        raise ValueError(f"请求 {count} 个 calibration 对象，但只有 {len(rows)} 个")

    def risk_bucket(row: dict[str, str]) -> tuple[int, int, int, int]:
        edge = int(row["source_edge_risk"].strip().lower() in {"1", "true", "yes"})
        padding = int(float(row["padding_fraction"]) > 0)
        low_coverage = int(float(row["gt_coverage_fraction"]) < 0.9)
        small = int(float(row["gt_short_edge"]) < 48)
        return edge, padding, low_coverage, small

    buckets: dict[tuple[int, tuple[int, int, int, int]], list[str]] = {}
    for row in rows:
        key = (int(row["class_id"]), risk_bucket(row))
        buckets.setdefault(key, []).append(row["annotation_uid"].strip())
    for values in buckets.values():
        values.sort()

    chosen: list[str] = []
    cursor = 0
    ordered_keys = sorted(buckets, key=lambda item: (item[0], -sum(item[1]), item[1]))
    while len(chosen) < count:
        progressed = False
        for key in ordered_keys:
            values = buckets[key]
            if cursor < len(values):
                chosen.append(values[cursor])
                progressed = True
                if len(chosen) == count:
                    break
        if not progressed:
            break
        cursor += 1
    if len(chosen) != count:
        raise RuntimeError("calibration 轮询选样未达到请求数量")
    return sorted(chosen)


def write_uid_list(path: str | Path, values: Iterable[str]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{value}\n" for value in values), encoding="utf-8")
