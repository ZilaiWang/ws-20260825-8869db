"""不均衡小样本训练清单生成。

本模块不复制、移动或改写原始图像和标签。它只读取冻结的 split manifest，
生成 Ultralytics 可消费的图像路径清单、数据 YAML 和类别不均衡报告。
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from rsdet.data.xh_dataset import FINE_NAMES


@dataclass(frozen=True)
class SplitSample:
    """冻结划分中的一张图像。"""

    image_id: int
    split: str
    relative_path: str
    image_path: Path
    label_path: Path
    labels: tuple[int, ...]

    @property
    def present_classes(self) -> tuple[int, ...]:
        return tuple(sorted(set(self.labels)))


@dataclass(frozen=True)
class PreparedUltralyticsData:
    """训练器需要的派生文件。"""

    base_yaml: Path
    balanced_yaml: Path
    train_list: Path
    balanced_train_list: Path
    val_list: Path
    statistics_json: Path


def _safe_resolve(root: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute():
        raise ValueError(f"manifest 中必须使用相对路径: {relative_path}")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"manifest 路径越出 data_root: {relative_path}")
    return resolved


def _label_path_for_image(root: Path, relative_path: str) -> Path:
    parts = list(Path(relative_path).parts)
    try:
        image_dir_index = parts.index("images")
    except ValueError as error:
        raise ValueError(f"图像路径必须包含 images 目录: {relative_path}") from error
    parts[image_dir_index] = "labels"
    parts[-1] = str(Path(parts[-1]).with_suffix(".txt"))
    return _safe_resolve(root, str(Path(*parts)))


def read_yolo_label_ids(label_path: str | Path, *, num_classes: int = 25) -> tuple[int, ...]:
    """读取一个 YOLO 标签文件中的细类 ID，并执行严格格式检查。"""
    path = Path(label_path)
    if not path.is_file():
        raise FileNotFoundError(f"标签文件不存在: {path}")
    labels: list[int] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        columns = line.split()
        if len(columns) != 5:
            raise ValueError(f"{path}:{line_number}: YOLO 标签必须为 5 列")
        try:
            class_id = int(columns[0])
            coordinates = [float(value) for value in columns[1:]]
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: 标签包含非法数值") from error
        if not 0 <= class_id < num_classes:
            raise ValueError(f"{path}:{line_number}: class_id={class_id} 越界")
        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(f"{path}:{line_number}: 坐标必须为有限数")
        if any(value < 0.0 or value > 1.0 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: YOLO 坐标必须在 [0, 1]")
        if coordinates[2] <= 0.0 or coordinates[3] <= 0.0:
            raise ValueError(f"{path}:{line_number}: bbox 宽高必须 > 0")
        labels.append(class_id)
    return tuple(labels)


def load_split_manifest(
    manifest_path: str | Path,
    data_root: str | Path,
    *,
    num_classes: int = 25,
) -> tuple[SplitSample, ...]:
    """加载集成契约定义的 split manifest，并解析每张图的类别。"""
    manifest = Path(manifest_path)
    root = Path(data_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data_root 不存在: {root}")
    if not manifest.is_file():
        raise FileNotFoundError(f"split manifest 不存在: {manifest}")

    document = json.loads(manifest.read_text(encoding="utf-8"))
    records = document.get("samples") if isinstance(document, Mapping) else None
    if not isinstance(records, list):
        raise ValueError("split manifest 顶层必须包含 samples 列表")

    samples: list[SplitSample] = []
    seen_ids: set[int] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"samples[{index}] 必须是对象")
        try:
            image_id = int(record["image_id"])
            split = str(record["split"])
            relative_path = str(record["relative_path"])
        except KeyError as error:
            raise ValueError(f"samples[{index}] 缺少字段: {error.args[0]}") from error
        if image_id in seen_ids:
            raise ValueError(f"manifest 中 image_id 重复: {image_id}")
        if split not in {"train", "val"}:
            raise ValueError(f"samples[{index}].split 必须是 train/val，当前为 {split!r}")

        image_path = _safe_resolve(root, relative_path)
        label_path = _label_path_for_image(root, relative_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        labels = read_yolo_label_ids(label_path, num_classes=num_classes)
        samples.append(
            SplitSample(
                image_id=image_id,
                split=split,
                relative_path=relative_path,
                image_path=image_path,
                label_path=label_path,
                labels=labels,
            )
        )
        seen_ids.add(image_id)

    if not samples:
        raise ValueError("split manifest 不包含任何样本")
    if not any(sample.split == "train" for sample in samples):
        raise ValueError("split manifest 缺少 train 样本")
    if not any(sample.split == "val" for sample in samples):
        raise ValueError("split manifest 缺少 val 样本，禁止用训练集冒充验证集")
    return tuple(samples)


def class_statistics(
    samples: Sequence[SplitSample],
    *,
    num_classes: int = 25,
) -> tuple[list[int], list[int]]:
    """返回实例数与正样本图像数。"""
    instance_counts = [0] * num_classes
    positive_image_counts = [0] * num_classes
    for sample in samples:
        for class_id in sample.labels:
            instance_counts[class_id] += 1
        for class_id in sample.present_classes:
            positive_image_counts[class_id] += 1
    return instance_counts, positive_image_counts


def effective_number_weights(
    counts: Sequence[int],
    *,
    beta: float = 0.9999,
    max_weight: float = 5.0,
) -> list[float]:
    """按 effective number 计算可审计的类别权重并限制极端值。"""
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta 必须在 [0, 1) 内")
    if max_weight < 1.0:
        raise ValueError("max_weight 必须 >= 1")
    if any(int(count) <= 0 for count in counts):
        raise ValueError("训练 split 必须覆盖全部 25 个细类")

    raw = [(1.0 - beta) / (1.0 - beta ** int(count)) for count in counts]
    mean = sum(raw) / len(raw)
    return [min(max_weight, value / mean) for value in raw]


def repeat_factors(
    samples: Sequence[SplitSample],
    *,
    repeat_threshold: float = 0.05,
    max_repeat: float = 4.0,
    num_classes: int = 25,
) -> dict[int, float]:
    """计算 LVIS 风格的图像级 repeat factor。

    每张图使用其中最稀有类别的 ``sqrt(threshold / frequency)``，并设上限，
    防止 17 个实例的极稀有类被无限复制而迅速过拟合。
    """
    if not 0.0 < repeat_threshold <= 1.0:
        raise ValueError("repeat_threshold 必须在 (0, 1] 内")
    if max_repeat < 1.0:
        raise ValueError("max_repeat 必须 >= 1")
    if not samples:
        raise ValueError("samples 不能为空")

    _, positive_counts = class_statistics(samples, num_classes=num_classes)
    frequencies = [count / len(samples) for count in positive_counts]
    factors: dict[int, float] = {}
    for sample in samples:
        factor = 1.0
        for class_id in sample.present_classes:
            frequency = frequencies[class_id]
            if frequency > 0.0:
                factor = max(factor, math.sqrt(repeat_threshold / frequency))
        factors[sample.image_id] = min(max_repeat, factor)
    return factors


def stochastic_repeat_counts(
    factors: Mapping[int, float],
    *,
    seed: int,
) -> dict[int, int]:
    """对小数 repeat factor 做确定性随机舍入。"""
    rng = random.Random(seed)
    counts: dict[int, int] = {}
    for image_id in sorted(factors):
        factor = max(1.0, float(factors[image_id]))
        whole = math.floor(factor)
        counts[image_id] = whole + int(rng.random() < factor - whole)
    return counts


def _write_lines(path: Path, paths: Sequence[Path]) -> None:
    path.write_text("".join(f"{item.as_posix()}\n" for item in paths), encoding="utf-8")


def _write_dataset_yaml(path: Path, train_list: Path, val_list: Path) -> None:
    content: dict[str, Any] = {
        "train": train_list.resolve().as_posix(),
        "val": val_list.resolve().as_posix(),
        "nc": len(FINE_NAMES),
        "names": {index: name for index, name in enumerate(FINE_NAMES)},
    }
    path.write_text(yaml.safe_dump(content, allow_unicode=True, sort_keys=False), encoding="utf-8")


def prepare_ultralytics_data(
    *,
    data_root: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    repeat_threshold: float = 0.05,
    max_repeat: float = 4.0,
    effective_beta: float = 0.9999,
    max_class_weight: float = 5.0,
) -> PreparedUltralyticsData:
    """物化基础/均衡两套训练清单以及统计报告。"""
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    samples = load_split_manifest(manifest_path, data_root, num_classes=len(FINE_NAMES))
    train_samples = tuple(sample for sample in samples if sample.split == "train")
    val_samples = tuple(sample for sample in samples if sample.split == "val")

    factors = repeat_factors(
        train_samples,
        repeat_threshold=repeat_threshold,
        max_repeat=max_repeat,
        num_classes=len(FINE_NAMES),
    )
    repeats = stochastic_repeat_counts(factors, seed=seed)
    base_paths = [sample.image_path for sample in train_samples]
    balanced_paths = [
        sample.image_path for sample in train_samples for _ in range(repeats[sample.image_id])
    ]
    random.Random(seed).shuffle(balanced_paths)
    val_paths = [sample.image_path for sample in val_samples]

    train_list = destination / "train.txt"
    balanced_train_list = destination / "train_balanced.txt"
    val_list = destination / "val.txt"
    base_yaml = destination / "dataset_base.yaml"
    balanced_yaml = destination / "dataset_balanced.yaml"
    statistics_json = destination / "imbalance_statistics.json"
    _write_lines(train_list, base_paths)
    _write_lines(balanced_train_list, balanced_paths)
    _write_lines(val_list, val_paths)
    _write_dataset_yaml(base_yaml, train_list, val_list)
    _write_dataset_yaml(balanced_yaml, balanced_train_list, val_list)

    instance_counts, positive_counts = class_statistics(train_samples, num_classes=len(FINE_NAMES))
    weights = effective_number_weights(
        instance_counts,
        beta=effective_beta,
        max_weight=max_class_weight,
    )
    report = {
        "seed": seed,
        "train_images": len(train_samples),
        "val_images": len(val_samples),
        "balanced_train_entries": len(balanced_paths),
        "repeat_threshold": repeat_threshold,
        "max_repeat": max_repeat,
        "effective_beta": effective_beta,
        "max_class_weight": max_class_weight,
        "classes": [
            {
                "id": class_id,
                "name": FINE_NAMES[class_id],
                "instances": instance_counts[class_id],
                "positive_images": positive_counts[class_id],
                "effective_number_weight": weights[class_id],
            }
            for class_id in range(len(FINE_NAMES))
        ],
        "samples": [
            {
                **asdict(sample),
                "image_path": str(sample.image_path),
                "label_path": str(sample.label_path),
                "repeat_factor": factors[sample.image_id],
                "repeat_count": repeats[sample.image_id],
            }
            for sample in train_samples
        ],
    }
    statistics_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return PreparedUltralyticsData(
        base_yaml=base_yaml,
        balanced_yaml=balanced_yaml,
        train_list=train_list,
        balanced_train_list=balanced_train_list,
        val_list=val_list,
        statistics_json=statistics_json,
    )


__all__ = [
    "PreparedUltralyticsData",
    "SplitSample",
    "class_statistics",
    "effective_number_weights",
    "load_split_manifest",
    "prepare_ultralytics_data",
    "read_yolo_label_ids",
    "repeat_factors",
    "stochastic_repeat_counts",
]
