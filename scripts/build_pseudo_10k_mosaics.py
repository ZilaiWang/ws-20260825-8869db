#!/usr/bin/env python3
"""由未见过的 CV 折图像构造带精确 GT 的 10000×10000 部署诊断图。

该产物不是新的验证集，也不用于宣称泛化性能；它只让不同切片/融合配置在
同一批未见对象、同一大图坐标系下做严格配对，提前暴露跨 tile 重复与尺度退化。
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path

from PIL import Image

CANVAS_SIZE = 10_000
GRID = 10
CELL_SIZE = CANVAS_SIZE // GRID
COARSE_NAMES = ("ship", "aircraft", "vehicle")


def _coarse_name(category_id: int) -> str:
    if 0 <= category_id < 4:
        return "ship"
    if 4 <= category_id < 24:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    raise ValueError(f"category_id outside 0--24: {category_id}")


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    index = len(parts) - 1 - parts[::-1].index("images")
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def _labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    rows: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, cx, cy, width, height = line.split()
        rows.append((int(label), float(cx), float(cy), float(width), float(height)))
    return rows


def _select_images(
    paths: list[Path],
    count: int,
    seed: int,
    required_classes: set[int] | None = None,
) -> list[Path]:
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    selected: list[Path] = []
    covered: set[int] = set()
    remaining = shuffled[:]
    target = set(range(25)) if required_classes is None else set(required_classes)
    while not target.issubset(covered):
        best = max(
            remaining,
            key=lambda path: len(
                ({row[0] for row in _labels(_label_path(path))} & target) - covered
            ),
        )
        gain = ({row[0] for row in _labels(_label_path(best))} & target) - covered
        if not gain:
            missing = sorted(target - covered)
            raise RuntimeError(f"输入集合无法覆盖要求的细类，缺失: {missing}")
        selected.append(best)
        covered.update(gain)
        remaining.remove(best)
    selected.extend(remaining[: count - len(selected)])
    return selected


def _largest_remainder_quotas(raw: dict[str, float], total: int) -> dict[str, int]:
    quotas = {key: int(math.floor(value)) for key, value in raw.items()}
    remaining = total - sum(quotas.values())
    ranked = sorted(raw, key=lambda key: (-(raw[key] - quotas[key]), key))
    for key in ranked[:remaining]:
        quotas[key] += 1
    return quotas


def _select_images_by_coarse_mix(
    paths: list[Path],
    count: int,
    seed: int,
    target_proportions: dict[str, float],
    *,
    required_classes: set[int],
) -> tuple[list[Path], dict[str, object]]:
    """Select source images that approximately realize an object-level coarse mix.

    The conversion from desired object proportions to source-image quotas uses
    the observed mean object count of each primary source category.  Fine-class
    coverage is satisfied first; quotas then fill the remaining slots.
    """

    if set(target_proportions) != set(COARSE_NAMES):
        raise ValueError("target_proportions must contain ship/aircraft/vehicle")
    if any(value <= 0.0 for value in target_proportions.values()):
        raise ValueError("target proportions must be positive")
    total = sum(target_proportions.values())
    normalized = {key: value / total for key, value in target_proportions.items()}
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    labels = {path: _labels(_label_path(path)) for path in shuffled}
    fine_sets = {path: {item[0] for item in rows} for path, rows in labels.items()}
    coarse_counts = {
        path: collections.Counter(_coarse_name(item[0]) for item in rows)
        for path, rows in labels.items()
    }

    def primary(path: Path) -> str | None:
        counts = coarse_counts[path]
        if not counts:
            return None
        return max(COARSE_NAMES, key=lambda name: (counts[name], -COARSE_NAMES.index(name)))

    selected: list[Path] = []
    remaining = shuffled[:]
    covered: set[int] = set()
    while not required_classes.issubset(covered):
        best = max(
            remaining,
            key=lambda path: (
                len((fine_sets[path] & required_classes) - covered),
                sum(coarse_counts[path].values()),
            ),
        )
        gain = (fine_sets[best] & required_classes) - covered
        if not gain:
            raise RuntimeError(
                f"cannot cover required fine classes: {sorted(required_classes - covered)}"
            )
        selected.append(best)
        remaining.remove(best)
        covered.update(gain)

    means: dict[str, float] = {}
    for name in COARSE_NAMES:
        relevant = [coarse_counts[path][name] for path in shuffled if primary(path) == name]
        if not relevant:
            raise RuntimeError(f"no source image for coarse category {name}")
        means[name] = sum(relevant) / len(relevant)
    quota_weights = {name: normalized[name] / means[name] for name in COARSE_NAMES}
    weight_total = sum(quota_weights.values())
    quotas = _largest_remainder_quotas(
        {name: count * quota_weights[name] / weight_total for name in COARSE_NAMES}, count
    )
    selected_primary = collections.Counter(primary(path) for path in selected)
    for name in COARSE_NAMES:
        need = max(0, quotas[name] - int(selected_primary[name]))
        candidates = [path for path in remaining if primary(path) == name]
        take = candidates[:need]
        selected.extend(take)
        taken = set(take)
        remaining = [path for path in remaining if path not in taken]
    if len(selected) < count:
        nonempty = [path for path in remaining if primary(path) is not None]
        selected.extend(nonempty[: count - len(selected)])
    if len(selected) < count:
        chosen = set(selected)
        selected.extend([path for path in remaining if path not in chosen][: count - len(selected)])
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("coarse-mix selection did not produce unique requested sources")

    actual_objects = collections.Counter(
        _coarse_name(item[0]) for path in selected for item in labels[path]
    )
    actual_total = sum(actual_objects.values())
    audit: dict[str, object] = {
        "target_object_proportions": normalized,
        "observed_objects_per_primary_source": means,
        "source_quotas": quotas,
        "selected_primary_sources": dict(collections.Counter(primary(path) for path in selected)),
        "actual_object_counts": {name: int(actual_objects[name]) for name in COARSE_NAMES},
        "actual_object_proportions": {
            name: (actual_objects[name] / actual_total if actual_total else 0.0)
            for name in COARSE_NAMES
        },
        "fine_classes_covered": sorted(covered),
    }
    return selected, audit


def build_mosaic(
    paths: list[Path], output_image: Path, image_id: int, seed: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = _select_images(paths, GRID * GRID, seed)
    return build_mosaic_from_selected(selected, output_image, image_id)


def build_mosaic_from_selected(
    selected: list[Path], output_image: Path, image_id: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build one mosaic from an already frozen, ordered list of 100 images."""
    if len(selected) != GRID * GRID:
        raise ValueError(f"one mosaic requires exactly {GRID * GRID} images")
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (114, 114, 114))
    annotations: list[dict[str, object]] = []
    annotation_id = image_id * 1_000_000
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    for index, path in enumerate(selected):
        row, column = divmod(index, GRID)
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            source_width, source_height = rgb.size
            scale = min(CELL_SIZE / source_width, CELL_SIZE / source_height)
            resized_width = max(1, round(source_width * scale))
            resized_height = max(1, round(source_height * scale))
            resized = rgb.resize((resized_width, resized_height), resampling)
        x_offset = column * CELL_SIZE + (CELL_SIZE - resized_width) // 2
        y_offset = row * CELL_SIZE + (CELL_SIZE - resized_height) // 2
        canvas.paste(resized, (x_offset, y_offset))
        for label, cx, cy, width, height in _labels(_label_path(path)):
            box_width = width * resized_width
            box_height = height * resized_height
            x = x_offset + (cx * resized_width) - box_width / 2.0
            y = y_offset + (cy * resized_height) - box_height / 2.0
            annotation_id += 1
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": label,
                    "bbox": [x, y, box_width, box_height],
                    "area": box_width * box_height,
                    "iscrowd": 0,
                }
            )
    output_image.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_image, quality=95, subsampling=0)
    return (
        {
            "id": image_id,
            "file_name": output_image.name,
            "width": CANVAS_SIZE,
            "height": CANVAS_SIZE,
            "source_images": [str(path) for path in selected],
        },
        annotations,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260829)
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("--count must be > 0")
    paths = [Path(line.strip()) for line in args.image_list.read_text().splitlines() if line.strip()]
    paths = [path for path in paths if path.is_file() and _label_path(path).is_file()]
    if len(paths) < GRID * GRID:
        raise RuntimeError("至少需要 100 张具备 YOLO 标签的图像")
    image_dir = args.output_dir / "images"
    images: list[dict[str, object]] = []
    annotations: list[dict[str, object]] = []
    for index in range(args.count):
        image, records = build_mosaic(
            paths,
            image_dir / f"pseudo_10k_{index:02d}.jpg",
            image_id=index + 1,
            seed=args.seed + index,
        )
        images.append(image)
        annotations.extend(records)
    categories = [{"id": index, "name": str(index)} for index in range(25)]
    payload = {"images": images, "annotations": annotations, "categories": categories}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "ground_truth.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"images": len(images), "annotations": len(annotations)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
