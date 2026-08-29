#!/usr/bin/env python3
"""由未见过的 CV 折图像构造带精确 GT 的 10000×10000 部署诊断图。

该产物不是新的验证集，也不用于宣称泛化性能；它只让不同切片/融合配置在
同一批未见对象、同一大图坐标系下做严格配对，提前暴露跨 tile 重复与尺度退化。
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

CANVAS_SIZE = 10_000
GRID = 10
CELL_SIZE = CANVAS_SIZE // GRID


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


def _select_images(paths: list[Path], count: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    shuffled = paths[:]
    rng.shuffle(shuffled)
    selected: list[Path] = []
    covered: set[int] = set()
    remaining = shuffled[:]
    while len(covered) < 25:
        best = max(
            remaining,
            key=lambda path: len({row[0] for row in _labels(_label_path(path))} - covered),
        )
        gain = {row[0] for row in _labels(_label_path(best))} - covered
        if not gain:
            missing = sorted(set(range(25)) - covered)
            raise RuntimeError(f"输入折无法覆盖全部 25 细类，缺失: {missing}")
        selected.append(best)
        covered.update(gain)
        remaining.remove(best)
    selected.extend(remaining[: count - len(selected)])
    return selected


def build_mosaic(
    paths: list[Path], output_image: Path, image_id: int, seed: int
) -> tuple[dict[str, object], list[dict[str, object]]]:
    selected = _select_images(paths, GRID * GRID, seed)
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
            "seed": seed,
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
