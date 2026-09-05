#!/usr/bin/env python3
"""Quantify how a 10x10 pseudo-10K mosaic differs from a continuous scene.

The pseudo benchmark deliberately preserves labels and approximate detector
scale, but every 1024 inference tile can contain several unrelated source
images plus letterbox fill.  This audit makes that limitation measurable.  It
does not estimate the hidden competition distribution or produce a model
admission decision.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image

CANVAS_SIZE = 10_000
GRID = 10
CELL_SIZE = CANVAS_SIZE // GRID


def _axis_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("require tile_size > overlap >= 0")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    final = length - tile_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _quantiles(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": None, "p10": None, "median": None, "p90": None, "max": None}

    def at(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": at(0.10),
        "median": statistics.median(ordered),
        "p90": at(0.90),
        "max": ordered[-1],
    }


def _intersection_area(first: Sequence[float], second: Sequence[float]) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _resolve_source(path_text: str, source_root: Path | None) -> Path:
    original = Path(path_text)
    if original.is_file():
        return original
    if source_root is not None:
        candidates = [
            source_root / original.name,
            source_root / "images" / "train" / original.name,
            source_root / "train" / original.name,
        ]
        for candidate in candidates:
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"cannot resolve pseudo source image: {path_text}")


def _content_rect(index: int, width: int, height: int) -> tuple[tuple[float, ...], float, int, int]:
    row, column = divmod(index, GRID)
    scale = min(CELL_SIZE / width, CELL_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    x = column * CELL_SIZE + (CELL_SIZE - resized_width) // 2
    y = row * CELL_SIZE + (CELL_SIZE - resized_height) // 2
    return (
        (float(x), float(y), float(x + resized_width), float(y + resized_height)),
        scale,
        resized_width,
        resized_height,
    )


def _crossed_cells(rect: Sequence[float]) -> int:
    x1, y1, x2, y2 = rect
    right = math.nextafter(x2, -math.inf)
    bottom = math.nextafter(y2, -math.inf)
    columns = math.floor(right / CELL_SIZE) - math.floor(x1 / CELL_SIZE) + 1
    rows = math.floor(bottom / CELL_SIZE) - math.floor(y1 / CELL_SIZE) + 1
    return columns * rows


def _distance_to_lines(value: float, lines: Sequence[int]) -> float:
    return min((abs(value - line) for line in lines), default=math.inf)


def _coarse(category_id: int) -> str:
    if 0 <= category_id <= 3:
        return "ship"
    if 4 <= category_id <= 23:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    return "invalid"


def audit(
    payloads: Sequence[dict[str, Any]],
    *,
    source_root: Path | None,
    tile_size: int,
    overlap: int,
    model_imgsz: int,
    border_margin: float,
) -> dict[str, Any]:
    x_starts = _axis_starts(CANVAS_SIZE, tile_size, overlap)
    y_starts = _axis_starts(CANVAS_SIZE, tile_size, overlap)
    tiles = [
        (float(x), float(y), float(x + tile_size), float(y + tile_size))
        for y in y_starts
        for x in x_starts
    ]
    cell_lines = tuple(range(CELL_SIZE, CANVAS_SIZE, CELL_SIZE))
    tile_x_lines = sorted(set(x_starts[1:] + [x + tile_size for x in x_starts[:-1]]))
    tile_y_lines = sorted(set(y_starts[1:] + [y + tile_size for y in y_starts[:-1]]))

    source_scales: list[float] = []
    detector_scale_ratios: list[float] = []
    padding_fractions: list[float] = []
    tile_padding_fractions: list[float] = []
    cells_per_tile: list[int] = []
    source_rects_by_image: dict[int, list[tuple[float, ...]]] = {}
    source_count = 0

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    for payload in payloads:
        images.extend(payload.get("images", []))
        annotations.extend(payload.get("annotations", []))
    image_ids = [int(image["id"]) for image in images]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("duplicate image ids across ground-truth inputs")

    for image in images:
        if int(image["width"]) != CANVAS_SIZE or int(image["height"]) != CANVAS_SIZE:
            raise ValueError("audit expects 10000x10000 pseudo images")
        sources = image.get("source_images")
        if not isinstance(sources, list) or len(sources) != GRID * GRID:
            raise ValueError("each pseudo image must declare exactly 100 source_images")
        content_rects: list[tuple[float, ...]] = []
        for index, source_text in enumerate(sources):
            source = _resolve_source(str(source_text), source_root)
            with Image.open(source) as opened:
                width, height = opened.size
            rect, scale, resized_width, resized_height = _content_rect(index, width, height)
            content_rects.append(rect)
            source_scales.append(scale)
            # Training letterboxes an original image directly to model_imgsz.  The
            # pseudo path first makes its long edge 1000, then scales a 1024 tile
            # to model_imgsz.  The ratio is independent of the source dimensions.
            direct_scale = model_imgsz / max(width, height)
            pseudo_detector_scale = scale * model_imgsz / tile_size
            detector_scale_ratios.append(pseudo_detector_scale / direct_scale)
            padding_fractions.append(1.0 - (resized_width * resized_height) / CELL_SIZE**2)
            source_count += 1
        source_rects_by_image[int(image["id"])] = content_rects
        for tile in tiles:
            content_area = sum(_intersection_area(tile, rect) for rect in content_rects)
            tile_padding_fractions.append(1.0 - content_area / (tile_size * tile_size))
            cells_per_tile.append(_crossed_cells(tile))

    gt_center_to_cell_seam: list[float] = []
    gt_center_to_tile_line: list[float] = []
    gt_full_view_count: list[int] = []
    gt_safe_view_count: list[int] = []
    gt_by_coarse: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for annotation in annotations:
        x, y, width, height = (float(value) for value in annotation["bbox"])
        box = (x, y, x + width, y + height)
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        cell_distance = min(
            _distance_to_lines(center_x, cell_lines),
            _distance_to_lines(center_y, cell_lines),
        )
        tile_distance = min(
            _distance_to_lines(center_x, tile_x_lines),
            _distance_to_lines(center_y, tile_y_lines),
        )
        full = 0
        safe = 0
        for tile in tiles:
            if tile[0] <= box[0] and tile[1] <= box[1] and tile[2] >= box[2] and tile[3] >= box[3]:
                full += 1
                left_ok = tile[0] == 0 or box[0] - tile[0] >= border_margin
                top_ok = tile[1] == 0 or box[1] - tile[1] >= border_margin
                right_ok = tile[2] == CANVAS_SIZE or tile[2] - box[2] >= border_margin
                bottom_ok = tile[3] == CANVAS_SIZE or tile[3] - box[3] >= border_margin
                if left_ok and top_ok and right_ok and bottom_ok:
                    safe += 1
        gt_center_to_cell_seam.append(cell_distance)
        gt_center_to_tile_line.append(tile_distance)
        gt_full_view_count.append(full)
        gt_safe_view_count.append(safe)
        bucket = gt_by_coarse[_coarse(int(annotation["category_id"]))]
        bucket["gt"] += 1
        bucket["cell_seam_center_le_16"] += int(cell_distance <= 16)
        bucket["cell_seam_center_le_32"] += int(cell_distance <= 32)
        bucket["tile_line_center_le_16"] += int(tile_distance <= 16)
        bucket["tile_line_center_le_32"] += int(tile_distance <= 32)
        bucket["no_full_tile_view"] += int(full == 0)
        bucket["no_safe_tile_view"] += int(safe == 0)

    unique_tile_count = len(tiles)
    return {
        "status": "complete",
        "role": "pseudo10k_fidelity_diagnostic_not_hidden_distribution_estimate",
        "geometry": {
            "canvas_size": CANVAS_SIZE,
            "grid": GRID,
            "cell_size": CELL_SIZE,
            "tile_size": tile_size,
            "overlap": overlap,
            "stride": tile_size - overlap,
            "model_imgsz": model_imgsz,
            "tiles_per_image": unique_tile_count,
            "all_tiles_cross_artificial_vertical_and_horizontal_seams": all(
                _crossed_cells(tile) >= 4 for tile in tiles
            ),
            "cells_intersected_per_tile": _quantiles(cells_per_tile[:unique_tile_count]),
        },
        "inventory": {
            "pseudo_images": len(images),
            "source_images": source_count,
            "annotations": len(annotations),
        },
        "source_transform": {
            "source_to_cell_scale": _quantiles(source_scales),
            "pseudo_vs_direct_detector_object_scale_ratio": _quantiles(detector_scale_ratios),
            "cell_flat_padding_fraction": _quantiles(padding_fractions),
            "tile_flat_padding_fraction": _quantiles(tile_padding_fractions),
        },
        "ground_truth_geometry": {
            "center_distance_to_artificial_cell_seam_px": _quantiles(gt_center_to_cell_seam),
            "center_distance_to_any_internal_tile_line_px": _quantiles(gt_center_to_tile_line),
            "full_tile_views": _quantiles(gt_full_view_count),
            "safe_tile_views_at_border_margin": _quantiles(gt_safe_view_count),
            "by_coarse": {name: dict(counts) for name, counts in sorted(gt_by_coarse.items())},
        },
        "interpretation_limits": [
            "Pseudo mosaics preserve labels and approximate detector pixel scale.",
            "Artificial seams, flat padding, target density, and discontinuous context are not representative of one continuous satellite scene.",
            "Use this proxy for tiling/runtime regression and paired engineering checks, not as an absolute hidden Recall/FDR forecast.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, action="append", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=256)
    parser.add_argument("--model-imgsz", type=int, default=1280)
    parser.add_argument("--border-margin", type=float, default=8.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.ground_truth]
    result = audit(
        payloads,
        source_root=args.source_root,
        tile_size=args.tile_size,
        overlap=args.overlap,
        model_imgsz=args.model_imgsz,
        border_margin=args.border_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
