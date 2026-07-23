#!/usr/bin/env python3
"""MG01：审计飞机 mask、inpaint 稳定性和纯背景 tile 可用性。"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from rsdet.grouping.contracts import (
    PROTOCOL_VERSION,
    atomic_write_json,
    canonical_pixel_sha256,
    sha256_file,
)
from rsdet.grouping.masks import apply_mask_fill, build_foreground_mask, select_background_tiles
from rsdet.grouping.registry import load_annotations, load_registry
from rsdet.grouping.view_review import build_view_review_rows, write_view_review_template


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 MAR20 地点描述子背景视图")
    parser.add_argument("--registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--mar20-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-count", type=int, default=120)
    parser.add_argument("--dilation-ratios", default="0.10,0.15,0.20")
    parser.add_argument("--fill-methods", default="blur,local_mean,telea")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--tile-stride", type=int, default=112)
    parser.add_argument("--tile-valid-fraction", type=float, default=0.95)
    parser.add_argument("--max-tiles", type=int, default=8)
    parser.add_argument("--rows-per-sheet", type=int, default=8)
    parser.add_argument("--primary-dilation", type=float, default=0.15)
    return parser.parse_args(argv)


def _round_robin_sample(rows: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    if count <= 0 or count > len(rows):
        raise ValueError("sample-count 必须在有效范围内")
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        hist = json.loads(row["fine_class_hist_json"])
        dominant = min(
            (key for key, value in hist.items() if value == max(hist.values())), key=int
        )
        key = (row["is_target"], row["official_side"], dominant)
        buckets[key].append(row)
    for values in buckets.values():
        values.sort(key=lambda item: int(item["mar20_number"]))
    selected = []
    cursor = 0
    keys = sorted(buckets)
    while len(selected) < count:
        progressed = False
        for key in keys:
            values = buckets[key]
            if cursor < len(values):
                selected.append(values[cursor])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
        cursor += 1
    return sorted(selected, key=lambda item: int(item["mar20_number"]))


def _thumbnail(image: Image.Image, size: tuple[int, int] = (224, 224)) -> Image.Image:
    value = ImageOps.contain(image.convert("RGB"), size, method=Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", size, "white")
    canvas.paste(value, ((size[0] - value.width) // 2, (size[1] - value.height) // 2))
    return canvas


def _write_sheets(
    output_dir: Path,
    visual_rows: list[tuple[str, list[tuple[str, Image.Image]]]],
    rows_per_sheet: int,
) -> list[dict[str, Any]]:
    sheets_dir = output_dir / "contact_sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    columns = max(len(images) for _, images in visual_rows)
    cell_width, cell_height = 224, 252
    artifacts = []
    for sheet_index, offset in enumerate(range(0, len(visual_rows), rows_per_sheet)):
        subset = visual_rows[offset : offset + rows_per_sheet]
        canvas = Image.new("RGB", (columns * cell_width, len(subset) * cell_height), "white")
        draw = ImageDraw.Draw(canvas)
        for row_index, (node_uid, images) in enumerate(subset):
            for column, (label, image) in enumerate(images):
                x = column * cell_width
                y = row_index * cell_height
                canvas.paste(_thumbnail(image), (x, y + 28))
                draw.text((x + 4, y + 4), f"{node_uid} | {label}", fill="black")
        path = sheets_dir / f"sheet-{sheet_index:03d}.jpg"
        canvas.save(path, quality=92, optimize=False, progressive=False)
        artifacts.append({"path": path.relative_to(output_dir).as_posix(), "sha256": sha256_file(path)})
    return artifacts


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    ratios = tuple(float(value) for value in args.dilation_ratios.split(",") if value.strip())
    methods = tuple(value.strip() for value in args.fill_methods.split(",") if value.strip())
    if not ratios or len(ratios) != len(set(ratios)):
        raise ValueError("dilation-ratios 必须非空且唯一")
    if not methods or len(methods) != len(set(methods)):
        raise ValueError("fill-methods 必须非空且唯一")
    if args.primary_dilation not in ratios:
        raise ValueError("primary-dilation 必须包含在 dilation-ratios")
    rows = _round_robin_sample(load_registry(args.registry), args.sample_count)
    annotations = load_annotations(args.annotations)
    mar20_root = Path(args.mar20_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    visual_rows = []
    failures = []
    for row in rows:
        node_uid = row["node_uid"]
        annotation = annotations[node_uid]
        path = (mar20_root / row["original_relative_path"]).resolve()
        try:
            path.relative_to(mar20_root)
        except ValueError as error:
            raise ValueError(f"图像路径逃逸: {path}") from error
        with Image.open(path) as source:
            source.load()
            rgb = ImageOps.exif_transpose(source).convert("RGB")
        boxes = [item["xyxy"] for item in annotation["boxes"]]
        primary_images: list[tuple[str, Image.Image]] = [("original", rgb)]
        for ratio in ratios:
            mask = build_foreground_mask(rgb.size, boxes, dilation_ratio=ratio)
            mask_fraction = float((np.asarray(mask, dtype=np.uint8) > 0).mean())
            tiles = select_background_tiles(
                rgb,
                mask,
                tile_size=args.tile_size,
                stride=args.tile_stride,
                min_valid_fraction=args.tile_valid_fraction,
                max_tiles=args.max_tiles,
            )
            if mask_fraction >= 1.0:
                failures.append(f"{node_uid}: ratio={ratio} mask 覆盖整图")
            for method in methods:
                filled = apply_mask_fill(rgb, mask, method=method)
                audit_rows.append(
                    {
                        "node_uid": node_uid,
                        "is_target": row["is_target"],
                        "official_side": row["official_side"],
                        "bbox_count": row["bbox_count"],
                        "dilation_ratio": ratio,
                        "fill_method": method,
                        "mask_fraction": mask_fraction,
                        "background_tile_count": len(tiles),
                        "render_pixel_sha256": canonical_pixel_sha256(filled),
                    }
                )
                if ratio == args.primary_dilation:
                    primary_images.append((method, filled))
            if ratio == args.primary_dilation:
                mask_rgb = Image.merge("RGB", (mask, mask, mask))
                primary_images.append(("mask", mask_rgb))
                if tiles:
                    primary_images.append(("background_tile_0", tiles[0][0]))
        visual_rows.append((node_uid, primary_images))
    fields = list(audit_rows[0])
    csv_path = output_dir / "view_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(audit_rows)
    sheets = _write_sheets(output_dir, visual_rows, args.rows_per_sheet)
    review_path = output_dir / "manual_view_review.csv"
    review_rows = build_view_review_rows(
        audit_rows,
        primary_dilation=args.primary_dilation,
        methods=methods,
    )
    write_view_review_template(review_path, review_rows)
    mask_values = np.asarray([float(row["mask_fraction"]) for row in audit_rows])
    tile_values = np.asarray([int(row["background_tile_count"]) for row in audit_rows])
    summary = {
        "status": "waiting_for_manual_view_review" if not failures else "machine_gate_fail",
        "machine_gate": "pass" if not failures else "fail",
        "protocol_version": PROTOCOL_VERSION,
        "sample_count": len(rows),
        "audit_row_count": len(audit_rows),
        "dilation_ratios": list(ratios),
        "fill_methods": list(methods),
        "mask_fraction": {
            "min": float(mask_values.min()),
            "median": float(np.median(mask_values)),
            "max": float(mask_values.max()),
        },
        "background_tile_count": {
            "min": int(tile_values.min()),
            "median": float(np.median(tile_values)),
            "max": int(tile_values.max()),
            "zero_rows": int((tile_values == 0).sum()),
        },
        "failures": failures,
        "manual_review_contract": {
            "schema_version": "mar20-view-review-wide-v2",
            "review_target": f"primary_dilation={args.primary_dilation}, methods are separate columns",
            "binary_columns": [key for key in review_rows[0] if key not in {"node_uid", "notes"}],
            "background_tile_aircraft_means": "contact sheet 中 background_tile_0 含明显飞机主体",
        },
        "artifacts": {
            "view_audit.csv": sha256_file(csv_path),
            "manual_view_review.csv": sha256_file(review_path),
            "contact_sheets": sheets,
        },
    }
    atomic_write_json(output_dir / "view_audit_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not failures else 2


if __name__ == "__main__":
    raise SystemExit(main())
