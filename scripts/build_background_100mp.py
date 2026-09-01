#!/usr/bin/env python3
"""Materialize a deterministic, GT-excluded Background-100MP benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


def _intersects(first: tuple[int, int, int, int], second: tuple[float, float, float, float]) -> bool:
    return not (
        first[2] <= second[0]
        or first[0] >= second[2]
        or first[3] <= second[1]
        or first[1] >= second[3]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=768)
    parser.add_argument("--gt-margin", type=int, default=64)
    parser.add_argument("--target-megapixels", type=float, default=100.0)
    parser.add_argument("--visual-exclusions", type=Path)
    args = parser.parse_args()
    if args.tile_size <= 0 or not 0 < args.stride <= args.tile_size:
        raise ValueError("require tile_size >= stride > 0")
    coco = json.loads(args.coco.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in coco["images"]}
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    margin = args.gt_margin
    for annotation in coco["annotations"]:
        x, y, width, height = map(float, annotation["bbox"])
        boxes.setdefault(int(annotation["image_id"]), []).append(
            (x - margin, y - margin, x + width + margin, y + height + margin)
        )
    candidates: list[tuple[str, int, tuple[int, int, int, int]]] = []
    for image_id, row in images.items():
        width, height = int(row["width"]), int(row["height"])
        if width < args.tile_size or height < args.tile_size:
            continue
        xs = sorted(set(range(0, width - args.tile_size + 1, args.stride)) | {width - args.tile_size})
        ys = sorted(set(range(0, height - args.tile_size + 1, args.stride)) | {height - args.tile_size})
        for y in ys:
            for x in xs:
                crop = (x, y, x + args.tile_size, y + args.tile_size)
                if any(_intersects(crop, gt_box) for gt_box in boxes.get(image_id, ())):
                    continue
                key = hashlib.sha256(f"bg100mp|{image_id}|{x}|{y}".encode()).hexdigest()
                candidates.append((key, image_id, crop))
    candidates.sort()
    required_pixels = int(args.target_megapixels * 1_000_000)
    excluded_keys: set[str] = set()
    if args.visual_exclusions is not None:
        exclusion_payload = json.loads(args.visual_exclusions.read_text(encoding="utf-8"))
        excluded_keys = {str(value) for value in exclusion_payload["candidate_keys"]}
    selected = []
    pixels = 0
    output_images = args.output / "images"
    output_images.mkdir(parents=True, exist_ok=True)
    for candidate_key, source_id, crop in candidates:
        if pixels >= required_pixels:
            break
        if candidate_key in excluded_keys:
            continue
        row = images[source_id]
        source = args.image_root / str(row["file_name"])
        if not source.is_file():
            raise FileNotFoundError(source)
        image_id = len(selected)
        destination = output_images / f"bg_{image_id:06d}.png"
        with Image.open(source) as image:
            image.convert("RGB").crop(crop).save(destination, optimize=True)
        record = {
            "image_id": image_id,
            "file_name": str(destination.relative_to(args.output)),
            "width": args.tile_size,
            "height": args.tile_size,
            "source_image_id": source_id,
            "source_file_name": str(row["file_name"]),
            "crop_xyxy": list(crop),
            "candidate_key": candidate_key,
            "gt_exclusion_margin": margin,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        }
        selected.append(record)
        pixels += args.tile_size * args.tile_size
    if pixels < required_pixels:
        raise RuntimeError(
            f"insufficient audited background: {pixels / 1e6:.3f}MP < {args.target_megapixels}MP"
        )
    manifest = args.output / "background_100mp_manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected),
        encoding="utf-8",
    )
    summary = {
        "version": "background_100mp_v1",
        "image_count": len(selected),
        "megapixels": pixels / 1_000_000.0,
        "source_image_count": len({row["source_image_id"] for row in selected}),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "object_free_by_gt_margin": True,
        "visual_exclusion_count": len(excluded_keys),
        "automatic_geometry_admission": True,
        "formal_admission": False,
        "status": "waiting_for_visual_background_audit",
    }
    (args.output / "background_100mp_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
