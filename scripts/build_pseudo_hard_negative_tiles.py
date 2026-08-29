#!/usr/bin/env python3
"""Materialize fold-pure 1024 hard-background tiles from pseudo-10K candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, width, height = (float(value) for value in box)
    return x, y, x + width, y + height


def iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx0, ly0, lx1, ly1 = xywh_to_xyxy(left)
    rx0, ry0, rx1, ry1 = xywh_to_xyxy(right)
    intersection = max(0.0, min(lx1, rx1) - max(lx0, rx0)) * max(
        0.0, min(ly1, ry1) - max(ly0, ry0)
    )
    union = (lx1 - lx0) * (ly1 - ly0) + (rx1 - rx0) * (ry1 - ry0) - intersection
    return intersection / union if union > 0.0 else 0.0


def crop_window(
    box: Sequence[float], *, image_width: int, image_height: int, tile_size: int
) -> tuple[int, int, int, int]:
    x, y, width, height = (float(value) for value in box)
    cx, cy = x + width / 2.0, y + height / 2.0
    x0 = min(max(0, round(cx - tile_size / 2)), max(0, image_width - tile_size))
    y0 = min(max(0, round(cy - tile_size / 2)), max(0, image_height - tile_size))
    return x0, y0, min(image_width, x0 + tile_size), min(image_height, y0 + tile_size)


def yolo_labels_for_crop(
    annotations: Sequence[Mapping[str, Any]], window: tuple[int, int, int, int]
) -> list[str]:
    x0, y0, x1, y1 = window
    crop_width, crop_height = x1 - x0, y1 - y0
    labels: list[str] = []
    for item in annotations:
        gx, gy, width, height = (float(value) for value in item["bbox"])
        cx, cy = gx + width / 2.0, gy + height / 2.0
        if not (x0 <= cx < x1 and y0 <= cy < y1):
            continue
        bx0, by0 = max(gx, x0), max(gy, y0)
        bx1, by1 = min(gx + width, x1), min(gy + height, y1)
        if bx1 <= bx0 or by1 <= by0:
            continue
        local_cx = ((bx0 + bx1) / 2.0 - x0) / crop_width
        local_cy = ((by0 + by1) / 2.0 - y0) / crop_height
        local_width = (bx1 - bx0) / crop_width
        local_height = (by1 - by0) / crop_height
        labels.append(
            f"{int(item['category_id'])} {local_cx:.8f} {local_cy:.8f} "
            f"{local_width:.8f} {local_height:.8f}"
        )
    return labels


def select_background_candidates(
    predictions: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    *,
    count: int,
    max_any_iou: float,
    min_center_distance: float,
) -> list[Mapping[str, Any]]:
    ranked = sorted(
        predictions,
        key=lambda item: -float(item.get("detector_score", item["score"])),
    )
    selected: list[Mapping[str, Any]] = []
    centers: list[tuple[float, float]] = []
    for item in ranked:
        if max((iou(item["bbox"], gt["bbox"]) for gt in annotations), default=0.0) > max_any_iou:
            continue
        x, y, width, height = (float(value) for value in item["bbox"])
        center = (x + width / 2.0, y + height / 2.0)
        if any(math.dist(center, previous) < min_center_distance for previous in centers):
            continue
        selected.append(item)
        centers.append(center)
        if len(selected) == count:
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--pseudo-root", type=Path, required=True)
    parser.add_argument("--held-out-fold", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tiles-per-image", type=int, default=160)
    parser.add_argument("--max-any-iou", type=float, default=0.10)
    parser.add_argument("--min-center-distance", type=float, default=384.0)
    args = parser.parse_args()

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in gt["images"]}
    gt_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    pred_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in gt["annotations"]:
        gt_by_image[int(item["image_id"])].append(item)
    for item in predictions:
        pred_by_image[int(item["image_id"])].append(item)

    image_dir = args.output_dir / "images" / "train"
    label_dir = args.output_dir / "labels" / "train"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    audit_rows: list[dict[str, Any]] = []
    Image.MAX_IMAGE_PIXELS = None
    for image_id, meta in sorted(images.items()):
        fold = int(meta["fold"])
        if fold == args.held_out_fold:
            continue
        selected = select_background_candidates(
            pred_by_image[image_id],
            gt_by_image[image_id],
            count=args.tiles_per_image,
            max_any_iou=args.max_any_iou,
            min_center_distance=args.min_center_distance,
        )
        source_path = args.pseudo_root / f"fold_{fold}" / "images" / meta["file_name"]
        with Image.open(source_path) as source:
            source.load()
            for rank, item in enumerate(selected):
                window = crop_window(
                    item["bbox"],
                    image_width=int(meta["width"]),
                    image_height=int(meta["height"]),
                    tile_size=args.tile_size,
                )
                labels = yolo_labels_for_crop(gt_by_image[image_id], window)
                stem = f"heldout{args.held_out_fold}_img{image_id}_rank{rank:04d}"
                image_path = image_dir / f"{stem}.jpg"
                label_path = label_dir / f"{stem}.txt"
                source.crop(window).save(image_path, quality=95, subsampling=0)
                label_path.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")
                audit_rows.append(
                    {
                        "image": str(image_path.resolve()),
                        "label": str(label_path.resolve()),
                        "source_image_id": image_id,
                        "source_fold": fold,
                        "candidate_score": float(item["score"]),
                        "window": list(window),
                        "label_count": len(labels),
                    }
                )
    if any(int(row["source_fold"]) == args.held_out_fold for row in audit_rows):
        raise RuntimeError("held-out fold leaked into hard-negative tiles")
    train_list = args.output_dir / "hard_negative_images.txt"
    train_list.write_text("\n".join(str(row["image"]) for row in audit_rows) + "\n", encoding="utf-8")
    summary = {
        "status": "complete",
        "protocol": "fold_pure_pseudo10k_hard_background_tiles_v1",
        "held_out_fold": args.held_out_fold,
        "tile_size": args.tile_size,
        "tiles_per_image": args.tiles_per_image,
        "max_any_iou": args.max_any_iou,
        "min_center_distance": args.min_center_distance,
        "tile_count": len(audit_rows),
        "empty_tile_count": sum(int(row["label_count"]) == 0 for row in audit_rows),
        "source_folds": sorted({int(row["source_fold"]) for row in audit_rows}),
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
        "train_list_sha256": _sha256(train_list),
        "tiles": audit_rows,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "tiles"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
