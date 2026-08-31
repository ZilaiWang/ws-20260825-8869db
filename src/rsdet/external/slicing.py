"""Scale-preserving deterministic tiling for external coarse datasets."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from rsdet.tiling.slicer import generate_tiles


def _intersection(box: list[float], tile: tuple[int, int, int, int]) -> list[float]:
    x, y, width, height = box
    x1 = max(x, float(tile[0]))
    y1 = max(y, float(tile[1]))
    x2 = min(x + width, float(tile[2]))
    y2 = min(y + height, float(tile[3]))
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _stable_empty_key(file_name: str, tile: tuple[int, int, int, int]) -> str:
    raw = f"{file_name}:{tile}".encode()
    return hashlib.sha256(raw).hexdigest()


def _slice_source_image(
    image_row: dict,
    annotations: list[dict],
    image_root: Path,
    output_image_root: Path,
    *,
    tile_size: int,
    overlap: int,
    min_visibility: float,
    empty_tiles_per_image: int,
) -> tuple[list[dict], int]:
    """Write one source image's tiles and return ID-free deterministic records."""
    source_path = image_root / image_row["file_name"]
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    width = int(image_row["width"])
    height = int(image_row["height"])
    tile_records = generate_tiles(width, height, tile_size, overlap)
    tiles = [
        (
            int(tile.x_offset),
            int(tile.y_offset),
            int(tile.x_offset + tile.width),
            int(tile.y_offset + tile.height),
        )
        for tile in tile_records
    ]
    assigned: dict[int, list[tuple[dict, list[float], float]]] = defaultdict(list)
    dropped_visibility = 0
    for annotation in annotations:
        bbox = [float(value) for value in annotation["bbox"]]
        area = bbox[2] * bbox[3]
        if area <= 0:
            dropped_visibility += 1
            continue
        center_x = bbox[0] + bbox[2] / 2.0
        center_y = bbox[1] + bbox[3] / 2.0
        candidates: list[tuple[float, int, list[float]]] = []
        for tile_index, tile in enumerate(tiles):
            if tile[0] <= center_x <= tile[2] and tile[1] <= center_y <= tile[3]:
                clipped = _intersection(bbox, tile)
                visibility = clipped[2] * clipped[3] / area
                candidates.append((visibility, tile_index, clipped))
        if not candidates:
            dropped_visibility += 1
            continue
        visibility, tile_index, clipped = max(candidates, key=lambda row: (row[0], -row[1]))
        if visibility < min_visibility:
            dropped_visibility += 1
            continue
        assigned[tile_index].append((annotation, clipped, visibility))

    positive_indices = sorted(assigned)
    negative_indices = [index for index in range(len(tiles)) if index not in assigned]
    negative_indices.sort(key=lambda index: _stable_empty_key(image_row["file_name"], tiles[index]))
    kept_indices = positive_indices + negative_indices[:empty_tiles_per_image]
    records: list[dict] = []
    with Image.open(source_path) as source_image:
        for tile_index in kept_indices:
            tile = tiles[tile_index]
            tile_name = (
                f"{int(image_row['id']):06d}_{Path(image_row['file_name']).stem}"
                f"_x{tile[0]}_y{tile[1]}.png"
            )
            source_image.crop(tile).save(output_image_root / tile_name, format="PNG")
            records.append(
                {
                    "file_name": tile_name,
                    "width": tile[2] - tile[0],
                    "height": tile[3] - tile[1],
                    "source_image_id": int(image_row["id"]),
                    "source_file_name": image_row["file_name"],
                    "x_offset": tile[0],
                    "y_offset": tile[1],
                    "tile": tile,
                    "annotations": assigned.get(tile_index, []),
                }
            )
    return records, dropped_visibility


def slice_coco(
    payload: dict,
    image_root: Path,
    output_image_root: Path,
    *,
    tile_size: int = 1024,
    overlap: int = 256,
    min_visibility: float = 0.7,
    empty_tiles_per_image: int = 2,
    workers: int = 1,
) -> tuple[dict, dict]:
    """Slice COCO images without resizing and assign each annotation to one tile."""
    if not 0.0 < min_visibility <= 1.0:
        raise ValueError("min_visibility must be in (0, 1]")
    if empty_tiles_per_image < 0:
        raise ValueError("empty_tiles_per_image must be non-negative")
    if workers < 1:
        raise ValueError("workers must be positive")
    image_root = image_root.resolve()
    output_image_root.mkdir(parents=True, exist_ok=True)
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    output_images: list[dict] = []
    output_annotations: list[dict] = []
    dropped_visibility = 0
    truncated_retained = 0
    positive_tiles = 0
    empty_tiles = 0
    coarse_counts: Counter[int] = Counter()
    output_annotation_id = 1

    image_rows = sorted(payload["images"], key=lambda row: int(row["id"]))

    def process(image_row: dict) -> tuple[list[dict], int]:
        return _slice_source_image(
            image_row,
            annotations_by_image[int(image_row["id"])],
            image_root,
            output_image_root,
            tile_size=tile_size,
            overlap=overlap,
            min_visibility=min_visibility,
            empty_tiles_per_image=empty_tiles_per_image,
        )

    if workers == 1:
        results = map(process, image_rows)
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(process, image_rows)
    try:
        for records, source_dropped in results:
            dropped_visibility += source_dropped
            for record in records:
                output_image_id = len(output_images) + 1
                rows = record.pop("annotations")
                tile = record.pop("tile")
                output_images.append({"id": output_image_id, **record})
                if rows:
                    positive_tiles += 1
                else:
                    empty_tiles += 1
                for annotation, clipped, visibility in rows:
                    local_bbox = [
                        clipped[0] - tile[0],
                        clipped[1] - tile[1],
                        clipped[2],
                        clipped[3],
                    ]
                    if visibility < 1.0 - 1e-9:
                        truncated_retained += 1
                    output_annotations.append(
                        {
                            "id": output_annotation_id,
                            "image_id": output_image_id,
                            "category_id": int(annotation["category_id"]),
                            "bbox": local_bbox,
                            "area": local_bbox[2] * local_bbox[3],
                            "iscrowd": int(annotation.get("iscrowd", 0)),
                            "source_annotation_id": int(annotation["id"]),
                            "source_visibility": visibility,
                        }
                    )
                    coarse_counts[int(annotation["category_id"])] += 1
                    output_annotation_id += 1
    finally:
        if workers != 1:
            executor.shutdown()

    output = {
        "images": output_images,
        "annotations": output_annotations,
        "categories": payload["categories"],
    }
    category_names = {int(row["id"]): row["name"] for row in payload["categories"]}
    audit = {
        "protocol": "scale_preserving_center_owned_external_slicing_v1",
        "tile_size": tile_size,
        "overlap": overlap,
        "min_visibility": min_visibility,
        "empty_tiles_per_source_image": empty_tiles_per_image,
        "workers": workers,
        "source_image_count": len(payload["images"]),
        "source_annotation_count": len(payload["annotations"]),
        "output_tile_count": len(output_images),
        "positive_tile_count": positive_tiles,
        "empty_tile_count": empty_tiles,
        "output_annotation_count": len(output_annotations),
        "dropped_visibility_count": dropped_visibility,
        "truncated_retained_count": truncated_retained,
        "coarse_category_counts": {
            category_names[key]: value for key, value in sorted(coarse_counts.items())
        },
        "resize_policy": "none",
        "duplicate_policy": "one annotation assigned to maximum-visibility center-containing tile",
    }
    return output, audit
