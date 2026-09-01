"""DOTA oriented-box import utilities for coarse detector pretraining."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image

COARSE_CATEGORIES = [
    {"id": 0, "name": "aircraft"},
    {"id": 1, "name": "ship"},
    {"id": 2, "name": "vehicle"},
    {"id": 3, "name": "other_remote_object"},
]

DOTA_TO_COARSE: dict[str, int | None] = {
    "plane": 0,
    "ship": 1,
    "small-vehicle": 2,
    "large-vehicle": 2,
    # Large scene/region boxes remain structured background.  Treating a whole
    # harbor or sports field as one foreground object creates giant boxes that
    # overlap the compact ships/vehicles we actually want to learn.
    "baseball-diamond": None,
    "tennis-court": None,
    "basketball-court": None,
    "ground-track-field": None,
    "harbor": None,
    "bridge": None,
    "roundabout": None,
    "soccer-ball-field": None,
    "swimming-pool": None,
    "helicopter": 3,
    "storage-tank": 3,
    "container-crane": 3,
}

PRIMARY_COMPACT_CATEGORIES = frozenset(
    {"plane", "ship", "small-vehicle", "large-vehicle"}
)
DIFFICULT_POLICIES = frozenset({"drop", "keep_primary", "keep_all_mapped"})


@dataclass(frozen=True)
class DotaObject:
    polygon: tuple[float, ...]
    category: str
    difficulty: int


def parse_dota_label(path: Path) -> tuple[list[DotaObject], int]:
    """Parse one DOTA labelTxt file and return objects plus invalid-line count."""
    objects: list[DotaObject] = []
    invalid = 0
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        fields = raw.strip().split()
        if not fields or fields[0].lower().startswith(("imagesource", "gsd")):
            continue
        if len(fields) < 10:
            invalid += 1
            continue
        try:
            polygon = tuple(float(value) for value in fields[:8])
            difficulty = int(fields[9])
        except ValueError:
            invalid += 1
            continue
        category = fields[8]
        if category not in DOTA_TO_COARSE:
            invalid += 1
            continue
        objects.append(DotaObject(polygon, category, difficulty))
    return objects, invalid


def _image_index(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for suffix in ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"):
        for path in image_root.rglob(suffix):
            if path.stem in index:
                raise ValueError(f"duplicate image stem {path.stem}: {index[path.stem]} / {path}")
            index[path.stem] = path
    return index


def import_dota(
    image_root: Path,
    label_root: Path,
    *,
    keep_difficult: bool = False,
    difficult_policy: str = "drop",
    require_exact_stem_set: bool = True,
) -> tuple[dict, dict]:
    """Convert DOTA labelTxt OBBs into clipped four-class COCO HBB annotations."""
    if difficult_policy not in DIFFICULT_POLICIES:
        raise ValueError(f"unsupported difficult policy: {difficult_policy}")
    if keep_difficult:
        if difficult_policy != "drop":
            raise ValueError("keep_difficult cannot be combined with difficult_policy")
        difficult_policy = "keep_all_mapped"
    image_root = image_root.resolve()
    label_paths = sorted(label_root.rglob("*.txt"))
    if not label_paths:
        raise ValueError(f"no DOTA labelTxt files under {label_root}")
    images_by_stem = _image_index(image_root)
    labels_by_stem = {path.stem: path for path in label_paths}
    images_without_labels = sorted(set(images_by_stem) - set(labels_by_stem))
    labels_without_images = sorted(set(labels_by_stem) - set(images_by_stem))
    if images_without_labels:
        raise ValueError(
            f"{len(images_without_labels)} images lack labels, first={images_without_labels[:5]}"
        )
    if require_exact_stem_set and labels_without_images:
        raise ValueError(
            f"{len(labels_without_images)} labels lack images, first={labels_without_images[:5]}"
        )
    images: list[dict] = []
    annotations: list[dict] = []
    original_counts: Counter[str] = Counter()
    retained_counts: Counter[str] = Counter()
    invalid_lines = 0
    dropped_difficult = 0
    retained_difficult = 0
    retained_difficult_counts: Counter[str] = Counter()
    dropped_invalid_box = 0
    dropped_scene_structure = 0
    annotation_id = 1

    for stem, image_path in sorted(images_by_stem.items()):
        label_path = labels_by_stem[stem]
        with Image.open(image_path) as image:
            width, height = image.size
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": image_path.relative_to(image_root).as_posix(),
                "width": width,
                "height": height,
                "source_label_file": str(label_path.resolve()),
            }
        )
        objects, invalid = parse_dota_label(label_path)
        invalid_lines += invalid
        for obj in objects:
            original_counts[obj.category] += 1
            coarse_id = DOTA_TO_COARSE[obj.category]
            if coarse_id is None:
                dropped_scene_structure += 1
                continue
            if obj.difficulty > 0:
                retain = difficult_policy == "keep_all_mapped" or (
                    difficult_policy == "keep_primary"
                    and obj.category in PRIMARY_COMPACT_CATEGORIES
                )
                if not retain:
                    dropped_difficult += 1
                    continue
                retained_difficult += 1
                retained_difficult_counts[obj.category] += 1
            xs = obj.polygon[0::2]
            ys = obj.polygon[1::2]
            x1 = max(0.0, min(float(width), min(xs)))
            y1 = max(0.0, min(float(height), min(ys)))
            x2 = max(0.0, min(float(width), max(xs)))
            y2 = max(0.0, min(float(height), max(ys)))
            if x2 <= x1 or y2 <= y1:
                dropped_invalid_box += 1
                continue
            bbox = [x1, y1, x2 - x1, y2 - y1]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": coarse_id,
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0,
                    "dota_polygon": list(obj.polygon),
                    "dota_category": obj.category,
                    "dota_difficulty": obj.difficulty,
                }
            )
            retained_counts[obj.category] += 1
            annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": COARSE_CATEGORIES,
    }
    audit = {
        "protocol": "dota_obb_to_clipped_hbb_coarse_v2",
        "image_root": str(image_root),
        "label_root": str(label_root.resolve()),
        "image_count": len(images),
        "annotation_count": len(annotations),
        "keep_difficult": difficult_policy == "keep_all_mapped",
        "difficult_policy": difficult_policy,
        "primary_compact_categories": sorted(PRIMARY_COMPACT_CATEGORIES),
        "require_exact_stem_set": require_exact_stem_set,
        "labels_without_images_count": len(labels_without_images),
        "labels_without_images_first": labels_without_images[:20],
        "dropped_difficult": dropped_difficult,
        "retained_difficult": retained_difficult,
        "retained_difficult_category_counts": dict(
            sorted(retained_difficult_counts.items())
        ),
        "dropped_invalid_box": dropped_invalid_box,
        "dropped_scene_structure": dropped_scene_structure,
        "invalid_label_lines": invalid_lines,
        "original_category_counts": dict(sorted(original_counts.items())),
        "retained_category_counts": dict(sorted(retained_counts.items())),
        "coarse_category_counts": _coarse_counts(annotations),
        "mapping": DOTA_TO_COARSE,
        "scene_structure_policy": (
            "large DOTA scene/region categories are retained in imagery as background, "
            "not trained as giant foreground boxes"
        ),
        "scientific_scope": "external coarse/objectness pretraining only",
    }
    return payload, audit


def _coarse_counts(annotations: Iterable[dict]) -> dict[str, int]:
    names = {row["id"]: row["name"] for row in COARSE_CATEGORIES}
    counts = Counter(names[int(row["category_id"])] for row in annotations)
    return dict(sorted(counts.items()))
