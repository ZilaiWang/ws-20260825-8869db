"""DIOR Pascal-VOC import utilities for four-class remote-sensing pretraining."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

from rsdet.external.dota import COARSE_CATEGORIES

DIOR_TO_COARSE: dict[str, int | None] = {
    "airplane": 0,
    "ship": 1,
    "vehicle": 2,
    "chimney": 3,
    "storagetank": 3,
    "windmill": 3,
    "airport": None,
    "baseballfield": None,
    "basketballcourt": None,
    "bridge": None,
    "dam": None,
    "expressway-service-area": None,
    "expressway-toll-station": None,
    "golffield": None,
    "groundtrackfield": None,
    "harbor": None,
    "overpass": None,
    "stadium": None,
    "tenniscourt": None,
    "trainstation": None,
}


def _image_index(image_root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for suffix in ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"):
        for path in image_root.rglob(suffix):
            if path.stem in output:
                raise ValueError(f"duplicate DIOR image stem {path.stem}")
            output[path.stem] = path
    return output


def import_dior(
    image_root: Path,
    annotation_root: Path,
    *,
    split_file: Path | None = None,
    keep_difficult: bool = False,
) -> tuple[dict, dict]:
    """Convert DIOR XML HBB annotations to the frozen four-class COCO ledger."""
    image_root = image_root.resolve()
    images_by_stem = _image_index(image_root)
    annotations_by_stem = {path.stem: path for path in annotation_root.rglob("*.xml")}
    if split_file is None:
        selected = sorted(annotations_by_stem)
    else:
        selected = [
            row.strip().split()[0]
            for row in split_file.read_text().splitlines()
            if row.strip()
        ]
        if len(selected) != len(set(selected)):
            raise ValueError("DIOR split contains duplicate image IDs")
    missing_images = sorted(set(selected) - set(images_by_stem))
    missing_annotations = sorted(set(selected) - set(annotations_by_stem))
    if missing_images or missing_annotations:
        raise ValueError(
            f"DIOR inventory mismatch: images={missing_images[:5]} xml={missing_annotations[:5]}"
        )

    images = []
    annotations = []
    original_counts: Counter[str] = Counter()
    coarse_counts: Counter[int] = Counter()
    dropped_scene = 0
    dropped_difficult = 0
    dropped_invalid = 0
    annotation_id = 1
    for stem in selected:
        image_path = images_by_stem[stem]
        with Image.open(image_path) as handle:
            width, height = handle.size
        image_id = len(images) + 1
        images.append(
            {
                "id": image_id,
                "file_name": image_path.relative_to(image_root).as_posix(),
                "width": width,
                "height": height,
                "source_annotation_file": str(annotations_by_stem[stem].resolve()),
            }
        )
        root = ET.parse(annotations_by_stem[stem]).getroot()
        for obj in root.findall("object"):
            name = str(obj.findtext("name", "")).strip().lower()
            if name not in DIOR_TO_COARSE:
                raise ValueError(f"unknown DIOR category {name!r}")
            original_counts[name] += 1
            difficult = int(obj.findtext("difficult", "0") or 0)
            if difficult > 0 and not keep_difficult:
                dropped_difficult += 1
                continue
            coarse_id = DIOR_TO_COARSE[name]
            if coarse_id is None:
                dropped_scene += 1
                continue
            box = obj.find("bndbox")
            if box is None:
                dropped_invalid += 1
                continue
            try:
                # Pascal VOC coordinates are one-based and inclusive.
                x1 = float(box.findtext("xmin", "nan")) - 1.0
                y1 = float(box.findtext("ymin", "nan")) - 1.0
                x2 = float(box.findtext("xmax", "nan"))
                y2 = float(box.findtext("ymax", "nan"))
            except ValueError:
                dropped_invalid += 1
                continue
            x1 = max(0.0, min(float(width), x1))
            y1 = max(0.0, min(float(height), y1))
            x2 = max(0.0, min(float(width), x2))
            y2 = max(0.0, min(float(height), y2))
            if x2 <= x1 or y2 <= y1:
                dropped_invalid += 1
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
                    "dior_category": name,
                    "dior_difficult": difficult,
                }
            )
            coarse_counts[coarse_id] += 1
            annotation_id += 1
    category_names = {int(row["id"]): str(row["name"]) for row in COARSE_CATEGORIES}
    payload = {
        "images": images,
        "annotations": annotations,
        "categories": COARSE_CATEGORIES,
    }
    audit = {
        "status": "complete",
        "protocol": "dior_voc_hbb_to_four_coarse_coco_v1",
        "image_count": len(images),
        "annotation_count": len(annotations),
        "split_file": str(split_file.resolve()) if split_file else None,
        "keep_difficult": keep_difficult,
        "dropped_difficult": dropped_difficult,
        "dropped_scene_structure": dropped_scene,
        "dropped_invalid_box": dropped_invalid,
        "original_category_counts": dict(sorted(original_counts.items())),
        "coarse_category_counts": {
            category_names[key]: value for key, value in sorted(coarse_counts.items())
        },
        "coordinate_policy": "Pascal VOC one-based inclusive to zero-based half-open HBB",
        "mapping": DIOR_TO_COARSE,
        "scientific_scope": "external coarse/objectness pretraining only",
    }
    return payload, audit


__all__ = ["DIOR_TO_COARSE", "import_dior"]
