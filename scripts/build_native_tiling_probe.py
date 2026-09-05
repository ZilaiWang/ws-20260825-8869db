#!/usr/bin/env python3
"""Build a labelled probe containing images that are changed by 1024 tiling.

The probe uses original, continuous competition images.  It deliberately keeps
only images whose long edge is above the production tile size but no larger
than the one-tile reference size.  The same pixels can therefore be evaluated
once as a whole image and once through the production overlapping-tile path.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

from PIL import Image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_image(root: Path, file_name: str) -> Path:
    relative = Path(file_name)
    candidates = (
        root / relative,
        root / relative.name,
        root / "images" / "train" / relative.name,
        root / "train" / relative.name,
    )
    existing = list(dict.fromkeys(path for path in candidates if path.is_file()))
    if len(existing) != 1:
        raise FileNotFoundError(f"expected one source for {file_name!r}, found {existing}")
    return existing[0]


def _coarse(category_id: int) -> str:
    if 0 <= category_id <= 3:
        return "ship"
    if 4 <= category_id <= 23:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    raise ValueError(f"unknown fine category id: {category_id}")


def build_probe(
    document: dict[str, Any],
    *,
    image_root: Path,
    split_tile_size: int,
    whole_tile_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not 0 < split_tile_size < whole_tile_size:
        raise ValueError("require 0 < split_tile_size < whole_tile_size")
    images = document.get("images")
    annotations = document.get("annotations")
    categories = document.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("input must be a COCO document with images and annotations")
    if not isinstance(categories, list) or len(categories) != 25:
        raise ValueError("the competition taxonomy must contain all 25 fine classes")

    selected: list[dict[str, Any]] = []
    resolved: dict[int, str] = {}
    for row in images:
        image_id = int(row["id"])
        width = int(row["width"])
        height = int(row["height"])
        long_edge = max(width, height)
        if not split_tile_size < long_edge <= whole_tile_size:
            continue
        path = _resolve_image(image_root, str(row["file_name"]))
        with Image.open(path) as opened:
            actual_size = opened.size
        if actual_size != (width, height):
            raise ValueError(
                f"COCO/image size mismatch for id={image_id}: "
                f"metadata={(width, height)} actual={actual_size}"
            )
        selected.append(dict(row))
        resolved[image_id] = str(path)

    selected_ids = {int(row["id"]) for row in selected}
    selected_annotations = [
        dict(row) for row in annotations if int(row["image_id"]) in selected_ids
    ]
    annotated_ids = {int(row["image_id"]) for row in selected_annotations}
    by_coarse: collections.Counter[str] = collections.Counter()
    for row in selected_annotations:
        by_coarse[_coarse(int(row["category_id"]))] += 1
    output = {
        "images": selected,
        "annotations": selected_annotations,
        "categories": categories,
    }
    audit = {
        "status": "complete",
        "role": "native_continuous_whole_vs_tiled_paired_probe",
        "split_tile_size": split_tile_size,
        "whole_tile_size": whole_tile_size,
        "images": len(selected),
        "annotations": len(selected_annotations),
        "annotations_by_coarse": dict(sorted(by_coarse.items())),
        "negative_images": sum(int(image["id"]) not in annotated_ids for image in selected),
        "selected_image_ids": sorted(selected_ids),
        "resolved_images": resolved,
        "limitations": [
            "The public training set contains few images larger than 1024 pixels.",
            "This probe diagnoses implementation and local tiling effects, not the hidden large-image distribution.",
            "A missing coarse class cannot be used to infer its hidden-set tiling behaviour.",
        ],
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-ground-truth", type=Path, required=True)
    parser.add_argument("--output-audit", type=Path, required=True)
    parser.add_argument("--split-tile-size", type=int, default=1024)
    parser.add_argument("--whole-tile-size", type=int, default=1280)
    args = parser.parse_args()
    document = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    output, audit = build_probe(
        document,
        image_root=args.image_root,
        split_tile_size=args.split_tile_size,
        whole_tile_size=args.whole_tile_size,
    )
    for path in (args.output_ground_truth, args.output_audit):
        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_ground_truth.write_text(
        json.dumps(output, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    audit["source_ground_truth_sha256"] = _sha256(args.ground_truth)
    audit["probe_ground_truth_sha256"] = _sha256(args.output_ground_truth)
    args.output_audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
