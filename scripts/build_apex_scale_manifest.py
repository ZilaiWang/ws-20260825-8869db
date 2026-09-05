#!/usr/bin/env python3
"""Build plan-16 object-scale positive rows for the APEX classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.augmentation.apex_boundary import coarse_name
from rsdet.augmentation.jitter_hard_negative import Box
from rsdet.augmentation.object_scale_refinement import ScaleCropPolicy, build_scale_crop

TARGET_SIZES = {"ship": (64, 96, 128, 192), "vehicle": (32, 48, 64)}


def _choose(values: tuple[int, ...], key: str, maximum: int) -> list[int]:
    digest = hashlib.sha256(key.encode()).digest()
    start = int.from_bytes(digest[:4], "big") % len(values)
    return sorted(values[(start + offset) % len(values)] for offset in range(maximum))


def _images(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "image_id": int(row["image_id"]),
            "relative_path": row["relative_path"],
            "fold": int(row["fold"]),
            "source_group": row["group_id"],
        }
        for row in rows
    ]


def _ground_truth(
    data_root: Path, image: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], int, int]:
    image_path = data_root / str(image["relative_path"])
    with Image.open(image_path) as source:
        width, height = source.size
    label_path = data_root / Path(
        str(image["relative_path"]).replace("images/", "labels/", 1)
    ).with_suffix(".txt")
    rows = []
    for line_number, text in enumerate(label_path.read_text().splitlines(), start=1):
        values = text.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_number} malformed")
        category_id = int(values[0])
        cx, cy, box_width, box_height = map(float, values[1:])
        rows.append(
            {
                "category_id": category_id,
                "bbox": Box(
                    (cx - box_width / 2) * width,
                    (cy - box_height / 2) * height,
                    (cx + box_width / 2) * width,
                    (cy + box_height / 2) * height,
                ),
            }
        )
    return rows, width, height


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-scales-per-object", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.max_scales_per_object <= 2:
        raise ValueError("output must be new and max scales must be one or two")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    with args.output.open("w", encoding="utf-8") as handle:
        for image in _images(args.images_csv):
            annotations, width, height = _ground_truth(args.data_root, image)
            boxes = [row["bbox"] for row in annotations]
            for target_index, annotation in enumerate(annotations):
                category_id = int(annotation["category_id"])
                if category_id not in {*range(4), 24}:
                    continue
                coarse = coarse_name(category_id)
                key = f"{args.seed}|{image['image_id']}|{target_index}|{category_id}"
                sizes = _choose(
                    TARGET_SIZES[coarse],
                    key,
                    min(args.max_scales_per_object, len(TARGET_SIZES[coarse])),
                )
                for target_size in sizes:
                    result = build_scale_crop(
                        annotation["bbox"],
                        image_width=width,
                        image_height=height,
                        all_boxes=boxes,
                        policy=ScaleCropPolicy(
                            network_size=224,
                            target_network_side=target_size,
                            seed=args.seed,
                        ),
                        stable_key=f"{key}|{target_size}",
                    )
                    if result is None:
                        counts[f"{coarse}:rejected"] += 1
                        continue
                    crop, _ = result
                    record = {
                        "row_id": f"scale:{image['image_id']}:{target_index}:{target_size}",
                        **image,
                        "bbox_xyxy": crop.as_list(),
                        "category_id": category_id,
                        "coarse": coarse,
                        "score": 1.0,
                        "role": "object_scale_positive",
                        "target": 1,
                        "sample_weight": 0.5,
                        "scale_bin": (
                            "tiny"
                            if target_size < 32
                            else "small"
                            if target_size < 64
                            else "medium"
                            if target_size < 128
                            else "large"
                        ),
                        "prediction_index": None,
                        "match_iou": 1.0,
                        "target_network_side": target_size,
                        "outside_policy": "reflect",
                    }
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    counts[f"{coarse}:accepted"] += 1
    summary = {
        "status": "complete",
        "schema_version": "hera_guard_apex_scale_classifier_v1",
        "counts": dict(sorted(counts.items())),
        "max_scales_per_object": args.max_scales_per_object,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "outside_policy": "reflect",
        "isolated_object_stretching": False,
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
