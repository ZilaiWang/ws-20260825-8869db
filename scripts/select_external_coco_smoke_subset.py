#!/usr/bin/env python3
"""Select a deterministic class-covering source-image subset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_subset(payload: dict, maximum_images: int) -> tuple[dict, dict]:
    if maximum_images <= 0:
        raise ValueError("maximum_images must be positive")
    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for row in payload["annotations"]:
        annotations_by_image[int(row["image_id"])].append(row)
    image_by_id = {int(row["id"]): row for row in payload["images"]}
    category_ids = sorted(int(row["id"]) for row in payload["categories"])
    selected: set[int] = set()
    for category_id in category_ids:
        candidates = []
        for image_id in image_by_id:
            count = sum(
                int(row["category_id"]) == category_id
                for row in annotations_by_image[image_id]
            )
            candidates.append((count, -image_id, image_id))
        count, _negative_id, image_id = max(candidates)
        if count > 0:
            selected.add(image_id)
    for image_id in sorted(image_by_id):
        if len(selected) >= maximum_images:
            break
        selected.add(image_id)
    selected = set(sorted(selected)[:maximum_images])
    annotations = [
        row for row in payload["annotations"] if int(row["image_id"]) in selected
    ]
    counts = Counter(int(row["category_id"]) for row in annotations)
    output = {
        "images": [image_by_id[image_id] for image_id in sorted(selected)],
        "annotations": annotations,
        "categories": payload["categories"],
    }
    names = {int(row["id"]): row["name"] for row in payload["categories"]}
    audit = {
        "status": "complete",
        "protocol": "deterministic_class_covering_external_smoke_subset_v1",
        "maximum_images": maximum_images,
        "selected_image_ids": sorted(selected),
        "selected_image_count": len(selected),
        "selected_annotation_count": len(annotations),
        "category_counts": {names[key]: counts[key] for key in category_ids},
        "all_categories_covered": all(counts[key] > 0 for key in category_ids),
    }
    return output, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--maximum-images", type=int, default=16)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    output, audit = select_subset(payload, args.maximum_images)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False) + "\n", encoding="utf-8")
    audit["input_sha256"] = _sha256(args.input)
    audit["output_sha256"] = _sha256(args.output)
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
