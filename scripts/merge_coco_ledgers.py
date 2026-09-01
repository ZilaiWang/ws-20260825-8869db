#!/usr/bin/env python3
"""Merge disjoint-image COCO ledgers and deterministically renumber annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    categories = documents[0]["categories"]
    if any(document["categories"] != categories for document in documents[1:]):
        raise ValueError("COCO category ledgers differ")
    images = [
        {**row, "fold": fold}
        for fold, document in enumerate(documents)
        for row in document["images"]
    ]
    annotations = [
        {**row, "id": index}
        for index, row in enumerate(
            row for document in documents for row in document["annotations"]
        )
    ]
    image_ids = [int(row["id"]) for row in images]
    annotation_ids = [int(row["id"]) for row in annotations]
    if len(image_ids) != len(set(image_ids)):
        raise ValueError("COCO image IDs overlap")
    if len(annotation_ids) != len(set(annotation_ids)):
        raise RuntimeError("deterministic annotation renumbering failed")
    if {int(row["image_id"]) for row in annotations} - set(image_ids):
        raise ValueError("annotation references unknown image")
    payload = {
        "images": sorted(images, key=lambda row: int(row["id"])),
        "annotations": sorted(annotations, key=lambda row: int(row["id"])),
        "categories": categories,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
