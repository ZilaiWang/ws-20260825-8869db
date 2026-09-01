#!/usr/bin/env python3
"""Merge disjoint-image COCO detection lists with duplicate-row auditing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    image_sets = []
    for path in args.input:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"prediction file is not a COCO list: {path}")
        ids = {int(row["image_id"]) for row in payload}
        if any(ids & previous for previous in image_sets):
            raise ValueError("prediction input image sets overlap")
        image_sets.append(ids)
        records.extend(payload)
    records.sort(
        key=lambda row: (
            int(row["image_id"]),
            -float(row["score"]),
            int(row["category_id"]),
            tuple(float(value) for value in row["bbox"]),
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
