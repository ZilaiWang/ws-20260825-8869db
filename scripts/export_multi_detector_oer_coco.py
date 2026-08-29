#!/usr/bin/env python3
"""Export internal multi-detector OER records as COCO prediction rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


def to_coco(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in records:
        x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
        if x1 <= x0 or y1 <= y0:
            continue
        row = {
            "image_id": int(item["image_id"]),
            "category_id": int(item["category_id"]),
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "score": float(item["score"]),
        }
        if "fold" in item:
            row["source_fold"] = int(item["fold"])
        if "model_key" in item:
            row["source_model"] = str(item["model_key"])
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    rows = to_coco(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"EXPORTED {len(rows)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
