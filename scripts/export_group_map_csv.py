#!/usr/bin/env python3
"""Export an audited image_id-to-group JSON mapping as a stable CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    mapping = json.loads(args.input.read_text(encoding="utf-8"))
    rows = [
        {"image_id": int(image_id), "group_id": str(group_id)}
        for image_id, group_id in mapping.items()
    ]
    rows.sort(key=lambda row: row["image_id"])
    if len(rows) != len({row["image_id"] for row in rows}):
        raise ValueError("duplicate image ids")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("image_id", "group_id"))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
