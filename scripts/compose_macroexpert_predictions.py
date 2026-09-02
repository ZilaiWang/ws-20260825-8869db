#!/usr/bin/env python3
"""Compose disjoint MacroExpert-M weak-family predictions with Y5 aircraft."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def compose(
    primary: list[dict], specialist: list[dict], routed_image_ids: set[int]
) -> list[dict]:
    weak = {0, 1, 2, 3, 24}
    aircraft = set(range(4, 24))
    output: list[dict] = []
    for row in primary:
        image_id = int(row["image_id"])
        label = int(row["category_id"])
        if image_id not in routed_image_ids or label in aircraft:
            output.append(row)
    for row in specialist:
        image_id = int(row["image_id"])
        label = int(row["category_id"])
        if image_id not in routed_image_ids:
            raise ValueError(f"specialist row outside routed images: {image_id}")
        if label not in weak:
            raise ValueError(f"specialist emitted forbidden official label: {label}")
        output.append(row)
    output.sort(key=lambda row: (int(row["image_id"]), int(row["category_id"]),
                                 -float(row["score"]), tuple(row["bbox"])))
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--specialist", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", type=Path)
    source.add_argument("--ground-truth", type=Path)
    parser.add_argument("--fold", type=int, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folds = set(args.fold)
    if args.manifest is not None:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        routed = {int(row["image_id"]) for row in manifest["samples"]
                  if int(row["fold"]) in folds}
    else:
        ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
        routed = {
            int(row["id"])
            for row in ground_truth["images"]
            if any(str(row.get("file_name", "")).startswith(f"fold{fold}_") for fold in folds)
        }
    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    specialist = json.loads(args.specialist.read_text(encoding="utf-8"))
    rows = compose(primary, specialist, routed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"primary": len(primary), "specialist": len(specialist),
                      "routed_images": len(routed), "output": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
