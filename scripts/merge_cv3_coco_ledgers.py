#!/usr/bin/env python3
"""Merge three disjoint single-fold COCO GT/prediction ledgers for CV3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, nargs=3, required=True)
    parser.add_argument("--pred", type=Path, nargs=3, required=True)
    parser.add_argument("--output-gt", type=Path, required=True)
    parser.add_argument("--output-pred", type=Path, required=True)
    args = parser.parse_args()

    merged_images: list[dict[str, Any]] = []
    merged_annotations: list[dict[str, Any]] = []
    merged_predictions: list[dict[str, Any]] = []
    categories: list[dict[str, Any]] | None = None
    seen_images: set[int] = set()
    fold_counts: dict[str, dict[str, int]] = {}
    for fold, (gt_path, pred_path) in enumerate(zip(args.gt, args.pred, strict=True)):
        gt = _load(gt_path)
        pred = _load(pred_path)
        if not isinstance(gt, dict) or not isinstance(pred, list):
            raise ValueError("GT must be a COCO object and predictions a COCO list")
        fold_images = {int(row["id"]) for row in gt["images"]}
        if not fold_images or fold_images & seen_images:
            raise ValueError(f"fold {fold} image ids are empty or overlap earlier folds")
        pred_images = {int(row["image_id"]) for row in pred}
        if not pred_images <= fold_images:
            raise ValueError(f"fold {fold} predictions contain non-held-out images")
        current_categories = gt["categories"]
        if categories is None:
            categories = current_categories
        elif current_categories != categories:
            raise ValueError("category taxonomies differ across folds")
        for row in gt["images"]:
            merged_images.append({**row, "fold": fold})
        merged_annotations.extend(gt["annotations"])
        merged_predictions.extend({**row, "fold": fold} for row in pred)
        seen_images.update(fold_images)
        fold_counts[str(fold)] = {
            "images": len(fold_images),
            "annotations": len(gt["annotations"]),
            "predictions": len(pred),
        }
    if len(seen_images) != 4481:
        raise ValueError(f"CV3 must cover 4481 unique images, got {len(seen_images)}")
    assert categories is not None
    output_gt = {
        "images": sorted(merged_images, key=lambda row: int(row["id"])),
        "annotations": merged_annotations,
        "categories": categories,
        "cv3_merge_audit": fold_counts,
    }
    args.output_gt.parent.mkdir(parents=True, exist_ok=True)
    args.output_pred.parent.mkdir(parents=True, exist_ok=True)
    args.output_gt.write_text(
        json.dumps(output_gt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output_pred.write_text(
        json.dumps(merged_predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        f"CV3_COCO_MERGE_PASS images={len(seen_images)} "
        f"annotations={len(merged_annotations)} predictions={len(merged_predictions)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
