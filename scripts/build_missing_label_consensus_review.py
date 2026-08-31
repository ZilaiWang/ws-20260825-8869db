#!/usr/bin/env python3
"""Build a strict OOF vehicle missing-label review manifest.

The command is deliberately read-only with respect to source annotations.  It
emits ranked CSV/JSON review files and a blank human decision column.  No row
is admitted as a label or ignore region until a separate compiler consumes a
completed review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from rsdet.analysis.missing_label_consensus import build_missing_label_candidates


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--category-id", type=int, default=24)
    parser.add_argument("--support-iou-min", type=float, default=0.35)
    parser.add_argument("--existing-gt-iou-max", type=float, default=0.05)
    parser.add_argument("--product-min", type=float, default=0.059)
    parser.add_argument("--primary-score-min", type=float, default=0.05)
    parser.add_argument("--specialist-score-min", type=float, default=0.05)
    parser.add_argument("--dedup-iou", type=float, default=0.50)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    seen_files: set[str] = set()

    for fold, raw_dir in enumerate(args.fold_dir):
        fold_dir = raw_dir.resolve()
        paths = {
            "primary": fold_dir / "y5_predictions.json",
            "specialist": fold_dir / "dfine_predictions.json",
            "ground_truth": fold_dir / "instances_val.json",
        }
        for path in paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        ground_truth = _load(paths["ground_truth"])
        images = {int(row["id"]): row for row in ground_truth["images"]}
        overlap = seen_files.intersection(str(row["file_name"]) for row in images.values())
        if overlap:
            raise ValueError(f"OOF folds overlap on file_name: {sorted(overlap)[:3]}")
        seen_files.update(str(row["file_name"]) for row in images.values())

        candidates = build_missing_label_candidates(
            _load(paths["primary"]),
            _load(paths["specialist"]),
            ground_truth["annotations"],
            category_id=args.category_id,
            support_iou_min=args.support_iou_min,
            existing_gt_iou_max=args.existing_gt_iou_max,
            product_min=args.product_min,
            primary_score_min=args.primary_score_min,
            specialist_score_min=args.specialist_score_min,
            dedup_iou=args.dedup_iou,
        )
        for local_rank, candidate in enumerate(candidates, start=1):
            image = images[int(candidate["image_id"])]
            x1, y1, x2, y2 = [float(value) for value in candidate["bbox_xyxy"]]
            sx1, sy1, sx2, sy2 = [
                float(value) for value in candidate["support_bbox_xyxy"]
            ]
            width = int(image["width"])
            height = int(image["height"])
            edge_distance = min(x1, y1, width - x2, height - y2)
            candidate_id = f"F{fold}-{int(candidate['image_id']):06d}-{int(candidate['primary_order']):06d}"
            rows.append(
                {
                    "candidate_id": candidate_id,
                    "fold": fold,
                    "fold_rank": local_rank,
                    "image_id": int(candidate["image_id"]),
                    "file_name": str(image["file_name"]),
                    "image_width": width,
                    "image_height": height,
                    "category_id": int(candidate["category_id"]),
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "support_bbox_x1": sx1,
                    "support_bbox_y1": sy1,
                    "support_bbox_x2": sx2,
                    "support_bbox_y2": sy2,
                    "primary_score": float(candidate["primary_score"]),
                    "support_score": float(candidate["support_score"]),
                    "support_iou": float(candidate["support_iou"]),
                    "agreement_product": float(candidate["agreement_product"]),
                    "maximum_gt_iou": float(candidate["maximum_gt_iou"]),
                    "nearest_gt_category_id": candidate["nearest_gt_category_id"],
                    "edge_distance_px": edge_distance,
                    "near_image_edge": edge_distance < 2.0,
                    "human_decision": "",
                    "review_note": "",
                }
            )
        inputs.append(
            {
                "fold": fold,
                "fold_dir": str(fold_dir),
                "image_count": len(images),
                "candidate_count": len(candidates),
                "sha256": {name: _sha256(path) for name, path in paths.items()},
            }
        )

    rows.sort(
        key=lambda row: (
            -float(row["agreement_product"]),
            -float(row["support_iou"]),
            int(row["fold"]),
            str(row["candidate_id"]),
        )
    )
    for global_rank, row in enumerate(rows, start=1):
        row["global_rank"] = global_rank

    fields = ["global_rank", *[key for key in rows[0] if key != "global_rank"]] if rows else ["global_rank"]
    csv_path = output_dir / "manual_missing_label_review.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "missing_label_candidates.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "status": "waiting_for_manual_missing_label_review",
        "protocol": "strict_oof_two_detector_vehicle_missing_label_review_v1",
        "automatic_annotation_admission": False,
        "automatic_ignore_admission": False,
        "candidate_count": len(rows),
        "image_count": len(seen_files),
        "per_fold_candidates": {
            str(item["fold"]): item["candidate_count"] for item in inputs
        },
        "thresholds": {
            "category_id": args.category_id,
            "support_iou_min": args.support_iou_min,
            "existing_gt_iou_max": args.existing_gt_iou_max,
            "product_min": args.product_min,
            "primary_score_min": args.primary_score_min,
            "specialist_score_min": args.specialist_score_min,
            "dedup_iou": args.dedup_iou,
        },
        "inputs": inputs,
        "outputs": {
            "review_csv": str(csv_path),
            "review_csv_sha256": _sha256(csv_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
