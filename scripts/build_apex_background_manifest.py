#!/usr/bin/env python3
"""Build reviewed Background-100MP hard negatives for APEX A3-lite."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.augmentation.apex_boundary import coarse_name, scale_bin

TARGET_CATEGORIES = frozenset({0, 1, 2, 3, 24})


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _images(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        int(row["image_id"]): {"fold": int(row["fold"]), "source_group": row["group_id"]}
        for row in rows
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--background-root", type=Path, required=True)
    parser.add_argument("--background-manifest", type=Path, required=True)
    parser.add_argument("--review-decision", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-source-bucket-cap", type=int, default=64)
    parser.add_argument("--ship-floor", type=float, default=0.003)
    parser.add_argument("--vehicle-floor", type=float, default=0.001)
    args = parser.parse_args()
    if args.output.exists() or args.per_source_bucket_cap <= 0:
        raise ValueError("output must be new and cap positive")

    background_rows = [
        json.loads(line) for line in args.background_manifest.read_text().splitlines() if line.strip()
    ]
    review = json.loads(args.review_decision.read_text())
    if (
        review.get("status") != "pass"
        or not review.get("all_tiles_visually_reviewed")
        or int(review.get("visible_or_ambiguous_targets_remaining", -1)) != 0
        or review.get("manifest_sha256") != _sha(args.background_manifest)
    ):
        raise ValueError("Background-100MP has not passed the frozen missing-label guard")
    by_image = {int(row["image_id"]): row for row in background_rows}
    if len(by_image) != len(background_rows):
        raise ValueError("duplicate background image ids")
    source_images = _images(args.images_csv)
    predictions = json.loads(args.predictions.read_text())
    buckets: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for prediction_index, prediction in enumerate(predictions):
        category_id = int(prediction["category_id"])
        if category_id not in TARGET_CATEGORIES:
            continue
        coarse = coarse_name(category_id)
        floor = args.ship_floor if coarse == "ship" else args.vehicle_floor
        if float(prediction["score"]) < floor:
            continue
        background = by_image[int(prediction["image_id"])]
        source_image_id = int(background["source_image_id"])
        source = source_images[source_image_id]
        x, y, width, height = map(float, prediction["bbox"])
        box = [x, y, x + width, y + height]
        record = {
            "row_id": f"reviewed-bg:{prediction_index}",
            "image_id": -1 - int(prediction["image_id"]),
            "relative_path": str((args.background_root / background["file_name"]).resolve()),
            "fold": int(source["fold"]),
            "source_group": str(source["source_group"]),
            "bbox_xyxy": box,
            "category_id": category_id,
            "coarse": coarse,
            "score": float(prediction["score"]),
            "role": "retrieved_background_negative",
            "target": 0,
            "sample_weight": 0.5,
            "scale_bin": scale_bin(box),
            "prediction_index": prediction_index,
            "match_iou": 0.0,
            "background_candidate_key": background["candidate_key"],
            "background_manifest_sha256": review["manifest_sha256"],
        }
        buckets[(str(source["source_group"]), category_id, record["scale_bin"])].append(record)

    selected: list[dict[str, Any]] = []
    for key in sorted(buckets):
        values = sorted(
            buckets[key], key=lambda row: (-float(row["score"]), str(row["row_id"]))
        )[: args.per_source_bucket_cap]
        selected.extend(values)
    selected.sort(key=lambda row: str(row["row_id"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            counts[f"{row['coarse']}:{row['scale_bin']}"] += 1
    summary = {
        "status": "complete",
        "schema_version": "hera_guard_apex_reviewed_background_negative_v1",
        "rows": len(selected),
        "counts": dict(sorted(counts.items())),
        "review_decision_sha256": _sha(args.review_decision),
        "background_manifest_sha256": _sha(args.background_manifest),
        "predictions_sha256": _sha(args.predictions),
        "output_sha256": _sha(args.output),
        "per_source_bucket_cap": args.per_source_bucket_cap,
        "compositing": False,
        "use": "reviewed active-FP hard negatives; not claimed as mask-based Domain-RAG composite",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
