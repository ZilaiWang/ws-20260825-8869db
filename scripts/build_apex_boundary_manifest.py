#!/usr/bin/env python3
"""Build the P40-aligned APEX proposal-boundary training manifest.

The output contains only Ship/Vehicle crops.  Detector proposals are labelled
with the same score-ordered, same-fine matching used by the competition;
ambiguous localisation cases are omitted.  GT positives and deterministic
LMP-style jitter negatives are appended as training-only rows.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.augmentation.apex_boundary import assign_proposal_roles, coarse_name, scale_bin
from rsdet.augmentation.jitter_hard_negative import Box, JitterPolicy, sample_hard_negative_boxes


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_images(path: Path) -> dict[int, dict[str, Any]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"image_id", "relative_path", "fold", "group_id"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(f"{path} missing columns: {sorted(required)}")
    output: dict[int, dict[str, Any]] = {}
    for row in rows:
        image_id = int(row["image_id"])
        if image_id in output:
            raise ValueError(f"duplicate image_id={image_id}")
        fold = int(row["fold"])
        if fold not in {0, 1, 2}:
            raise ValueError(f"invalid fold={fold}")
        output[image_id] = {
            "image_id": image_id,
            "relative_path": row["relative_path"].strip(),
            "fold": fold,
            "source_group": row["group_id"].strip(),
        }
    return output


def _load_yolo_ground_truth(
    image_path: Path, relative_path: str
) -> tuple[list[dict[str, Any]], int, int]:
    with Image.open(image_path) as image:
        width, height = image.size
    label_relative = Path(relative_path.replace("images/", "labels/", 1)).with_suffix(".txt")
    label_path = image_path.parents[2] / label_relative
    rows: list[dict[str, Any]] = []
    if label_path.is_file():
        for line_number, text in enumerate(label_path.read_text().splitlines(), start=1):
            if not text.strip():
                continue
            values = text.split()
            if len(values) != 5:
                raise ValueError(f"{label_path}:{line_number} malformed YOLO label")
            category_id = int(values[0])
            cx, cy, box_width, box_height = map(float, values[1:])
            box = [
                (cx - box_width / 2) * width,
                (cy - box_height / 2) * height,
                (cx + box_width / 2) * width,
                (cy + box_height / 2) * height,
            ]
            if (
                not all(math.isfinite(value) for value in box)
                or box[2] <= box[0]
                or box[3] <= box[1]
            ):
                raise ValueError(f"{label_path}:{line_number} invalid box")
            rows.append({"category_id": category_id, "bbox_xyxy": box})
    return rows, width, height


def _row(
    *,
    row_id: str,
    image: dict[str, Any],
    box: list[float],
    category_id: int,
    score: float,
    role: str,
    target: int,
    weight: float,
    prediction_index: int | None,
    match_iou: float | None,
) -> dict[str, Any]:
    return {
        "row_id": row_id,
        "image_id": image["image_id"],
        "relative_path": image["relative_path"],
        "fold": image["fold"],
        "source_group": image["source_group"],
        "bbox_xyxy": [float(value) for value in box],
        "category_id": int(category_id),
        "coarse": coarse_name(category_id),
        "score": float(score),
        "role": role,
        "target": int(target),
        "sample_weight": float(weight),
        "scale_bin": scale_bin(box),
        "prediction_index": prediction_index,
        "match_iou": match_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-csv", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ship-floor", type=float, default=0.003)
    parser.add_argument("--vehicle-floor", type=float, default=0.001)
    parser.add_argument("--jitter-per-object", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 <= args.vehicle_floor <= args.ship_floor < 1:
        raise ValueError("require 0 <= vehicle_floor <= ship_floor < 1")

    images = _read_images(args.images_csv)
    raw_predictions = json.loads(args.predictions.read_text())
    if not isinstance(raw_predictions, list):
        raise ValueError("predictions must be a COCO-style list")
    proposals: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for source_index, raw in enumerate(raw_predictions):
        image_id = int(raw["image_id"])
        category_id = int(raw["category_id"])
        if image_id not in images or category_id not in {*range(4), 24}:
            continue
        score = float(raw["score"])
        floor = args.ship_floor if category_id < 4 else args.vehicle_floor
        if score < floor:
            continue
        x, y, width, height = map(float, raw["bbox"])
        if width <= 0 or height <= 0:
            continue
        proposals[image_id].append(
            {
                "category_id": category_id,
                "bbox_xyxy": [x, y, x + width, y + height],
                "score": score,
                "source_prediction_index": source_index,
            }
        )

    args.output.mkdir(parents=True)
    manifest_path = args.output / "apex_boundary_manifest.jsonl"
    counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    row_count = 0
    Image.MAX_IMAGE_PIXELS = None
    with manifest_path.open("w", encoding="utf-8") as handle:
        for offset, (image_id, image) in enumerate(sorted(images.items()), start=1):
            image_path = args.data_root / image["relative_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            ground_truth, width, height = _load_yolo_ground_truth(
                image_path, image["relative_path"]
            )
            image_proposals = proposals.get(image_id, [])
            roles = assign_proposal_roles(ground_truth, image_proposals)
            for local_index, (proposal, role) in enumerate(
                zip(image_proposals, roles, strict=True)
            ):
                if role["target"] is None:
                    counts["ignore_geometry"] += 1
                    continue
                role_name = str(role["role"])
                weight = 0.35 if role_name == "fp_bg" else 1.0
                record = _row(
                    row_id=f"pred:{image_id}:{proposal['source_prediction_index']}",
                    image=image,
                    box=proposal["bbox_xyxy"],
                    category_id=proposal["category_id"],
                    score=proposal["score"],
                    role=role_name,
                    target=role["target"],
                    weight=weight,
                    prediction_index=local_index,
                    match_iou=role["match_iou"],
                )
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                counts[f"{record['coarse']}:{role_name}"] += 1
                source_counts[image["source_group"]] += 1
                row_count += 1

            target_ground_truth = [
                row for row in ground_truth if int(row["category_id"]) in {*range(4), 24}
            ]
            all_boxes = [Box(*map(float, row["bbox_xyxy"])) for row in ground_truth]
            for gt_index, gt in enumerate(target_ground_truth):
                category_id = int(gt["category_id"])
                box = [float(value) for value in gt["bbox_xyxy"]]
                record = _row(
                    row_id=f"gt:{image_id}:{gt_index}",
                    image=image,
                    box=box,
                    category_id=category_id,
                    score=1.0,
                    role="gt_positive",
                    target=1,
                    weight=0.5,
                    prediction_index=None,
                    match_iou=1.0,
                )
                handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                counts[f"{record['coarse']}:gt_positive"] += 1
                row_count += 1
                policy = JitterPolicy(
                    iou_low=0.10 if category_id < 4 else 0.05,
                    iou_high=0.42 if category_id < 4 else 0.28,
                    count=args.jitter_per_object,
                    seed=args.seed,
                )
                target_box = Box(*box)
                others = [candidate for candidate in all_boxes if candidate != target_box]
                jitters = sample_hard_negative_boxes(
                    target_box,
                    image_width=width,
                    image_height=height,
                    policy=policy,
                    stable_key=f"{image_id}:{gt_index}:{category_id}",
                    other_ground_truth=others,
                )
                for jitter_index, jitter in enumerate(jitters):
                    record = _row(
                        row_id=f"jitter:{image_id}:{gt_index}:{jitter_index}",
                        image=image,
                        box=jitter.as_list(),
                        category_id=category_id,
                        score=0.0,
                        role="jitter_hard_negative",
                        target=0,
                        weight=0.5,
                        prediction_index=None,
                        match_iou=None,
                    )
                    handle.write(json.dumps(record, separators=(",", ":")) + "\n")
                    counts[f"{record['coarse']}:jitter_hard_negative"] += 1
                    row_count += 1
            if offset % 250 == 0:
                print(json.dumps({"images": offset, "rows": row_count}), flush=True)

    if not source_counts or set(image["fold"] for image in images.values()) != {0, 1, 2}:
        raise RuntimeError("incomplete source/fold inventory")
    summary = {
        "status": "complete",
        "manifest_version": "hera_guard_apex_boundary_v1",
        "images": len(images),
        "source_groups": len({image["source_group"] for image in images.values()}),
        "rows": row_count,
        "counts": dict(sorted(counts.items())),
        "floors": {"ship": args.ship_floor, "vehicle": args.vehicle_floor},
        "jitter_per_object_requested": args.jitter_per_object,
        "inputs": {
            "images_csv": str(args.images_csv),
            "images_csv_sha256": sha256_file(args.images_csv),
            "predictions": str(args.predictions),
            "predictions_sha256": sha256_file(args.predictions),
        },
        "manifest_sha256": sha256_file(manifest_path),
    }
    (args.output / "manifest_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "status.txt").write_text("complete\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
