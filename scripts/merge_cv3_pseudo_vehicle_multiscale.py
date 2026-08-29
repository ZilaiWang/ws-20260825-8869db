#!/usr/bin/env python3
"""Merge a conservative 800-pixel vehicle expert into 1024 predictions.

The primary ledger is retained in full.  Only category 24 (vehicle) is read
from the secondary ledger.  Duplicate vehicle boxes are removed independently
per image with deterministic same-class NMS.  No threshold is selected here;
the existing cross-fit frontier evaluator remains responsible for calibration.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

VEHICLE_CATEGORY_ID = 24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iou_xywh(first: list[float], second: list[float]) -> float:
    ax0, ay0, aw, ah = (float(value) for value in first)
    bx0, by0, bw, bh = (float(value) for value in second)
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    width = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    height = max(0.0, min(ay1, by1) - max(ay0, by0))
    intersection = width * height
    union = aw * ah + bw * bh - intersection
    return intersection / union if union > 0.0 else 0.0


def _validate(predictions: Any, source: str) -> list[dict[str, Any]]:
    if not isinstance(predictions, list):
        raise ValueError(f"{source} must contain a COCO detection list")
    validated: list[dict[str, Any]] = []
    for index, item in enumerate(predictions):
        if not isinstance(item, dict):
            raise ValueError(f"{source}[{index}] is not an object")
        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        score = float(item["score"])
        bbox = [float(value) for value in item["bbox"]]
        if len(bbox) != 4 or not all(math.isfinite(value) for value in [score, *bbox]):
            raise ValueError(f"{source}[{index}] has invalid score/bbox")
        if image_id <= 0 or category_id < 0 or category_id > VEHICLE_CATEGORY_ID:
            raise ValueError(f"{source}[{index}] has invalid IDs")
        if score < 0.0 or bbox[2] <= 0.0 or bbox[3] <= 0.0:
            raise ValueError(f"{source}[{index}] has invalid geometry/score")
        validated.append(dict(item))
    return validated


def merge_vehicle_predictions(
    primary: list[dict[str, Any]],
    secondary: list[dict[str, Any]],
    *,
    nms_iou: float,
    secondary_score_scale: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Retain primary predictions and add NMS-filtered secondary vehicles."""

    if not 0.0 <= nms_iou <= 1.0:
        raise ValueError("nms_iou must be in [0, 1]")
    if not math.isfinite(secondary_score_scale) or secondary_score_scale <= 0.0:
        raise ValueError("secondary_score_scale must be finite and positive")

    passthrough = [dict(item) for item in primary if int(item["category_id"]) != 24]
    vehicles: dict[int, list[tuple[int, int, dict[str, Any]]]] = defaultdict(list)
    for index, item in enumerate(primary):
        if int(item["category_id"]) == VEHICLE_CATEGORY_ID:
            output = dict(item)
            output["multiscale_source"] = "primary_1024"
            vehicles[int(item["image_id"])].append((0, index, output))
    secondary_vehicle_count = 0
    for index, item in enumerate(secondary):
        if int(item["category_id"]) != VEHICLE_CATEGORY_ID:
            continue
        secondary_vehicle_count += 1
        output = dict(item)
        output["score"] = min(1.0, float(output["score"]) * secondary_score_scale)
        output["multiscale_source"] = "secondary_800"
        vehicles[int(item["image_id"])].append((1, index, output))

    selected: list[dict[str, Any]] = []
    kept_secondary = 0
    for image_id in sorted(vehicles):
        ordered = sorted(
            vehicles[image_id],
            key=lambda value: (-float(value[2]["score"]), value[0], value[1]),
        )
        kept: list[tuple[int, int, dict[str, Any]]] = []
        for candidate in ordered:
            if all(
                _iou_xywh(candidate[2]["bbox"], previous[2]["bbox"]) <= nms_iou
                for previous in kept
            ):
                kept.append(candidate)
                kept_secondary += int(candidate[0] == 1)
        selected.extend(item for _, _, item in kept)

    merged = passthrough + selected
    merged.sort(
        key=lambda item: (
            int(item["image_id"]),
            int(item["category_id"]),
            -float(item["score"]),
            tuple(float(value) for value in item["bbox"]),
        )
    )
    summary = {
        "primary_total": len(primary),
        "primary_vehicle": sum(
            int(item["category_id"] == VEHICLE_CATEGORY_ID) for item in primary
        ),
        "secondary_total": len(secondary),
        "secondary_vehicle": secondary_vehicle_count,
        "kept_vehicle_total": len(selected),
        "kept_secondary_vehicle": kept_secondary,
        "merged_total": len(merged),
    }
    return merged, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--secondary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--secondary-score-scale", type=float, default=1.0)
    args = parser.parse_args()

    primary = _validate(json.loads(args.primary.read_text(encoding="utf-8")), "primary")
    secondary = _validate(
        json.loads(args.secondary.read_text(encoding="utf-8")), "secondary"
    )
    primary_images = {int(item["image_id"]) for item in primary}
    secondary_images = {int(item["image_id"]) for item in secondary}
    if primary_images != secondary_images:
        raise ValueError(
            "primary and secondary image sets differ: "
            f"{sorted(primary_images)} vs {sorted(secondary_images)}"
        )

    merged, counts = merge_vehicle_predictions(
        primary,
        secondary,
        nms_iou=args.nms_iou,
        secondary_score_scale=args.secondary_score_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {
        "status": "complete",
        "protocol": "primary_1024_plus_vehicle_only_800_same_class_nms_v1",
        "primary": {"path": str(args.primary.resolve()), "sha256": _sha256(args.primary)},
        "secondary": {
            "path": str(args.secondary.resolve()),
            "sha256": _sha256(args.secondary),
        },
        "output": {"path": str(args.output.resolve()), "sha256": _sha256(args.output)},
        "nms_iou": args.nms_iou,
        "secondary_score_scale": args.secondary_score_scale,
        **counts,
    }
    args.summary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
