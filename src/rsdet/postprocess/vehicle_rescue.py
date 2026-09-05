"""Incumbent-preserving, category-limited rescue; never consumes annotations."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any

from rsdet.evaluation.official_metric import compute_iou


def append_vehicle_rescue(
    incumbent: dict[int, list[dict[str, Any]]],
    auxiliary: dict[int, list[dict[str, Any]]],
    *,
    dedup_iou: float = 0.35,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    """Inputs are already thresholded, xyxy records in the exact image universe.

    Every incumbent record is preserved, including its score and ordering.
    Auxiliary non-Vehicle records are ignored; accepted Vehicle records are
    appended in deterministic score/coordinate order. No gain in greedy TP
    count is guaranteed: new boxes may change matching order, so evaluate it.
    """
    if not isfinite(dedup_iou) or not 0 < dedup_iou <= 1:
        raise ValueError("dedup_iou must be finite in (0, 1]")
    if set(incumbent) != set(auxiliary):
        raise ValueError("both views must cover the same explicit image universe")
    for view in (incumbent, auxiliary):
        for rows in view.values():
            for row in rows:
                label = row["category_id"]
                box = row["bbox_xyxy"]
                score = float(row["score"])
                if int(label) != label or not 0 <= int(label) < 25:
                    raise ValueError("invalid official category")
                if not isfinite(score) or not 0 <= score <= 1:
                    raise ValueError("invalid score")
                if (len(box) != 4 or not all(isfinite(x) for x in box)
                        or box[2] <= box[0] or box[3] <= box[1]):
                    raise ValueError("invalid xyxy bbox")
    result = deepcopy(incumbent)
    stats = {"auxiliary_vehicle": 0, "suppressed_overlap": 0, "added_vehicle": 0}
    for image_id in sorted(result):
        kept = [r for r in result[image_id] if r["category_id"] == 24]
        candidates = sorted(
            (r for r in auxiliary[image_id] if r["category_id"] == 24),
            key=lambda r: (-r["score"], tuple(r["bbox_xyxy"])),
        )
        stats["auxiliary_vehicle"] += len(candidates)
        for row in candidates:
            if any(compute_iou(row["bbox_xyxy"], old["bbox_xyxy"]) >= dedup_iou
                   for old in kept):
                stats["suppressed_overlap"] += 1
                continue
            copy = deepcopy(row)
            result[image_id].append(copy)
            kept.append(copy)
            stats["added_vehicle"] += 1
    return result, stats
