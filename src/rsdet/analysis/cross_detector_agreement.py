"""Fold-safe cross-detector proposal agreement utilities.

The specialist detector is evidence only: it never creates a deployed box in
this module.  For each primary proposal we expose the strongest same-fine
specialist score whose geometry reaches the official coarse-class IoU.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from rsdet.evaluation.official_metric import compute_iou


def best_same_fine_support(
    primary: Sequence[Mapping[str, Any]],
    specialist: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float,
) -> list[dict[str, float]]:
    """Return strongest same-image/same-fine support for every primary row.

    Records use normalized ``bbox_xyxy`` boxes.  Ties are deterministic:
    highest specialist score, then highest IoU, then earliest input row.
    """

    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be finite and in [0, 1]")
    index: dict[tuple[int, int], list[tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for order, raw in enumerate(specialist):
        item = _validated_record(raw, require_score=True)
        index[(item["image_id"], item["category_id"])].append((order, item))

    result: list[dict[str, float]] = []
    for raw in primary:
        item = _validated_record(raw, require_score=True)
        best = (0.0, 0.0, 0)
        for order, other in index.get((item["image_id"], item["category_id"]), ()):  # noqa: B007
            overlap = compute_iou(item["bbox_xyxy"], other["bbox_xyxy"])
            if overlap < iou_threshold:
                continue
            candidate = (float(other["score"]), overlap, -order)
            if candidate > best:
                best = candidate
        support_score, support_iou, _ = best
        result.append(
            {
                "support_score": support_score,
                "support_iou": support_iou,
                "agreement_product": float(item["score"]) * support_score,
            }
        )
    return result


def marginal_false_detection_rate(delta_tp: int, delta_fp: int) -> float:
    """Return the FDR of net newly admitted predictions."""

    if delta_tp < 0 or delta_fp < 0:
        raise ValueError("delta_tp and delta_fp must be non-negative")
    denominator = delta_tp + delta_fp
    return delta_fp / denominator if denominator else 0.0


def _validated_record(raw: Mapping[str, Any], *, require_score: bool) -> dict[str, Any]:
    image_id = int(raw["image_id"])
    category_id = int(raw["category_id"])
    box = [float(value) for value in raw["bbox_xyxy"]]
    if (
        len(box) != 4
        or not all(math.isfinite(value) for value in box)
        or box[2] <= box[0]
        or box[3] <= box[1]
    ):
        raise ValueError(f"invalid bbox_xyxy={box}")
    result: dict[str, Any] = {
        "image_id": image_id,
        "category_id": category_id,
        "bbox_xyxy": box,
    }
    if require_score:
        score = float(raw["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score={score}")
        result["score"] = score
    return result


__all__ = ["best_same_fine_support", "marginal_false_detection_rate"]
