"""Official-match-aligned labels and conservative rescue for APEX.

The module is intentionally model agnostic.  It labels existing detector
proposals using the competition's score-ordered, same-fine, one-to-one match
and selects a probability cutoff using only calibration groups.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from rsdet.evaluation.official_metric import compute_iou

SHIP_IDS = frozenset(range(4))
AIRCRAFT_IDS = frozenset(range(4, 24))
VEHICLE_IDS = frozenset({24})


def coarse_name(category_id: int) -> str:
    if category_id in SHIP_IDS:
        return "ship"
    if category_id in AIRCRAFT_IDS:
        return "aircraft"
    if category_id in VEHICLE_IDS:
        return "vehicle"
    raise ValueError(f"unknown category_id: {category_id}")


def official_iou_threshold(category_id: int) -> float:
    return 0.35 if category_id in VEHICLE_IDS else 0.50


def scale_bin(box_xyxy: Sequence[float]) -> str:
    if len(box_xyxy) != 4:
        raise ValueError("box must be xyxy")
    side = math.sqrt(
        max(0.0, float(box_xyxy[2]) - float(box_xyxy[0]))
        * max(0.0, float(box_xyxy[3]) - float(box_xyxy[1]))
    )
    if side < 32:
        return "tiny"
    if side < 64:
        return "small"
    if side < 128:
        return "medium"
    return "large"


def assign_proposal_roles(
    ground_truth: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    background_iou_limit: float = 0.02,
) -> list[dict[str, Any]]:
    """Label proposals without changing their order.

    Positive means the proposal becomes the canonical official TP when all
    supplied low-floor predictions participate.  Duplicates, wrong-fine
    matches and unambiguous background are reliable negatives.  Localization
    boundary cases are ignored to avoid teaching one-pixel noise as semantic
    background.
    """

    if not 0 <= background_iou_limit < 1:
        raise ValueError("background_iou_limit must be in [0, 1)")
    gt_rows = [dict(row) for row in ground_truth]
    pred_rows = [dict(row) for row in predictions]
    matched: set[int] = set()
    output: list[dict[str, Any] | None] = [None] * len(pred_rows)
    by_category: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(pred_rows):
        by_category[int(row["category_id"])].append(index)

    for category_id, indices in sorted(by_category.items()):
        threshold = official_iou_threshold(category_id)
        indices.sort(key=lambda index: (-float(pred_rows[index]["score"]), index))
        same_indices = [
            i for i, row in enumerate(gt_rows) if int(row["category_id"]) == category_id
        ]
        coarse_indices = [
            i
            for i, row in enumerate(gt_rows)
            if coarse_name(int(row["category_id"])) == coarse_name(category_id)
        ]
        for prediction_index in indices:
            box = pred_rows[prediction_index]["bbox_xyxy"]
            overlaps = {
                gt_index: compute_iou(list(box), list(gt_rows[gt_index]["bbox_xyxy"]))
                for gt_index in range(len(gt_rows))
            }
            available = [
                (overlaps[gt_index], gt_index)
                for gt_index in same_indices
                if gt_index not in matched and overlaps[gt_index] >= threshold
            ]
            if available:
                overlap, gt_index = max(available, key=lambda item: (item[0], -item[1]))
                matched.add(gt_index)
                role, target = "canonical_tp", 1
                matched_category: int | None = category_id
            else:
                same_iou = max((overlaps[i] for i in same_indices), default=0.0)
                coarse_iou = max((overlaps[i] for i in coarse_indices), default=0.0)
                any_iou = max(overlaps.values(), default=0.0)
                matched_category = None
                if same_iou >= threshold:
                    role, target = "fp_duplicate", 0
                elif coarse_iou >= threshold:
                    role, target = "fp_cls", 0
                elif any_iou <= background_iou_limit:
                    role, target = "fp_bg", 0
                else:
                    role, target = "ignore_geometry", None
                overlap = max(same_iou, coarse_iou)
            output[prediction_index] = {
                "role": role,
                "target": target,
                "match_iou": float(overlap),
                "matched_category_id": matched_category,
            }
    if any(row is None for row in output):
        raise RuntimeError("proposal role assignment incomplete")
    return [dict(row) for row in output if row is not None]


@dataclass(frozen=True)
class PrecisionThreshold:
    threshold: float
    precision: float
    true_positives: int
    false_positives: int
    candidate_count: int


def select_precision_threshold(
    probabilities: Sequence[float],
    targets: Sequence[int],
    *,
    minimum_precision: float,
    minimum_true_positives: int = 1,
) -> PrecisionThreshold | None:
    """Choose the largest-TP prefix satisfying the registered precision."""

    if len(probabilities) != len(targets) or not 0 < minimum_precision <= 1:
        raise ValueError("invalid threshold selection inputs")
    rows = sorted(
        (
            (float(probability), int(target), index)
            for index, (probability, target) in enumerate(zip(probabilities, targets, strict=True))
        ),
        key=lambda row: (-row[0], row[2]),
    )
    if any(
        not math.isfinite(probability) or not 0 <= probability <= 1 for probability, _, _ in rows
    ):
        raise ValueError("probabilities must be finite in [0, 1]")
    if any(target not in {0, 1} for _, target, _ in rows):
        raise ValueError("targets must be binary")
    tp = fp = 0
    eligible: list[PrecisionThreshold] = []
    for offset, (probability, target, _) in enumerate(rows):
        tp += target
        fp += 1 - target
        if offset + 1 < len(rows) and rows[offset + 1][0] == probability:
            continue
        precision = tp / (tp + fp)
        if tp >= minimum_true_positives and precision >= minimum_precision:
            eligible.append(PrecisionThreshold(probability, precision, tp, fp, tp + fp))
    if not eligible:
        return None
    return max(
        eligible, key=lambda item: (item.true_positives, -item.false_positives, item.threshold)
    )


def rescue_indices(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[float],
    *,
    detector_threshold: float,
    rescue_floor: float,
    quality_threshold: float,
) -> set[int]:
    """Return tail indices allowed to bypass the frozen detector threshold."""

    if len(rows) != len(probabilities):
        raise ValueError("rows and probabilities differ")
    selected: set[int] = set()
    for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
        score = float(row["score"])
        if rescue_floor <= score < detector_threshold and float(probability) >= quality_threshold:
            selected.add(index)
    return selected
