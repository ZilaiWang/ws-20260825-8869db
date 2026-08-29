"""Deployable scalar evidence for cross-model proposal risk resolution.

The module deliberately uses no image pixels or ground-truth-derived feature at
inference time.  It combines detector confidence, proposal geometry, proposal
density and agreement with another detector.  Ground truth is used only to
construct fold-held-out training labels.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

MULTI_DETECTOR_BASE_COLUMNS = (
    "detector_score",
    "model_y5",
    "model_m3",
    "coarse_aircraft",
    "coarse_ship",
    "coarse_vehicle",
)
MULTI_DETECTOR_GEOMETRY_COLUMNS = MULTI_DETECTOR_BASE_COLUMNS + (
    "log_short_edge",
    "log_area",
    "log_aspect",
    "log_model_image_count",
)
MULTI_DETECTOR_AGREEMENT_COLUMNS = MULTI_DETECTOR_GEOMETRY_COLUMNS + (
    "same_fine_best_iou",
    "same_fine_other_score",
    "same_coarse_best_iou",
    "same_coarse_other_score",
)


def box_iou(first: Sequence[float], second: Sequence[float]) -> float:
    """Return IoU for two half-open xyxy boxes."""

    ax0, ay0, ax1, ay1 = (float(value) for value in first)
    bx0, by0, bx1, by1 = (float(value) for value in second)
    intersection = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(
        0.0, min(ay1, by1) - max(ay0, by0)
    )
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def class_aware_nms_records(
    records: Iterable[Mapping[str, Any]], *, iou_threshold: float
) -> list[dict[str, Any]]:
    """Apply deterministic per-image, per-fine-class NMS to generic records."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    groups: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for order, raw in enumerate(records):
        item = dict(raw)
        item.setdefault("stable_order", order)
        groups[(int(item["image_id"]), int(item["category_id"]))].append(item)
    kept: list[dict[str, Any]] = []
    for key in sorted(groups):
        ordered = sorted(
            groups[key],
            key=lambda item: (-float(item["score"]), int(item["stable_order"])),
        )
        selected: list[dict[str, Any]] = []
        for item in ordered:
            if all(
                box_iou(item["bbox_xyxy"], previous["bbox_xyxy"]) < iou_threshold
                for previous in selected
            ):
                selected.append(item)
        kept.extend(selected)
    kept.sort(key=lambda item: int(item["stable_order"]))
    return kept


def candidate_validity_labels(
    records: Sequence[Mapping[str, Any]],
    *,
    gt_boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    category_mapping: Mapping[int, str],
    iou_thresholds: Mapping[str, float],
) -> np.ndarray:
    """Label whether each candidate can validly match any same-fine GT.

    Duplicate handling is intentionally delegated to deterministic NMS.  This
    target estimates candidate validity rather than encoding an arbitrary
    detector ordering into the label.
    """

    labels = np.zeros(len(records), dtype=np.int64)
    for index, item in enumerate(records):
        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        coarse = category_mapping[category_id]
        threshold = float(iou_thresholds[coarse])
        labels[index] = int(
            any(
                int(gt["category_id"]) == category_id
                and box_iou(item["bbox_xyxy"], gt["bbox_xyxy"]) >= threshold
                for gt in gt_boxes.get(image_id, ())
            )
        )
    return labels


def prediction_ledger(
    predictions: Iterable[Mapping[str, Any]], image_ids: Iterable[int]
) -> dict[int, list[dict[str, Any]]]:
    """Create a ledger whose official trace indices follow each filtered list.

    An explicit source index would develop gaps whenever a score threshold
    removes an earlier record.  The official evaluator already assigns the
    required per-image input-list index when this optional field is absent.
    """

    ledger: dict[int, list[dict[str, Any]]] = {int(image_id): [] for image_id in image_ids}
    for item in predictions:
        image_id = int(item["image_id"])
        if image_id not in ledger:
            continue
        record = dict(item)
        record.pop("source_prediction_index", None)
        ledger[image_id].append(record)
    return ledger


def build_multi_detector_features(
    records: Sequence[Mapping[str, Any]],
    *,
    category_mapping: Mapping[int, str],
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Build the full deployable agreement feature matrix."""

    counts: dict[tuple[int, str], int] = defaultdict(int)
    by_image_model: dict[tuple[int, str], list[int]] = defaultdict(list)
    for index, item in enumerate(records):
        key = (int(item["image_id"]), str(item["model_key"]).upper())
        counts[key] += 1
        by_image_model[key].append(index)

    result = np.zeros((len(records), len(MULTI_DETECTOR_AGREEMENT_COLUMNS)), dtype=np.float64)
    for index, item in enumerate(records):
        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        model = str(item["model_key"]).upper()
        if model not in {"Y5", "M3"}:
            raise ValueError(f"unsupported model_key={model}")
        coarse = category_mapping[category_id]
        if coarse not in {"aircraft", "ship", "vehicle"}:
            raise ValueError(f"unsupported coarse class={coarse}")
        x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
        width, height = x1 - x0, y1 - y0
        if width <= 0.0 or height <= 0.0:
            raise ValueError("proposal has non-positive extent")
        short_edge = min(width, height)
        aspect = max(width / height, height / width)
        other_model = "M3" if model == "Y5" else "Y5"
        same_fine_iou = same_fine_score = 0.0
        same_coarse_iou = same_coarse_score = 0.0
        for other_index in by_image_model.get((image_id, other_model), ()):
            other = records[other_index]
            other_category = int(other["category_id"])
            overlap = box_iou(item["bbox_xyxy"], other["bbox_xyxy"])
            other_score = float(other["score"])
            if category_mapping[other_category] == coarse and overlap > same_coarse_iou:
                same_coarse_iou, same_coarse_score = overlap, other_score
            if other_category == category_id and overlap > same_fine_iou:
                same_fine_iou, same_fine_score = overlap, other_score
        result[index] = (
            float(item["score"]),
            float(model == "Y5"),
            float(model == "M3"),
            float(coarse == "aircraft"),
            float(coarse == "ship"),
            float(coarse == "vehicle"),
            math.log1p(short_edge),
            math.log1p(width * height),
            math.log(aspect),
            math.log1p(counts[(image_id, model)]),
            same_fine_iou,
            same_fine_score,
            same_coarse_iou,
            same_coarse_score,
        )
    if not np.isfinite(result).all():
        raise RuntimeError("multi-detector feature matrix contains NaN/Inf")
    return result, MULTI_DETECTOR_AGREEMENT_COLUMNS


__all__ = [
    "MULTI_DETECTOR_AGREEMENT_COLUMNS",
    "MULTI_DETECTOR_BASE_COLUMNS",
    "MULTI_DETECTOR_GEOMETRY_COLUMNS",
    "box_iou",
    "build_multi_detector_features",
    "candidate_validity_labels",
    "class_aware_nms_records",
    "prediction_ledger",
]
