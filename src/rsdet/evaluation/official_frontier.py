"""Fixed-risk frontier computed from the official prediction-first matcher.

This module is the single source of truth for SCOPE/HERA frontier labels.  It
does not reimplement TP assignment: candidate NMS is deterministic, then
``evaluate_predictions_with_trace`` supplies the exact official match trace.
The frontier scan only consumes that trace and evaluates *complete* equal-score
blocks, so an operating point can never select an arbitrary subset of ties.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import groupby
from typing import Any, Iterable, Mapping, Sequence

from rsdet.evaluation.official_metric import (
    OfficialEvaluationTrace,
    OverallMetrics,
    evaluate_predictions_with_trace,
)
from rsdet.postprocess.nms import class_aware_nms_predictions


@dataclass(frozen=True)
class FixedRiskPoint:
    """Best prefix at one FDR ceiling."""

    target_fdr: float
    recall: float
    fdr: float
    score_threshold: float | None
    tp: int
    fp: int
    fn: int
    selected: int


@dataclass(frozen=True)
class OfficialFrontierResult:
    """Official trace plus fixed-risk operating points.

    ``total_tp``/``total_fp`` describe all NMS-kept predictions.  They are
    deliberately distinct from each point's active TP/FP counts.
    """

    points: dict[float, FixedRiskPoint]
    n_gt: int
    n_kept: int
    total_tp: int
    total_fp: int
    selected_candidate_ids: dict[float, tuple[int, ...]]
    selected_tp_candidate_ids: dict[float, tuple[int, ...]]
    selected_fp_candidate_ids: dict[float, tuple[int, ...]]
    kept_predictions: dict[int, list[dict[str, Any]]]
    trace: OfficialEvaluationTrace
    official_metrics: OverallMetrics


def _validated_levels(fdr_levels: Sequence[float]) -> tuple[float, ...]:
    levels: list[float] = []
    for raw in fdr_levels:
        level = float(raw)
        if not math.isfinite(level) or not 0.0 <= level <= 1.0:
            raise ValueError(f"invalid FDR level: {raw}")
        if level not in levels:
            levels.append(level)
    if not levels:
        raise ValueError("fdr_levels must not be empty")
    return tuple(levels)


def _normalize_predictions(
    predictions: Iterable[Mapping[str, Any]],
    *,
    image_ids: set[int],
) -> dict[int, list[dict[str, Any]]]:
    """Build a deterministic image ledger and validate global candidate IDs."""

    by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    seen_ids: set[int] = set()
    for order, raw in enumerate(predictions):
        image_id = int(raw["image_id"])
        if image_id not in image_ids:
            continue
        candidate_id = int(raw.get("source_prediction_index", raw.get("candidate_id", order)))
        if candidate_id in seen_ids:
            raise ValueError(f"source_prediction_index must be globally unique: {candidate_id}")
        seen_ids.add(candidate_id)
        box = [float(value) for value in raw["bbox_xyxy"]]
        record = {
            "image_id": image_id,
            "category_id": int(raw["category_id"]),
            "score": float(raw["score"]),
            "bbox_xyxy": box,
            "source_prediction_index": candidate_id,
        }
        by_image[image_id].append(record)

    # Official matching is stable for equal scores.  Freeze that stable order
    # by candidate ID rather than inheriting caller/container iteration order.
    for records in by_image.values():
        records.sort(key=lambda row: (-row["score"], row["source_prediction_index"]))
    return by_image


def scan_fixed_risk_points(
    *,
    scored_tp_rows: Iterable[tuple[float, int, bool]],
    n_gt: int,
    fdr_levels: Sequence[float],
) -> dict[float, FixedRiskPoint]:
    """Scan complete score-tie blocks.

    Rows are ``(score, candidate_id, is_tp)``.  Candidate ID is only a stable
    order inside a tie; all rows in the tie are added before an operating point
    is evaluated.
    """

    levels = _validated_levels(fdr_levels)
    rows = sorted(scored_tp_rows, key=lambda row: (-float(row[0]), int(row[1])))
    empty_recall = 1.0 if n_gt == 0 else 0.0
    points = {
        level: FixedRiskPoint(
            target_fdr=level,
            recall=empty_recall,
            fdr=0.0,
            score_threshold=None,
            tp=0,
            fp=0,
            fn=n_gt,
            selected=0,
        )
        for level in levels
    }

    tp = fp = 0
    for score, block in groupby(rows, key=lambda row: float(row[0])):
        for _, _, is_tp in block:
            if is_tp:
                tp += 1
            else:
                fp += 1
        selected = tp + fp
        recall = tp / n_gt if n_gt else 1.0
        fdr = fp / selected if selected else 0.0
        for level in levels:
            current = points[level]
            if fdr <= level and recall > current.recall:
                points[level] = FixedRiskPoint(
                    target_fdr=level,
                    recall=recall,
                    fdr=fdr,
                    score_threshold=score,
                    tp=tp,
                    fp=fp,
                    fn=max(n_gt - tp, 0),
                    selected=selected,
                )
    return points


def official_fixed_risk_frontier(
    *,
    gt_boxes: dict[int, list[dict[str, Any]]],
    predictions: Iterable[Mapping[str, Any]],
    category_mapping: dict[int, str],
    iou_thresholds: dict[str, float],
    image_ids: set[int] | None = None,
    fdr_levels: Sequence[float] = (0.12, 0.11, 0.10),
    nms_iou: float = 0.50,
) -> OfficialFrontierResult:
    """Run deterministic NMS, official matching, and fixed-risk scanning."""

    scope = set(gt_boxes) if image_ids is None else {int(value) for value in image_ids}
    scoped_gt = {image_id: list(gt_boxes.get(image_id, [])) for image_id in sorted(scope)}
    pred_by_image = _normalize_predictions(predictions, image_ids=scope)
    kept = class_aware_nms_predictions(pred_by_image, nms_iou)
    metrics, trace = evaluate_predictions_with_trace(
        scoped_gt,
        kept,
        class_names=list(dict.fromkeys(category_mapping.values())),
        category_mapping=category_mapping,
        iou_thresholds=iou_thresholds,
    )

    tp_ids = {int(match.prediction_index) for match in trace.matches}
    rows: list[tuple[float, int, bool]] = []
    for image_id in sorted(kept):
        for record in kept[image_id]:
            candidate_id = int(record["source_prediction_index"])
            rows.append((float(record["score"]), candidate_id, candidate_id in tp_ids))

    n_gt = sum(len(items) for items in scoped_gt.values())
    points = scan_fixed_risk_points(
        scored_tp_rows=rows,
        n_gt=n_gt,
        fdr_levels=fdr_levels,
    )
    selected_candidate_ids: dict[float, tuple[int, ...]] = {}
    selected_tp_candidate_ids: dict[float, tuple[int, ...]] = {}
    selected_fp_candidate_ids: dict[float, tuple[int, ...]] = {}
    ranked_rows = sorted(rows, key=lambda row: (-row[0], row[1]))
    for level, point in points.items():
        selected = tuple(
            candidate_id
            for score, candidate_id, _ in ranked_rows
            if point.score_threshold is not None and score >= point.score_threshold
        )
        selected_candidate_ids[level] = selected
        selected_tp_candidate_ids[level] = tuple(
            candidate_id for candidate_id in selected if candidate_id in tp_ids
        )
        selected_fp_candidate_ids[level] = tuple(
            candidate_id for candidate_id in selected if candidate_id not in tp_ids
        )
    return OfficialFrontierResult(
        points=points,
        n_gt=n_gt,
        n_kept=len(rows),
        total_tp=int(metrics.details["tp"]),
        total_fp=int(metrics.details["fp"]),
        selected_candidate_ids=selected_candidate_ids,
        selected_tp_candidate_ids=selected_tp_candidate_ids,
        selected_fp_candidate_ids=selected_fp_candidate_ids,
        kept_predictions=kept,
        trace=trace,
        official_metrics=metrics,
    )


def min_fdr_at_recall(
    result: OfficialFrontierResult,
    *,
    recall_levels: Sequence[float],
) -> dict[float, float]:
    """Return minimum FDR among complete score-tie prefixes reaching recall."""

    targets: list[float] = []
    for raw in recall_levels:
        target = float(raw)
        if not math.isfinite(target) or not 0.0 <= target <= 1.0:
            raise ValueError(f"invalid recall target: {raw}")
        if target not in targets:
            targets.append(target)
    if not targets:
        raise ValueError("recall_levels must not be empty")

    tp_ids = {int(match.prediction_index) for match in result.trace.matches}
    rows: list[tuple[float, int, bool]] = []
    for image_id in sorted(result.kept_predictions):
        for record in result.kept_predictions[image_id]:
            candidate_id = int(record["source_prediction_index"])
            rows.append((float(record["score"]), candidate_id, candidate_id in tp_ids))
    rows.sort(key=lambda row: (-row[0], row[1]))
    best = {target: 1.0 for target in targets}
    tp = fp = 0
    for _, block in groupby(rows, key=lambda row: row[0]):
        for _, _, is_tp in block:
            if is_tp:
                tp += 1
            else:
                fp += 1
        recall = tp / result.n_gt if result.n_gt else 1.0
        fdr = fp / (tp + fp) if tp + fp else 0.0
        for target in targets:
            if recall >= target:
                best[target] = min(best[target], fdr)
    return best


__all__ = [
    "FixedRiskPoint",
    "OfficialFrontierResult",
    "official_fixed_risk_frontier",
    "min_fdr_at_recall",
    "scan_fixed_risk_points",
]
