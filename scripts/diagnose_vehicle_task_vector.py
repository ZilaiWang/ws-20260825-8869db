#!/usr/bin/env python3
"""Diagnose whether a Vehicle task vector improves ranking or only calibration.

This is an exploratory, read-only audit over already materialized OOF predictions.
It never writes checkpoints or changes the frozen admission decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import (
    compute_iou,
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.task_vector_policy import score_from_fine_counts
from rsdet.utils.config import load_config

VEHICLE_ID = 24


def _thresholds(start: float, stop: float, step: float, extras: list[float]) -> list[float]:
    if not 0 <= start <= stop <= 1 or step <= 0:
        raise ValueError("invalid threshold grid")
    count = int(math.floor((stop - start) / step + 1e-9))
    values = [start + index * step for index in range(count + 1)]
    values.extend(extras)
    return sorted({round(value, 9) for value in values if start <= value <= stop})


def _filter_rows(
    rows: dict[int, list[dict[str, Any]]],
    *,
    default_threshold: float,
    vehicle_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for row in items
            if float(row["score"])
            >= (vehicle_threshold if int(row["category_id"]) == VEHICLE_ID else default_threshold)
        ]
        for image_id, items in rows.items()
    }


def _compose_vehicle_only(
    baseline: dict[int, list[dict[str, Any]]],
    candidate: dict[int, list[dict[str, Any]]],
    *,
    default_threshold: float,
    vehicle_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    image_ids = sorted(set(baseline) | set(candidate))
    return {
        image_id: [
            *(
                row
                for row in baseline.get(image_id, [])
                if int(row["category_id"]) != VEHICLE_ID
                and float(row["score"]) >= default_threshold
            ),
            *(
                row
                for row in candidate.get(image_id, [])
                if int(row["category_id"]) == VEHICLE_ID
                and float(row["score"]) >= vehicle_threshold
            ),
        ]
        for image_id in image_ids
    }


def _merge(mappings: list[dict[int, list[dict[str, Any]]]]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for mapping in mappings:
        overlap = set(output) & set(mapping)
        if overlap:
            raise ValueError(f"fold image IDs overlap: {sorted(overlap)[:5]}")
        output.update(mapping)
    return output


def _summary(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    protocol: Any,
) -> dict[str, Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    platform = platform_metrics_payload(
        build_platform_observed_metrics(ranking, latency_seconds=0.0)
    )
    vehicle = ranking.per_fine[VEHICLE_ID]
    return {
        "absolute_score": platform["absolute_score"],
        "vehicle": {
            "tp": vehicle.tp,
            "fp": vehicle.fp,
            "fn": vehicle.fn,
            "recall": vehicle.recall,
            "fdr": vehicle.fdr,
            "predictions": vehicle.tp + vehicle.fp,
        },
        "_fine_counts": {
            int(fine_id): {"tp": row.tp, "fp": row.fp, "fn": row.fn}
            for fine_id, row in ranking.per_fine.items()
        },
    }


def _summary_from_counts(
    counts: dict[int, dict[str, int]], category_mapping: dict[int, str]
) -> dict[str, Any]:
    score = score_from_fine_counts(counts, category_mapping)
    vehicle = counts[VEHICLE_ID]
    tp, fp, fn = vehicle["tp"], vehicle["fp"], vehicle["fn"]
    return {
        "absolute_score": score["total_score"],
        "vehicle": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else 1.0,
            "fdr": fp / (tp + fp) if tp + fp else 0.0,
            "predictions": tp + fp,
        },
        "_fine_counts": counts,
    }


def _public(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _combine_rows(
    rows: list[dict[str, Any]], category_mapping: dict[int, str]
) -> dict[str, Any]:
    counts: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for row in rows:
        for fine_id, values in row["_fine_counts"].items():
            for name in ("tp", "fp", "fn"):
                counts[int(fine_id)][name] += int(values[name])
    return _summary_from_counts(dict(counts), category_mapping)


def _policy_key(row: dict[str, Any], *, candidate: bool) -> tuple[float, float, float]:
    # Stable conservative tie-break: score, then smaller alpha, then threshold
    # closest to the frozen deployment threshold.
    return (
        float(row["absolute_score"]),
        -float(row["alpha"]) if candidate else 0.0,
        -abs(float(row["vehicle_threshold"]) - float(row["deployment_threshold"])),
    )


def _single_fold_policy(
    fold: int,
    fold_gt: dict[int, dict[int, list[dict[str, Any]]]],
    fold_candidates: dict[int, dict[float, dict[int, list[dict[str, Any]]]]],
    fold_nonvehicle_counts: dict[int, dict[int, dict[str, int]]],
    protocol: Any,
    *,
    alpha: float,
    deployment_threshold: float,
    vehicle_threshold: float,
) -> dict[str, Any]:
    vehicle_pred = _vehicle_only(
        fold_candidates[fold][alpha], vehicle_threshold
    )
    trace = _vehicle_trace(fold_gt[fold], vehicle_pred)
    counts = {
        fine_id: dict(row) for fine_id, row in fold_nonvehicle_counts[fold].items()
    }
    counts[VEHICLE_ID] = {
        "tp": len(trace.matches),
        "fp": len(trace.unmatched_predictions),
        "fn": len(trace.unmatched_ground_truths),
    }
    result = _summary_from_counts(counts, protocol.category_mapping)
    result.update(
        {
            "alpha": alpha,
            "vehicle_threshold": vehicle_threshold,
            "deployment_threshold": deployment_threshold,
        }
    )
    return result


def _nested_policy_cv(
    cache: dict[tuple[int, float, float], dict[str, Any]],
    alphas: list[float],
    thresholds: list[float],
    category_mapping: dict[int, str],
    deployment_threshold: float,
) -> dict[str, Any]:
    outer: dict[str, Any] = {}
    for heldout in range(3):
        training = [fold for fold in range(3) if fold != heldout]
        baseline_grid = []
        for threshold in thresholds:
            row = _combine_rows(
                [cache[(fold, 0.0, threshold)] for fold in training], category_mapping
            )
            row.update(
                {
                    "alpha": 0.0,
                    "vehicle_threshold": threshold,
                    "deployment_threshold": deployment_threshold,
                }
            )
            baseline_grid.append(row)
        candidate_grid = []
        for alpha in alphas:
            for threshold in thresholds:
                row = _combine_rows(
                    [cache[(fold, alpha, threshold)] for fold in training],
                    category_mapping,
                )
                row.update(
                    {
                        "alpha": alpha,
                        "vehicle_threshold": threshold,
                        "deployment_threshold": deployment_threshold,
                    }
                )
                candidate_grid.append(row)
        selected_baseline = max(baseline_grid, key=lambda row: _policy_key(row, candidate=False))
        selected_candidate = max(candidate_grid, key=lambda row: _policy_key(row, candidate=True))
        baseline_test = cache[
            (heldout, 0.0, float(selected_baseline["vehicle_threshold"]))
        ]
        candidate_test = cache[
            (
                heldout,
                float(selected_candidate["alpha"]),
                float(selected_candidate["vehicle_threshold"]),
            )
        ]
        outer[str(heldout)] = {
            "training_selected_baseline": _public(selected_baseline),
            "training_selected_candidate": _public(selected_candidate),
            "heldout_baseline": _public(baseline_test),
            "heldout_candidate": _public(candidate_test),
            "heldout_score_delta": candidate_test["absolute_score"]
            - baseline_test["absolute_score"],
        }
    deltas = [row["heldout_score_delta"] for row in outer.values()]
    return {
        "outer_folds": outer,
        "score_delta_sum": sum(deltas),
        "score_delta_mean": sum(deltas) / len(deltas),
        "every_fold_positive": all(delta > 0 for delta in deltas),
    }


def _frontier_diagnostics(
    cache: dict[tuple[int, float, float], dict[str, Any]],
    alphas: list[float],
    thresholds: list[float],
    category_mapping: dict[int, str],
    deployment_threshold: float,
    official_vehicle_recall: float,
    official_vehicle_fdr: float,
) -> dict[str, Any]:
    partitions = {
        "aggregate": [0, 1, 2],
        "fold_0": [0],
        "fold_1": [1],
        "fold_2": [2],
    }
    output: dict[str, Any] = {}
    for name, folds in partitions.items():
        baseline_reference = _combine_rows(
            [cache[(fold, 0.0, deployment_threshold)] for fold in folds],
            category_mapping,
        )
        baseline_frontier = [
            _combine_rows(
                [cache[(fold, 0.0, threshold)] for fold in folds], category_mapping
            )
            for threshold in thresholds
        ]
        by_alpha: dict[str, Any] = {}
        for alpha in alphas:
            candidate_frontier = []
            for threshold in thresholds:
                row = _combine_rows(
                    [cache[(fold, alpha, threshold)] for fold in folds], category_mapping
                )
                row["vehicle_threshold"] = threshold
                candidate_frontier.append(row)
            fp_budget = baseline_reference["vehicle"]["fp"]
            candidates_under_fp_budget = [
                row for row in candidate_frontier if row["vehicle"]["fp"] <= fp_budget
            ]
            best_under_fp = max(
                candidates_under_fp_budget,
                key=lambda row: (
                    row["vehicle"]["tp"],
                    row["absolute_score"],
                    -abs(row["vehicle_threshold"] - deployment_threshold),
                ),
            )
            non_dominated = []
            for candidate in candidate_frontier:
                vehicle = candidate["vehicle"]
                dominated = any(
                    baseline["vehicle"]["tp"] >= vehicle["tp"]
                    and baseline["vehicle"]["fp"] <= vehicle["fp"]
                    and (
                        baseline["vehicle"]["tp"] > vehicle["tp"]
                        or baseline["vehicle"]["fp"] < vehicle["fp"]
                    )
                    for baseline in baseline_frontier
                )
                if not dominated:
                    non_dominated.append(candidate)
            closest_official_recall = min(
                candidate_frontier,
                key=lambda row: (
                    abs(row["vehicle"]["recall"] - official_vehicle_recall),
                    row["vehicle"]["fdr"],
                ),
            )
            closest_official_joint = min(
                candidate_frontier,
                key=lambda row: (
                    abs(row["vehicle"]["recall"] - official_vehicle_recall)
                    + abs(row["vehicle"]["fdr"] - official_vehicle_fdr),
                    abs(row["vehicle_threshold"] - deployment_threshold),
                ),
            )
            by_alpha[str(alpha)] = {
                "best_at_or_below_baseline_deployment_fp": _public(best_under_fp),
                "tp_gain_at_same_fp_budget": (
                    best_under_fp["vehicle"]["tp"]
                    - baseline_reference["vehicle"]["tp"]
                ),
                "candidate_grid_points_not_dominated_by_baseline_frontier": len(
                    non_dominated
                ),
                "best_non_dominated_by_score": (
                    _public(max(non_dominated, key=lambda row: row["absolute_score"]))
                    if non_dominated
                    else None
                ),
                "maximum_recall_on_materialized_predictions": _public(
                    max(candidate_frontier, key=lambda row: row["vehicle"]["recall"])
                ),
                "closest_to_official_vehicle_recall": _public(closest_official_recall),
                "closest_to_official_vehicle_recall_fdr_joint": _public(
                    closest_official_joint
                ),
            }
        output[name] = {
            "baseline_deployment_reference": _public(baseline_reference),
            "by_alpha": by_alpha,
        }
    return output


def _vehicle_only(rows: dict[int, list[dict[str, Any]]], threshold: float) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for row in items
            if int(row["category_id"]) == VEHICLE_ID and float(row["score"]) >= threshold
        ]
        for image_id, items in rows.items()
    }


def _vehicle_trace(
    gt: dict[int, list[dict[str, Any]]], pred: dict[int, list[dict[str, Any]]]
) -> Any:
    filtered_gt = {
        image_id: [row for row in items if int(row["category_id"]) == VEHICLE_ID]
        for image_id, items in gt.items()
    }
    _, trace = evaluate_predictions_with_trace(
        filtered_gt,
        pred,
        class_names=["vehicle"],
        category_mapping={VEHICLE_ID: "vehicle"},
        iou_thresholds={"vehicle": 0.35},
    )
    return trace


def _match_prediction_boxes(
    baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], iou_threshold: float = 0.98
) -> tuple[set[int], set[int]]:
    used: set[int] = set()
    retained: set[int] = set()
    for candidate_index, candidate_row in enumerate(candidate):
        options = [
            (compute_iou(candidate_row["bbox_xyxy"], row["bbox_xyxy"]), index)
            for index, row in enumerate(baseline)
            if index not in used
        ]
        if options:
            best_iou, best_index = max(options)
            if best_iou >= iou_threshold:
                used.add(best_index)
                retained.add(candidate_index)
    return retained, used


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def value(q: float) -> float:
        position = q * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {"min": ordered[0], "p25": value(0.25), "median": value(0.5), "p75": value(0.75), "max": ordered[-1]}


def _source_domain(file_name: str) -> str:
    name = Path(file_name).name
    if name.startswith("fsc_"):
        return "vehicle_source"
    if name.startswith("MAR20_"):
        return "aircraft_source"
    return "ship_source"


def _error_structure(
    fold_gt: dict[int, dict[int, list[dict[str, Any]]]],
    fold_base: dict[int, dict[int, list[dict[str, Any]]]],
    fold_candidates: dict[int, dict[float, dict[int, list[dict[str, Any]]]]],
    image_meta: dict[int, dict[str, Any]],
    group_of_image: dict[int, str],
    *,
    alpha: float,
    threshold: float,
) -> dict[str, Any]:
    counts = Counter()
    scores: dict[str, list[float]] = defaultdict(list)
    relative_areas: dict[str, list[float]] = defaultdict(list)
    groups: dict[str, Counter[str]] = defaultdict(Counter)
    domains: dict[str, Counter[str]] = defaultdict(Counter)
    fold_rows: dict[str, Any] = {}
    rescued_gt_total = 0
    for fold in range(3):
        gt = fold_gt[fold]
        baseline = _vehicle_only(fold_base[fold], threshold)
        candidate = _vehicle_only(fold_candidates[fold][alpha], threshold)
        base_trace = _vehicle_trace(gt, baseline)
        candidate_trace = _vehicle_trace(gt, candidate)
        base_matched_gt = {
            (event.image_id, event.ground_truth_index) for event in base_trace.matches
        }
        candidate_matched_gt = {
            (event.image_id, event.ground_truth_index) for event in candidate_trace.matches
        }
        rescued = candidate_matched_gt - base_matched_gt
        rescued_gt_total += len(rescued)
        match_keys = {
            (event.image_id, event.prediction_index): event for event in candidate_trace.matches
        }
        fp_keys = {
            (event.image_id, event.prediction_index) for event in candidate_trace.unmatched_predictions
        }
        fold_added = Counter()
        for image_id in sorted(set(baseline) | set(candidate)):
            retained, matched_base = _match_prediction_boxes(
                baseline.get(image_id, []), candidate.get(image_id, [])
            )
            counts["retained"] += len(retained)
            counts["removed"] += len(baseline.get(image_id, [])) - len(matched_base)
            for index, row in enumerate(candidate.get(image_id, [])):
                if index in retained:
                    continue
                key = (image_id, index)
                meta = image_meta[image_id]
                group = group_of_image[image_id]
                domain = _source_domain(str(meta["file_name"]))
                if key in match_keys:
                    label = "added_tp"
                elif key in fp_keys:
                    vehicle_gt = [
                        gt_row
                        for gt_row in gt.get(image_id, [])
                        if int(gt_row["category_id"]) == VEHICLE_ID
                    ]
                    other_gt = [
                        gt_row
                        for gt_row in gt.get(image_id, [])
                        if int(gt_row["category_id"]) != VEHICLE_ID
                    ]
                    max_vehicle_iou = max(
                        (compute_iou(row["bbox_xyxy"], gt_row["bbox_xyxy"]) for gt_row in vehicle_gt),
                        default=0.0,
                    )
                    max_other_iou = max(
                        (compute_iou(row["bbox_xyxy"], gt_row["bbox_xyxy"]) for gt_row in other_gt),
                        default=0.0,
                    )
                    if max_vehicle_iou >= 0.35:
                        label = "added_fp_duplicate"
                    elif max_other_iou >= 0.5:
                        label = "added_fp_other_class_overlap"
                    elif max_vehicle_iou >= 0.1:
                        label = "added_fp_localization"
                    else:
                        label = "added_fp_background"
                else:
                    raise AssertionError(f"prediction missing from trace: {key}")
                counts[label] += 1
                fold_added[label] += 1
                scores[label].append(float(row["score"]))
                width = max(0.0, float(row["bbox_xyxy"][2]) - float(row["bbox_xyxy"][0]))
                height = max(0.0, float(row["bbox_xyxy"][3]) - float(row["bbox_xyxy"][1]))
                relative_areas[label].append(
                    width * height / (float(meta["width"]) * float(meta["height"]))
                )
                groups[label][group] += 1
                domains[label][domain] += 1
        fold_rows[str(fold)] = {
            "baseline_predictions": sum(map(len, baseline.values())),
            "candidate_predictions": sum(map(len, candidate.values())),
            "rescued_ground_truths": len(rescued),
            "added_breakdown": dict(fold_added),
        }
    added_tp = counts["added_tp"]
    added_fp = sum(value for key, value in counts.items() if key.startswith("added_fp_"))
    return {
        "alpha": alpha,
        "threshold": threshold,
        "counts": dict(counts),
        "incremental_precision": added_tp / (added_tp + added_fp) if added_tp + added_fp else None,
        "rescued_ground_truths": rescued_gt_total,
        "score_quantiles": {key: _quantiles(value) for key, value in scores.items()},
        "relative_box_area_quantiles": {
            key: _quantiles(value) for key, value in relative_areas.items()
        },
        "top_groups": {
            key: counter.most_common(10) for key, counter in groups.items()
        },
        "source_domains": {key: dict(counter) for key, counter in domains.items()},
        "per_fold": fold_rows,
    }


def _proposal_reservoir(
    fold_gt: dict[int, dict[int, list[dict[str, Any]]]],
    fold_candidates: dict[int, dict[float, dict[int, list[dict[str, Any]]]]],
    alphas: list[float],
    *,
    deployment_threshold: float,
    low_threshold: float,
) -> dict[str, Any]:
    gt = _merge([fold_gt[fold] for fold in range(3)])
    output: dict[str, Any] = {}
    for alpha in alphas:
        raw = _merge([fold_candidates[fold][alpha] for fold in range(3)])
        deployment = _vehicle_only(raw, deployment_threshold)
        low = _vehicle_only(raw, low_threshold)
        deployment_trace = _vehicle_trace(gt, deployment)
        low_trace = _vehicle_trace(gt, low)
        deployment_gt = {
            (event.image_id, event.ground_truth_index)
            for event in deployment_trace.matches
        }
        low_gt = {
            (event.image_id, event.ground_truth_index) for event in low_trace.matches
        }
        low_match_by_gt = {
            (event.image_id, event.ground_truth_index): event for event in low_trace.matches
        }
        recoverable = low_gt - deployment_gt
        recoverable_scores = [low_match_by_gt[key].score for key in recoverable]
        deployment_scores = [event.score for event in deployment_trace.matches]
        proposal_miss = {
            (event.image_id, event.ground_truth_index)
            for event in low_trace.unmatched_ground_truths
        }
        output[str(alpha)] = {
            "deployment": {
                "tp": len(deployment_trace.matches),
                "fp": len(deployment_trace.unmatched_predictions),
                "fn": len(deployment_trace.unmatched_ground_truths),
            },
            "low_floor": {
                "threshold": low_threshold,
                "tp": len(low_trace.matches),
                "fp": len(low_trace.unmatched_predictions),
                "fn": len(low_trace.unmatched_ground_truths),
            },
            "deployment_true_positive_score_quantiles": _quantiles(deployment_scores),
            "recoverable_below_deployment_threshold": len(recoverable),
            "recoverable_match_score_quantiles": _quantiles(recoverable_scores),
            "not_covered_by_any_materialized_vehicle_box": len(proposal_miss),
            "fraction_of_deployment_fn_that_is_low_score_not_geometry": (
                len(recoverable) / len(deployment_trace.unmatched_ground_truths)
                if deployment_trace.unmatched_ground_truths
                else 0.0
            ),
        }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deployment-threshold", type=float, default=0.536)
    parser.add_argument("--historical-threshold", type=float, default=0.546)
    parser.add_argument("--threshold-start", type=float, default=0.001)
    parser.add_argument("--threshold-stop", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument("--official-vehicle-recall", type=float, default=0.831579)
    parser.add_argument("--official-vehicle-fdr", type=float, default=0.21)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = sorted(manifest["folds"], key=lambda row: int(row["fold"]))
    if [int(row["fold"]) for row in folds] != [0, 1, 2]:
        raise ValueError("manifest must contain folds 0, 1, 2")
    alphas = sorted(float(value) for value in manifest["alphas"])
    if 0.0 not in alphas:
        raise ValueError("alpha=0 baseline is required")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    thresholds = _thresholds(
        args.threshold_start,
        args.threshold_stop,
        args.threshold_step,
        [args.deployment_threshold, args.historical_threshold],
    )

    fold_gt: dict[int, dict[int, list[dict[str, Any]]]] = {}
    fold_base: dict[int, dict[int, list[dict[str, Any]]]] = {}
    fold_candidates: dict[int, dict[float, dict[int, list[dict[str, Any]]]]] = {}
    image_meta: dict[int, dict[str, Any]] = {}
    for row in folds:
        fold = int(row["fold"])
        gt_path = Path(row["gt"])
        gt_payload = json.loads(gt_path.read_text(encoding="utf-8"))
        fold_gt[fold] = load_coco_ground_truth(gt_path)
        for image in gt_payload["images"]:
            image_id = int(image["id"])
            if image_id in image_meta:
                raise ValueError(f"image id overlaps across folds: {image_id}")
            image_meta[image_id] = image
        fold_base[fold] = load_coco_predictions(Path(row["baseline"]))
        fold_candidates[fold] = {
            float(alpha): load_coco_predictions(Path(path))
            for alpha, path in row["candidates"].items()
        }

    group_payload = json.loads(args.image_groups.read_text(encoding="utf-8"))
    group_of_image = {
        int(row["image_id"]): str(row["group_id"]) for row in group_payload["samples"]
    }
    if set(image_meta) - set(group_of_image):
        raise ValueError("image group manifest is incomplete")

    cache: dict[tuple[int, float, float], dict[str, Any]] = {}
    fold_nonvehicle_counts: dict[int, dict[int, dict[str, int]]] = {}
    for fold in range(3):
        baseline_at_deployment = _compose_vehicle_only(
            fold_base[fold],
            fold_base[fold],
            default_threshold=args.deployment_threshold,
            vehicle_threshold=args.deployment_threshold,
        )
        baseline_summary = _summary(fold_gt[fold], baseline_at_deployment, protocol)
        fold_nonvehicle_counts[fold] = {
            int(fine_id): dict(row)
            for fine_id, row in baseline_summary["_fine_counts"].items()
            if int(fine_id) != VEHICLE_ID
        }
    for fold in range(3):
        for alpha in alphas:
            for threshold in thresholds:
                cache[(fold, alpha, threshold)] = _single_fold_policy(
                    fold,
                    fold_gt,
                    fold_candidates,
                    fold_nonvehicle_counts,
                    protocol,
                    alpha=alpha,
                    deployment_threshold=args.deployment_threshold,
                    vehicle_threshold=threshold,
                )

    fixed_thresholds: dict[str, Any] = {}
    for threshold in (args.deployment_threshold, args.historical_threshold):
        fixed_thresholds[str(threshold)] = {
            str(alpha): {
                "aggregate": _public(
                    _combine_rows(
                        [cache[(fold, alpha, threshold)] for fold in range(3)],
                        protocol.category_mapping,
                    )
                ),
                "per_fold": {
                    str(fold): _public(cache[(fold, alpha, threshold)])
                    for fold in range(3)
                },
            }
            for alpha in alphas
        }

    support = Counter()
    support_groups: set[str] = set()
    for fold, gt in fold_gt.items():
        support[f"fold_{fold}_images"] = len(gt)
        vehicle_images = {
            image_id
            for image_id, rows in gt.items()
            if any(int(row["category_id"]) == VEHICLE_ID for row in rows)
        }
        vehicle_gt = sum(
            int(row["category_id"]) == VEHICLE_ID for rows in gt.values() for row in rows
        )
        support[f"fold_{fold}_vehicle_images"] = len(vehicle_images)
        support[f"fold_{fold}_vehicle_gt"] = vehicle_gt
        support_groups.update(group_of_image[image_id] for image_id in vehicle_images)

    payload = {
        "status": "complete",
        "protocol": "vehicle_task_vector_error_structure_v1_exploratory",
        "does_not_change_frozen_attempt4_decision": True,
        "deployment_threshold": args.deployment_threshold,
        "historical_attempt4_threshold": args.historical_threshold,
        "threshold_grid": thresholds,
        "dataset_support": {
            **dict(support),
            "vehicle_source_group_count": len(support_groups),
            "official_vehicle_gt_reference": 95,
            "official_p40_vehicle_recall_reference": args.official_vehicle_recall,
            "official_p40_vehicle_fdr_reference": args.official_vehicle_fdr,
        },
        "fixed_thresholds": fixed_thresholds,
        "nested_policy_cv": _nested_policy_cv(
            cache,
            alphas,
            thresholds,
            protocol.category_mapping,
            args.deployment_threshold,
        ),
        "frontier_diagnostics": _frontier_diagnostics(
            cache,
            alphas,
            thresholds,
            protocol.category_mapping,
            args.deployment_threshold,
            args.official_vehicle_recall,
            args.official_vehicle_fdr,
        ),
        "proposal_reservoir": _proposal_reservoir(
            fold_gt,
            fold_candidates,
            alphas,
            deployment_threshold=args.deployment_threshold,
            low_threshold=args.threshold_start,
        ),
        "alpha_0p5_error_structure_at_deployment_threshold": _error_structure(
            fold_gt,
            fold_base,
            fold_candidates,
            image_meta,
            group_of_image,
            alpha=0.5,
            threshold=args.deployment_threshold,
        ),
        "alpha_0p5_error_structure_at_historical_threshold": _error_structure(
            fold_gt,
            fold_base,
            fold_candidates,
            image_meta,
            group_of_image,
            alpha=0.5,
            threshold=args.historical_threshold,
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
