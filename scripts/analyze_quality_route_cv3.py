#!/usr/bin/env python3
"""Nested CV3 admission for a quality reranker with coarse identity bypasses.

For every held-out fold, each coarse class independently chooses either the
frozen detector score or the candidate quality score using only the other two
folds.  A quality route is eligible only when it respects the baseline coarse
Recall and FDR guards.  This prevents a large Aircraft gain from masking a
Ship/Vehicle regression and gives every coarse class an exact identity escape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.data.xh_dataset import coarse_name
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    COARSE_ORDER,
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config

T = TypeVar("T")


def _scoped(mapping: dict[int, list[T]], ids: set[int]) -> dict[int, list[T]]:
    return {image_id: list(mapping.get(image_id, ())) for image_id in sorted(ids)}


def _filter_one(
    pred: dict[int, list[dict[str, Any]]], coarse: str, threshold: float
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for row in rows
            if coarse_name(int(row["category_id"])) == coarse
            and float(row["score"]) >= threshold
        ]
        for image_id, rows in pred.items()
    }


def _merge_routes(
    routes: dict[str, dict[int, list[dict[str, Any]]]], image_ids: set[int]
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for coarse in COARSE_ORDER
            for row in routes[coarse].get(image_id, ())
        ]
        for image_id in sorted(image_ids)
    }


def _metrics(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    *,
    protocol: Any,
    latency: float,
) -> dict[str, Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return platform_metrics_payload(
        build_platform_observed_metrics(
            ranking,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
            latency_seconds=latency,
            latency_max_seconds=protocol.latency_max_seconds,
        )
    )


def _coarse_utility(metrics: dict[str, Any], coarse: str) -> float:
    row = metrics["score_payload"]["per_coarse"][coarse]
    return float(row["recall_points"]) + float(row["fdr_points"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--baseline-threshold", type=float, default=0.15)
    parser.add_argument("--grid-step", type=float, default=0.005)
    parser.add_argument("--maximum-recall-drop", type=float, default=0.005)
    parser.add_argument("--maximum-fdr-increase", type=float, default=0.0)
    parser.add_argument("--minimum-training-utility-gain", type=float, default=0.0)
    parser.add_argument(
        "--require-each-selection-fold",
        action="store_true",
        help="Require the quality route to satisfy every selection fold separately.",
    )
    parser.add_argument("--latency-baseline", type=float, default=2.473167)
    parser.add_argument("--latency-candidate", type=float, default=2.623167)
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(row["id"]): int(row["fold"]) for row in raw_gt["images"]}
    fold_ids = {
        fold: {image_id for image_id, value in fold_by_image.items() if value == fold}
        for fold in (0, 1, 2)
    }
    if any(not ids for ids in fold_ids.values()):
        raise ValueError("GT must contain non-empty folds 0/1/2")
    gt = load_coco_ground_truth(args.gt)
    baseline = load_coco_predictions(args.baseline)
    candidate = load_coco_predictions(args.candidate)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    grid = build_threshold_grid(0.001, 0.996, args.grid_step)

    merged_baseline: dict[int, list[dict[str, Any]]] = {}
    merged_candidate: dict[int, list[dict[str, Any]]] = {}
    folds: dict[str, Any] = {}
    for held_out in (0, 1, 2):
        train_ids = set(fold_by_image) - fold_ids[held_out]
        train_gt = _scoped(gt, train_ids)
        held_gt = _scoped(gt, fold_ids[held_out])
        selected: dict[str, dict[str, Any]] = {}
        held_routes: dict[str, dict[int, list[dict[str, Any]]]] = {}
        held_baseline_routes: dict[str, dict[int, list[dict[str, Any]]]] = {}
        for coarse in COARSE_ORDER:
            baseline_train_rows = _filter_one(
                _scoped(baseline, train_ids), coarse, args.baseline_threshold
            )
            baseline_train = _metrics(
                train_gt,
                baseline_train_rows,
                protocol=protocol,
                latency=args.latency_baseline,
            )
            base_row = baseline_train["per_coarse"][coarse]
            base_utility = _coarse_utility(baseline_train, coarse)
            per_selection_fold_baseline: dict[int, dict[str, Any]] = {}
            if args.require_each_selection_fold:
                for selection_fold in sorted({0, 1, 2} - {held_out}):
                    selection_ids = fold_ids[selection_fold]
                    per_selection_fold_baseline[selection_fold] = _metrics(
                        _scoped(gt, selection_ids),
                        _filter_one(
                            _scoped(baseline, selection_ids),
                            coarse,
                            args.baseline_threshold,
                        ),
                        protocol=protocol,
                        latency=args.latency_baseline,
                    )
            eligible: list[tuple[float, float, float, float, dict[str, Any]]] = []
            for threshold in grid:
                rows = _filter_one(_scoped(candidate, train_ids), coarse, threshold)
                metrics = _metrics(
                    train_gt,
                    rows,
                    protocol=protocol,
                    latency=args.latency_candidate,
                )
                coarse_row = metrics["per_coarse"][coarse]
                utility = _coarse_utility(metrics, coarse)
                combined_pass = (
                    float(coarse_row["macro_recall"])
                    >= float(base_row["macro_recall"]) - args.maximum_recall_drop
                    and float(coarse_row["macro_fdr"])
                    <= float(base_row["macro_fdr"]) + args.maximum_fdr_increase
                    and utility
                    > base_utility + args.minimum_training_utility_gain
                )
                each_fold_pass = True
                if args.require_each_selection_fold and combined_pass:
                    for selection_fold, fold_baseline in per_selection_fold_baseline.items():
                        selection_ids = fold_ids[selection_fold]
                        fold_candidate = _metrics(
                            _scoped(gt, selection_ids),
                            _filter_one(
                                _scoped(candidate, selection_ids), coarse, threshold
                            ),
                            protocol=protocol,
                            latency=args.latency_candidate,
                        )
                        fold_base_row = fold_baseline["per_coarse"][coarse]
                        fold_candidate_row = fold_candidate["per_coarse"][coarse]
                        each_fold_pass = each_fold_pass and (
                            float(fold_candidate_row["macro_recall"])
                            >= float(fold_base_row["macro_recall"])
                            - args.maximum_recall_drop
                            and float(fold_candidate_row["macro_fdr"])
                            <= float(fold_base_row["macro_fdr"])
                            + args.maximum_fdr_increase
                            and _coarse_utility(fold_candidate, coarse)
                            > _coarse_utility(fold_baseline, coarse)
                            + args.minimum_training_utility_gain
                        )
                if (
                    combined_pass
                    and each_fold_pass
                ):
                    eligible.append(
                        (
                            utility,
                            float(coarse_row["macro_recall"]),
                            -float(coarse_row["macro_fdr"]),
                            float(threshold),
                            metrics,
                        )
                    )
            if eligible:
                best = max(eligible, key=lambda row: row[:4])
                route = "quality"
                threshold = best[3]
                train_metrics = best[4]
                held_rows = _filter_one(
                    _scoped(candidate, fold_ids[held_out]), coarse, threshold
                )
            else:
                route = "identity"
                threshold = float(args.baseline_threshold)
                train_metrics = baseline_train
                held_rows = _filter_one(
                    _scoped(baseline, fold_ids[held_out]), coarse, threshold
                )
            held_routes[coarse] = held_rows
            held_baseline_routes[coarse] = _filter_one(
                _scoped(baseline, fold_ids[held_out]),
                coarse,
                args.baseline_threshold,
            )
            selected[coarse] = {
                "route": route,
                "threshold": threshold,
                "baseline_training_metrics": baseline_train,
                "selected_training_metrics": train_metrics,
            }
        held_baseline = _merge_routes(held_baseline_routes, fold_ids[held_out])
        held_candidate = _merge_routes(held_routes, fold_ids[held_out])
        merged_baseline.update(held_baseline)
        merged_candidate.update(held_candidate)
        folds[str(held_out)] = {
            "selection_folds": sorted({0, 1, 2} - {held_out}),
            "held_out_fold": held_out,
            "selected_routes": selected,
            "baseline_heldout_metrics": _metrics(
                held_gt,
                held_baseline,
                protocol=protocol,
                latency=args.latency_baseline,
            ),
            "candidate_heldout_metrics": _metrics(
                held_gt,
                held_candidate,
                protocol=protocol,
                latency=args.latency_candidate,
            ),
        }

    baseline_metrics = _metrics(
        gt, merged_baseline, protocol=protocol, latency=args.latency_baseline
    )
    candidate_metrics = _metrics(
        gt, merged_candidate, protocol=protocol, latency=args.latency_candidate
    )
    coarse_drops = {
        coarse: float(baseline_metrics["per_coarse"][coarse]["macro_recall"])
        - float(candidate_metrics["per_coarse"][coarse]["macro_recall"])
        for coarse in COARSE_ORDER
    }
    delta = {
        "gate_recall": candidate_metrics["gate_recall"] - baseline_metrics["gate_recall"],
        "gate_fdr": candidate_metrics["gate_fdr"] - baseline_metrics["gate_fdr"],
        "absolute_score": candidate_metrics["absolute_score"] - baseline_metrics["absolute_score"],
        "latency_seconds": args.latency_candidate - args.latency_baseline,
        "per_coarse_recall_drop": coarse_drops,
        "max_coarse_recall_drop": max(coarse_drops.values()),
    }
    admitted = (
        delta["gate_recall"] >= 0.0
        and delta["gate_fdr"] <= 0.0
        and delta["absolute_score"] > 0.0
        and delta["max_coarse_recall_drop"] <= args.maximum_recall_drop
    )
    payload = {
        "version": "quality_coarse_identity_bypass_nested_cv3_v1",
        "metric_protocol": protocol.metric_protocol,
        "selection_uses_held_out_labels": False,
        "selection_policy": (
            "per coarse and outer fold choose identity or quality on the other two folds; "
            "quality must improve coarse recall+FDR subscore under recall/FDR guards"
        ),
        "baseline_threshold": args.baseline_threshold,
        "grid_step": args.grid_step,
        "maximum_recall_drop": args.maximum_recall_drop,
        "maximum_fdr_increase": args.maximum_fdr_increase,
        "require_each_selection_fold": args.require_each_selection_fold,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_vs_baseline": delta,
        "folds": folds,
        "preliminary_normal_cv3_admission": admitted,
        "formal_module_admission": False,
        "remaining_gates": ["Hard", "Sentinel-A", "Background-100MP", "latency-3090"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
