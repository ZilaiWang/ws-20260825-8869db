#!/usr/bin/env python3
"""Nested outer-CV3 admission replay for the sparse Ship quality route."""

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


def _filter(
    pred: dict[int, list[dict[str, Any]]], thresholds: dict[str, float]
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for row in rows
            if float(row["score"]) >= thresholds[coarse_name(int(row["category_id"]))]
        ]
        for image_id, rows in pred.items()
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


def _select_ship_threshold(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    *,
    protocol: Any,
    latency: float,
    fixed_non_ship: dict[str, float],
    grid: list[float],
    target_ship_fdr: float,
) -> tuple[float, dict[str, Any]]:
    rows: list[tuple[float, dict[str, Any]]] = []
    for threshold in grid:
        thresholds = {**fixed_non_ship, "ship": float(threshold)}
        metrics = _metrics(
            gt, _filter(pred, thresholds), protocol=protocol, latency=latency
        )
        rows.append((float(threshold), metrics))
    if not rows:
        raise RuntimeError("Ship threshold grid produced no result")
    feasible = [
        row
        for row in rows
        if float(row[1]["per_coarse"]["ship"]["macro_fdr"])
        <= float(target_ship_fdr)
    ]
    candidates = feasible or rows
    if feasible:
        selected = max(
            candidates,
            key=lambda row: (
                float(row[1]["per_coarse"]["ship"]["macro_recall"]),
                float(row[1]["absolute_score"]),
                -float(row[1]["per_coarse"]["ship"]["macro_fdr"]),
                float(row[0]),
            ),
        )
    else:
        selected = min(
            candidates,
            key=lambda row: (
                float(row[1]["per_coarse"]["ship"]["macro_fdr"]),
                -float(row[1]["per_coarse"]["ship"]["macro_recall"]),
                -float(row[0]),
            ),
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--aircraft-threshold", type=float, default=0.301)
    parser.add_argument("--vehicle-threshold", type=float, default=0.366)
    parser.add_argument("--ship-baseline-threshold", type=float, default=0.150)
    parser.add_argument("--grid-step", type=float, default=0.005)
    parser.add_argument("--latency-baseline", type=float, default=2.5)
    parser.add_argument("--latency-candidate", type=float, default=2.55)
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
    fixed = {
        "aircraft": float(args.aircraft_threshold),
        "vehicle": float(args.vehicle_threshold),
    }
    grid = build_threshold_grid(0.001, 0.996, args.grid_step)
    merged_baseline: dict[int, list[dict[str, Any]]] = {}
    merged_candidate: dict[int, list[dict[str, Any]]] = {}
    folds: dict[str, Any] = {}
    for held_out in (0, 1, 2):
        train_ids = set(fold_by_image) - fold_ids[held_out]
        train_gt = _scoped(gt, train_ids)
        baseline_threshold = float(args.ship_baseline_threshold)
        baseline_train_metrics = _metrics(
            train_gt,
            _filter(
                _scoped(baseline, train_ids),
                {**fixed, "ship": baseline_threshold},
            ),
            protocol=protocol,
            latency=args.latency_baseline,
        )
        target_ship_fdr = float(
            baseline_train_metrics["per_coarse"]["ship"]["macro_fdr"]
        )
        candidate_threshold, candidate_train_metrics = _select_ship_threshold(
            train_gt,
            _scoped(candidate, train_ids),
            protocol=protocol,
            latency=args.latency_candidate,
            fixed_non_ship=fixed,
            grid=grid,
            target_ship_fdr=target_ship_fdr,
        )
        held_gt = _scoped(gt, fold_ids[held_out])
        held_baseline = _filter(
            _scoped(baseline, fold_ids[held_out]),
            {**fixed, "ship": baseline_threshold},
        )
        held_candidate = _filter(
            _scoped(candidate, fold_ids[held_out]),
            {**fixed, "ship": candidate_threshold},
        )
        merged_baseline.update(held_baseline)
        merged_candidate.update(held_candidate)
        folds[str(held_out)] = {
            "selection_folds": sorted({0, 1, 2} - {held_out}),
            "held_out_fold": held_out,
            "baseline_ship_threshold": baseline_threshold,
            "candidate_ship_threshold": candidate_threshold,
            "candidate_training_target_ship_fdr": target_ship_fdr,
            "baseline_training_metrics": baseline_train_metrics,
            "candidate_training_metrics": candidate_train_metrics,
            "baseline_heldout_metrics": _metrics(
                held_gt, held_baseline, protocol=protocol, latency=args.latency_baseline
            ),
            "candidate_heldout_metrics": _metrics(
                held_gt, held_candidate, protocol=protocol, latency=args.latency_candidate
            ),
        }
    baseline_metrics = _metrics(
        gt, merged_baseline, protocol=protocol, latency=args.latency_baseline
    )
    candidate_metrics = _metrics(
        gt, merged_candidate, protocol=protocol, latency=args.latency_candidate
    )
    coarse_drops = {
        name: baseline_metrics["per_coarse"][name]["macro_recall"]
        - candidate_metrics["per_coarse"][name]["macro_recall"]
        for name in COARSE_ORDER
    }
    delta = {
        "gate_recall": candidate_metrics["gate_recall"] - baseline_metrics["gate_recall"],
        "gate_fdr": candidate_metrics["gate_fdr"] - baseline_metrics["gate_fdr"],
        "absolute_score": candidate_metrics["absolute_score"]
        - baseline_metrics["absolute_score"],
        "latency_seconds": args.latency_candidate - args.latency_baseline,
        "max_coarse_recall_drop": max(coarse_drops.values()),
        "per_coarse_recall_drop": coarse_drops,
    }
    preliminary_admission = (
        delta["gate_recall"] >= 0.005
        and delta["gate_fdr"] <= 0.0
        and delta["absolute_score"] > 0.0
        and delta["max_coarse_recall_drop"] <= 0.005
    )
    payload = {
        "version": "ship_quality_nested_outer_cv3_v1",
        "metric_protocol": protocol.metric_protocol,
        "selection_uses_held_out_labels": False,
        "selection_policy": (
            "baseline ship threshold frozen; candidate maximizes training-fold Ship "
            "macro Recall under the training-fold baseline Ship macro FDR"
        ),
        "ship_threshold_grid_step": args.grid_step,
        "non_ship_thresholds_frozen": fixed,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_vs_baseline": delta,
        "folds": folds,
        "preliminary_normal_cv3_admission": preliminary_admission,
        "formal_module_admission": False,
        "remaining_gates": ["Hard", "Sentinel-A", "Background-100MP", "latency-3090"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
