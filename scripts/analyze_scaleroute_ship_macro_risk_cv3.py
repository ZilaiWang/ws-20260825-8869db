#!/usr/bin/env python3
"""Fit Ship-only hierarchical thresholds inside a frozen ScaleRoute R2 CV3."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.hierarchical_thresholds import build_hierarchical_curves
from rsdet.evaluation.macro_risk_v2 import fit_macro_risk_v2, group_bootstrap_admission
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config

SHIP_LABELS = frozenset(range(4))
AIRCRAFT_LABELS = frozenset(range(4, 24))
VEHICLE_LABEL = 24


def _frontier_thresholds(frontier: dict[str, Any], level: str) -> dict[int, float]:
    raw = frontier["frontiers"][level]["crossfit_thresholds"]
    result = {int(fold): float(value) for fold, value in raw.items()}
    if set(result) != {0, 1, 2}:
        raise ValueError("frontier thresholds must cover folds 0, 1 and 2")
    return result


def compose_fold(
    *,
    image_ids: set[int],
    primary: dict[int, list[dict[str, Any]]],
    expert: dict[int, list[dict[str, Any]]],
    primary_threshold: float,
    expert_threshold: float,
    ship_thresholds: dict[int, float] | None,
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for image_id in sorted(image_ids):
        rows: list[dict[str, Any]] = []
        for row in primary.get(image_id, []):
            label = int(row["category_id"])
            score = float(row["score"])
            if label in SHIP_LABELS:
                threshold = (
                    primary_threshold
                    if ship_thresholds is None
                    else float(ship_thresholds[label])
                )
                if score >= threshold:
                    rows.append(row)
            elif label in AIRCRAFT_LABELS and score >= primary_threshold:
                rows.append(row)
        rows.extend(
            row
            for row in expert.get(image_id, [])
            if int(row["category_id"]) == VEHICLE_LABEL
            and float(row["score"]) >= expert_threshold
        )
        output[image_id] = rows
    return output


def _platform(
    gt: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    *,
    protocol: Any,
    latency: float,
) -> dict[str, Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        predictions,
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--primary-pred", type=Path, required=True)
    parser.add_argument("--expert-pred", type=Path, required=True)
    parser.add_argument("--primary-frontier", type=Path, required=True)
    parser.add_argument("--expert-frontier", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fdr-level", default="0.150")
    parser.add_argument("--target-fdr-ship", type=float, default=0.15)
    parser.add_argument("--latency", type=float, default=8.0)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(row["id"]): int(row["fold"]) for row in raw_gt["images"]}
    fold_ids = {
        fold: {image_id for image_id, value in fold_by_image.items() if value == fold}
        for fold in (0, 1, 2)
    }
    if any(not image_ids for image_ids in fold_ids.values()):
        raise ValueError("GT must contain non-empty folds 0, 1 and 2")
    with args.group_map.open(encoding="utf-8", newline="") as handle:
        group_by_image = {
            int(row["image_id"]): str(row["group_id"]) for row in csv.DictReader(handle)
        }
    if set(fold_by_image) - set(group_by_image):
        raise ValueError("group map does not cover all GT images")

    gt = load_coco_ground_truth(args.gt)
    primary = load_coco_predictions(args.primary_pred)
    expert = load_coco_predictions(args.expert_pred)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    primary_frontier = json.loads(args.primary_frontier.read_text(encoding="utf-8"))
    expert_frontier = json.loads(args.expert_frontier.read_text(encoding="utf-8"))
    primary_thresholds = _frontier_thresholds(primary_frontier, args.fdr_level)
    expert_thresholds = _frontier_thresholds(expert_frontier, args.fdr_level)
    threshold_grid = build_threshold_grid(0.001, 0.996, 0.005)

    baseline: dict[int, list[dict[str, Any]]] = {}
    candidate: dict[int, list[dict[str, Any]]] = {}
    fold_audit: dict[str, Any] = {}
    for held_out in (0, 1, 2):
        train_ids = set(fold_by_image) - fold_ids[held_out]
        group_sets: dict[int, set[str]] = defaultdict(set)
        for image_id in train_ids:
            for item in gt.get(image_id, []):
                group_sets[int(item["category_id"])].add(group_by_image[image_id])
        curves = build_hierarchical_curves(
            gt,
            primary,
            image_ids=train_ids,
            thresholds=threshold_grid,
            protocol=protocol,
        )
        fit = fit_macro_risk_v2(
            curves,
            protocol=protocol,
            group_counts_by_fine={key: len(value) for key, value in group_sets.items()},
            target_fdr_by_coarse={
                "ship": args.target_fdr_ship,
                "aircraft": 0.15,
                "vehicle": 0.15,
            },
        )
        ship_thresholds = {
            label: float(fit["fine_thresholds"][label]) for label in SHIP_LABELS
        }
        baseline.update(
            compose_fold(
                image_ids=fold_ids[held_out],
                primary=primary,
                expert=expert,
                primary_threshold=primary_thresholds[held_out],
                expert_threshold=expert_thresholds[held_out],
                ship_thresholds=None,
            )
        )
        candidate.update(
            compose_fold(
                image_ids=fold_ids[held_out],
                primary=primary,
                expert=expert,
                primary_threshold=primary_thresholds[held_out],
                expert_threshold=expert_thresholds[held_out],
                ship_thresholds=ship_thresholds,
            )
        )
        fold_audit[str(held_out)] = {
            "train_image_count": len(train_ids),
            "held_out_image_count": len(fold_ids[held_out]),
            "primary_threshold": primary_thresholds[held_out],
            "expert_threshold": expert_thresholds[held_out],
            "ship_thresholds": ship_thresholds,
            "ship_fine_audit": {
                str(label): fit["fine_audit"][label] for label in sorted(SHIP_LABELS)
            },
        }

    baseline_metrics = _platform(gt, baseline, protocol=protocol, latency=args.latency)
    candidate_metrics = _platform(gt, candidate, protocol=protocol, latency=args.latency)
    baseline_robust = group_bootstrap_admission(
        gt,
        baseline,
        group_by_image=group_by_image,
        protocol=protocol,
        iterations=args.bootstrap_iterations,
    )
    candidate_robust = group_bootstrap_admission(
        gt,
        candidate,
        group_by_image=group_by_image,
        protocol=protocol,
        iterations=args.bootstrap_iterations,
    )
    delta = {
        "gate_recall_pp": 100.0
        * (candidate_metrics["gate_recall"] - baseline_metrics["gate_recall"]),
        "gate_fdr_pp": 100.0
        * (candidate_metrics["gate_fdr"] - baseline_metrics["gate_fdr"]),
        "absolute_score": candidate_metrics["absolute_score"]
        - baseline_metrics["absolute_score"],
        "per_coarse": {
            coarse: {
                "recall_pp": 100.0
                * (
                    candidate_metrics["per_coarse"][coarse]["macro_recall"]
                    - baseline_metrics["per_coarse"][coarse]["macro_recall"]
                ),
                "fdr_pp": 100.0
                * (
                    candidate_metrics["per_coarse"][coarse]["macro_fdr"]
                    - baseline_metrics["per_coarse"][coarse]["macro_fdr"]
                ),
            }
            for coarse in protocol.class_names
        },
    }
    admitted = (
        delta["gate_recall_pp"] >= 0.0
        and delta["gate_fdr_pp"] < 0.0
        and delta["absolute_score"] > 0.0
        and candidate_robust.recall_p10 >= baseline_robust.recall_p10
        and candidate_robust.fdr_p90 < baseline_robust.fdr_p90
    )
    payload = {
        "schema_version": "scaleroute_ship_macro_risk_outer_cv3_v1",
        "metric_protocol": protocol.metric_protocol,
        "selection_uses_held_out_labels": False,
        "non_ship_branches_are_frozen": True,
        "fdr_level": args.fdr_level,
        "target_fdr_ship": args.target_fdr_ship,
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_vs_baseline": delta,
        "baseline_group_bootstrap": baseline_robust.__dict__,
        "candidate_group_bootstrap": candidate_robust.__dict__,
        "folds": fold_audit,
        "admitted": admitted,
        "deployment_thresholds_require_all_oof_refit_after_admission": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"SHIP_MACRO_RISK_PASS admitted={admitted} "
        f"delta_recall_pp={delta['gate_recall_pp']:.6f} "
        f"delta_fdr_pp={delta['gate_fdr_pp']:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
