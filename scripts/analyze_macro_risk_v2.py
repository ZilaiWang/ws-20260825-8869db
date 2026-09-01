#!/usr/bin/env python3
"""Outer-CV3 MacroRisk V2 fine-threshold fit and group-robust admission."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.hierarchical_thresholds import (
    build_hierarchical_curves,
    filter_by_thresholds,
)
from rsdet.evaluation.macro_risk_v2 import (
    fit_macro_risk_v2,
    group_bootstrap_admission,
)
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config

T = TypeVar("T")


def _scoped(mapping: dict[int, list[T]], ids: set[int]) -> dict[int, list[T]]:
    return {image_id: list(mapping.get(image_id, [])) for image_id in sorted(ids)}


def _platform(gt: dict[int, list[dict[str, Any]]], pred: dict[int, list[dict[str, Any]]], protocol: Any, latency: float) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--group-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--target-fdr-ship", type=float, default=0.15)
    parser.add_argument("--target-fdr-aircraft", type=float, default=0.12)
    parser.add_argument("--target-fdr-vehicle", type=float, default=0.15)
    parser.add_argument("--latency", type=float, default=2.5)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(row["id"]): int(row["fold"]) for row in raw_gt["images"]}
    fold_ids = {
        fold: {image_id for image_id, value in fold_by_image.items() if value == fold}
        for fold in (0, 1, 2)
    }
    if any(not ids for ids in fold_ids.values()):
        raise ValueError("GT must contain non-empty folds 0/1/2")
    with args.group_map.open(encoding="utf-8", newline="") as handle:
        group_by_image = {
            int(row["image_id"]): str(row["group_id"]) for row in csv.DictReader(handle)
        }
    if set(fold_by_image) - set(group_by_image):
        raise ValueError("group map does not cover all GT images")
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    thresholds = build_threshold_grid(0.001, 0.996, 0.005)
    targets = {
        "ship": args.target_fdr_ship,
        "aircraft": args.target_fdr_aircraft,
        "vehicle": args.target_fdr_vehicle,
    }
    crossfit: dict[int, list[dict[str, Any]]] = {}
    fold_audit: dict[str, Any] = {}
    for held_out in (0, 1, 2):
        train_ids = set(fold_by_image) - fold_ids[held_out]
        group_sets: dict[int, set[str]] = defaultdict(set)
        for image_id in train_ids:
            for item in gt.get(image_id, ()):
                group_sets[int(item["category_id"])].add(group_by_image[image_id])
        curves = build_hierarchical_curves(
            gt,
            pred,
            image_ids=train_ids,
            thresholds=thresholds,
            protocol=protocol,
        )
        fit = fit_macro_risk_v2(
            curves,
            protocol=protocol,
            group_counts_by_fine={key: len(value) for key, value in group_sets.items()},
            target_fdr_by_coarse=targets,
        )
        held_pred = filter_by_thresholds(
            _scoped(pred, fold_ids[held_out]), fit["fine_thresholds"]
        )
        crossfit.update(held_pred)
        fold_audit[str(held_out)] = {
            "train_image_count": len(train_ids),
            "held_out_image_count": len(fold_ids[held_out]),
            "fit": fit,
        }
    metrics = _platform(gt, crossfit, protocol, args.latency)
    baseline_predictions = filter_by_thresholds(pred, {key: 0.15 for key in range(25)})
    baseline_metrics = _platform(gt, baseline_predictions, protocol, args.latency)
    robust = group_bootstrap_admission(
        gt,
        crossfit,
        group_by_image=group_by_image,
        protocol=protocol,
        iterations=args.bootstrap_iterations,
    )
    delta = {
        "gate_recall_pp": 100.0 * (metrics["gate_recall"] - baseline_metrics["gate_recall"]),
        "gate_fdr_pp": 100.0 * (metrics["gate_fdr"] - baseline_metrics["gate_fdr"]),
        "absolute_score": metrics["absolute_score"] - baseline_metrics["absolute_score"],
    }
    admitted = (
        metrics["recall_pass"]
        and metrics["fdr_pass"]
        and robust.admitted
        and robust.joint_pass_probability >= 0.80
        and delta["gate_recall_pp"] >= 0.5
        and delta["gate_fdr_pp"] <= 0.0
        and delta["absolute_score"] > 0.0
    )
    payload = {
        "version": "macro_risk_v2_outer_cv3",
        "metric_protocol": protocol.metric_protocol,
        "threshold_selection_uses_held_out": False,
        "baseline_global_0p15": baseline_metrics,
        "metrics": metrics,
        "delta_vs_baseline": delta,
        "group_bootstrap": robust.__dict__,
        "folds": fold_audit,
        "admitted": admitted,
        "deployment_thresholds_must_be_refit_on_all_oof_only_after_admission": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
