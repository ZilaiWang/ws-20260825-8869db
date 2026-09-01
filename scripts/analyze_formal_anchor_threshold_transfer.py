#!/usr/bin/env python3
"""Project a finite threshold policy set onto the observed formal anchor.

The formal result is used only as an immutable anchor.  For every OOF fold we
measure a candidate's paired per-coarse delta versus threshold 0.15, add that
delta to the anchor rates, and recompute the seven-subscore platform result.
No hidden labels are inferred and no continuous threshold search is performed.
Admission requires both a positive worst-fold projection and a useful median
gain, so the result is deliberately conservative.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.data.xh_dataset import coarse_name
from rsdet.evaluation.absolute_score import platform_confirmed_score
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import COARSE_ORDER
from rsdet.evaluation.protocol import parse_evaluation_protocol
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
            if float(row["score"])
            >= float(thresholds[coarse_name(int(row["category_id"]))])
        ]
        for image_id, rows in pred.items()
    }


def _rates(gt: dict[int, list[dict[str, Any]]], pred: dict[int, list[dict[str, Any]]], protocol: Any) -> dict[str, dict[str, float]]:
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return {
        coarse: {
            "recall": float(ranking.per_coarse[coarse].macro_recall),
            "fdr": float(ranking.per_coarse[coarse].macro_fdr),
        }
        for coarse in COARSE_ORDER
    }


def _project(
    anchor: dict[str, dict[str, float]],
    baseline: dict[str, dict[str, float]],
    candidate: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    return {
        coarse: {
            metric: min(
                1.0,
                max(
                    0.0,
                    float(anchor[coarse][metric])
                    + float(candidate[coarse][metric])
                    - float(baseline[coarse][metric]),
                ),
            )
            for metric in ("recall", "fdr")
        }
        for coarse in COARSE_ORDER
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(row["id"]): int(row["fold"]) for row in raw_gt["images"]}
    fold_ids = {
        fold: {image_id for image_id, value in fold_by_image.items() if value == fold}
        for fold in (0, 1, 2)
    }
    if any(not ids for ids in fold_ids.values()):
        raise ValueError("GT must contain folds 0/1/2")
    gt = load_coco_ground_truth(args.gt)
    predictions = load_coco_predictions(args.predictions)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    anchor_raw = json.loads(args.anchor.read_text(encoding="utf-8"))
    anchor = {
        coarse: {
            "recall": float(anchor_raw["per_coarse"][coarse]["recall"]),
            "fdr": float(anchor_raw["per_coarse"][coarse]["fdr"]),
        }
        for coarse in COARSE_ORDER
    }
    latency = float(anchor_raw["latency_seconds"])
    spec = json.loads(args.policies.read_text(encoding="utf-8"))
    policies = spec["policies"]
    if not policies or policies[0]["name"] != "identity_015":
        raise ValueError("first policy must be identity_015")
    baseline_thresholds = {coarse: 0.15 for coarse in COARSE_ORDER}
    anchor_score = platform_confirmed_score(anchor, latency)
    results: list[dict[str, Any]] = []
    for policy in policies:
        thresholds = {coarse: float(policy[coarse]) for coarse in COARSE_ORDER}
        fold_rows = []
        for fold in (0, 1, 2):
            ids = fold_ids[fold]
            fold_gt = _scoped(gt, ids)
            fold_pred = _scoped(predictions, ids)
            baseline_rates = _rates(
                fold_gt, _filter(fold_pred, baseline_thresholds), protocol
            )
            candidate_rates = _rates(
                fold_gt, _filter(fold_pred, thresholds), protocol
            )
            projected = _project(anchor, baseline_rates, candidate_rates)
            score = platform_confirmed_score(projected, latency)
            recall_drops = {
                coarse: baseline_rates[coarse]["recall"]
                - candidate_rates[coarse]["recall"]
                for coarse in COARSE_ORDER
            }
            fold_rows.append(
                {
                    "fold": fold,
                    "baseline_oof": baseline_rates,
                    "candidate_oof": candidate_rates,
                    "projected_formal": projected,
                    "projected_score": score,
                    "score_gain": float(score["total_score"])
                    - float(anchor_score["total_score"]),
                    "max_coarse_recall_drop": max(recall_drops.values()),
                }
            )
        gains = [float(row["score_gain"]) for row in fold_rows]
        drops = [float(row["max_coarse_recall_drop"]) for row in fold_rows]
        gate_passes = [
            bool(row["projected_score"]["hard_gates"]["fdr_pass"])
            for row in fold_rows
        ]
        admission = spec["admission"]
        admitted = (
            min(gains) >= float(admission["minimum_worst_fold_projected_score_gain"])
            and statistics.median(gains)
            >= float(admission["minimum_median_projected_score_gain"])
            and max(drops) <= float(admission["maximum_any_coarse_recall_drop"])
            and (
                all(gate_passes)
                if bool(admission["require_projected_gate_fdr_pass"])
                else True
            )
        )
        results.append(
            {
                "name": str(policy["name"]),
                "thresholds": thresholds,
                "folds": fold_rows,
                "median_projected_score_gain": statistics.median(gains),
                "worst_projected_score_gain": min(gains),
                "maximum_fold_coarse_recall_drop": max(drops),
                "all_projected_fdr_gates_pass": all(gate_passes),
                "admitted": admitted,
            }
        )
    payload = {
        "version": "formal_anchor_finite_threshold_transfer_v1",
        "metric_protocol": protocol.metric_protocol,
        "formal_anchor_used_only_as_fixed_baseline": True,
        "continuous_threshold_search": False,
        "anchor_score": anchor_score,
        "admission": spec["admission"],
        "results": results,
        "admitted_policies": [row["name"] for row in results if row["admitted"]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
