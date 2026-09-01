#!/usr/bin/env python3
"""Leakage-safe CV3 replay of selective D-FINE vehicle reject/rescue."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.contracts import Prediction
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.submission.vehicle_rescue import (
    VehicleRescueConfig,
    apply_vehicle_reject_rescue,
)
from rsdet.utils.config import load_config


def _prediction(image_id: int, rows: list[dict[str, Any]]) -> Prediction:
    return Prediction(
        image_id,
        [list(row["bbox_xyxy"]) for row in rows],
        [float(row["score"]) for row in rows],
        [int(row["category_id"]) for row in rows],
    )


def _rows(prediction: Prediction) -> list[dict[str, Any]]:
    return [
        {"bbox_xyxy": list(box), "score": float(score), "category_id": int(label)}
        for box, score, label in zip(
            prediction.boxes_xyxy,
            prediction.scores,
            prediction.labels,
            strict=True,
        )
    ]


def replay(
    primary: dict[int, list[dict[str, Any]]],
    specialist: dict[int, list[dict[str, Any]]],
    config: VehicleRescueConfig,
    output_threshold: float,
) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for image_id in sorted(set(primary) | set(specialist)):
        output[image_id] = [
            row
            for row in _rows(
                apply_vehicle_reject_rescue(
                    _prediction(image_id, primary.get(image_id, [])),
                    _prediction(image_id, specialist.get(image_id, [])),
                    config=config,
                )
            )
            if float(row["score"]) >= output_threshold
        ]
    return output


def _metrics(gt: dict[int, list[dict[str, Any]]], pred: dict[int, list[dict[str, Any]]], protocol: Any, latency: float) -> dict[str, Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    platform = build_platform_observed_metrics(
        ranking,
        recall_min=protocol.recall_min,
        fdr_max=protocol.fdr_max,
        latency_seconds=latency,
        latency_max_seconds=protocol.latency_max_seconds,
    )
    return platform_metrics_payload(platform)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--latency", type=float, default=4.9)
    parser.add_argument("--core", type=float, nargs="+", default=(0.15, 0.18, 0.20, 0.22, 0.25))
    parser.add_argument("--product", type=float, nargs="+", default=(0.03, 0.04, 0.05, 0.059, 0.07))
    parser.add_argument("--candidate-floor", type=float, default=0.001)
    parser.add_argument("--support-iou", type=float, default=0.35)
    parser.add_argument("--base-threshold", type=float, default=0.15)
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    folds: dict[int, dict[str, Any]] = {}
    for fold in (0, 1, 2):
        directory = args.root / f"fold_{fold}"
        folds[fold] = {
            "gt": load_coco_ground_truth(directory / "instances_val.json"),
            "primary": load_coco_predictions(directory / "y5_predictions.json"),
            "specialist": load_coco_predictions(directory / "dfine_predictions.json"),
        }

    candidate_table = list(itertools.product(args.core, args.product))
    crossfit_predictions: dict[int, list[dict[str, Any]]] = {}
    fold_audit: dict[str, Any] = {}
    for held_out in (0, 1, 2):
        scored: list[tuple[float, float, float, VehicleRescueConfig, dict[str, Any]]] = []
        train_gt: dict[int, list[dict[str, Any]]] = {}
        train_primary: dict[int, list[dict[str, Any]]] = {}
        train_specialist: dict[int, list[dict[str, Any]]] = {}
        offset = 0
        for fold in (0, 1, 2):
            if fold == held_out:
                continue
            for image_id in sorted(folds[fold]["gt"]):
                synthetic = offset
                offset += 1
                train_gt[synthetic] = folds[fold]["gt"].get(image_id, [])
                train_primary[synthetic] = folds[fold]["primary"].get(image_id, [])
                train_specialist[synthetic] = folds[fold]["specialist"].get(image_id, [])
        train_baseline = {
            image_id: [
                row for row in rows if float(row["score"]) >= args.base_threshold
            ]
            for image_id, rows in train_primary.items()
        }
        baseline = _metrics(train_gt, train_baseline, protocol, args.latency)
        base_vehicle = baseline["per_coarse"]["vehicle"]
        for core, product in candidate_table:
            config = VehicleRescueConfig(
                core_threshold=core,
                candidate_floor=args.candidate_floor,
                support_iou=args.support_iou,
                rescue_product_threshold=product,
                promoted_score=min(1.0, core + 1e-6),
            )
            metrics = _metrics(
                train_gt,
                replay(
                    train_primary,
                    train_specialist,
                    config,
                    output_threshold=args.base_threshold,
                ),
                protocol,
                args.latency,
            )
            vehicle = metrics["per_coarse"]["vehicle"]
            recall_guard = vehicle["macro_recall"] >= base_vehicle["macro_recall"] - 0.005
            score = float(metrics["absolute_score"] or -1.0) if recall_guard else -1.0
            scored.append((score, vehicle["macro_recall"], -vehicle["macro_fdr"], config, metrics))
        selected = max(scored, key=lambda row: row[:3])
        config = selected[3]
        held_pred = replay(
            folds[held_out]["primary"],
            folds[held_out]["specialist"],
            config,
            output_threshold=args.base_threshold,
        )
        crossfit_predictions.update(held_pred)
        fold_audit[str(held_out)] = {
            "selected_config": config.__dict__,
            "training_metrics": selected[4],
            "held_out_baseline": _metrics(
                folds[held_out]["gt"],
                {
                    image_id: [
                        row for row in rows if float(row["score"]) >= args.base_threshold
                    ]
                    for image_id, rows in folds[held_out]["primary"].items()
                },
                protocol,
                args.latency,
            ),
            "held_out_candidate": _metrics(
                folds[held_out]["gt"], held_pred, protocol, args.latency
            ),
        }

    all_gt: dict[int, list[dict[str, Any]]] = {}
    all_primary: dict[int, list[dict[str, Any]]] = {}
    for fold in (0, 1, 2):
        all_gt.update(folds[fold]["gt"])
        all_primary.update(folds[fold]["primary"])
    baseline_predictions = {
        image_id: [row for row in rows if float(row["score"]) >= args.base_threshold]
        for image_id, rows in all_primary.items()
    }
    baseline = _metrics(all_gt, baseline_predictions, protocol, args.latency)
    candidate = _metrics(all_gt, crossfit_predictions, protocol, args.latency)
    admitted = (
        candidate["gate_recall"] >= baseline["gate_recall"] - 0.005
        and candidate["gate_fdr"] <= baseline["gate_fdr"]
        and (candidate["absolute_score"] or 0.0) > (baseline["absolute_score"] or 0.0)
    )
    payload = {
        "version": "vehicle_reject_rescue_cv3_v1",
        "metric_protocol": protocol.metric_protocol,
        "selection_is_train_folds_only": True,
        "baseline": baseline,
        "candidate": candidate,
        "folds": fold_audit,
        "admitted": admitted,
        "deployment_mode": "selective_vehicle_reject_rescue" if admitted else "identity",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
