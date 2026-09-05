#!/usr/bin/env python3
"""Audit D4-only runtime ownership, reference parity, quality, and latency."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _canonical(
    rows: dict[int, list[dict[str, Any]]], labels: set[int]
) -> list[tuple[Any, ...]]:
    return sorted(
        (
            image_id,
            int(row["category_id"]),
            float(row["score"]),
            tuple(float(value) for value in row["bbox_xyxy"]),
        )
        for image_id, items in rows.items()
        for row in items
        if int(row["category_id"]) in labels
    )


def _metrics(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
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
        build_platform_observed_metrics(ranking, latency_seconds=latency)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline-pred", type=Path, required=True)
    parser.add_argument("--candidate-pred", type=Path, required=True)
    parser.add_argument("--reference-v3-pred", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-p40-latency", type=float, default=3.551833)
    args = parser.parse_args()
    gt = load_coco_ground_truth(args.gt)
    baseline = load_coco_predictions(args.baseline_pred)
    candidate = load_coco_predictions(args.candidate_pred)
    reference = load_coco_predictions(args.reference_v3_pred)
    ship = set(range(4))
    aircraft = set(range(4, 24))
    vehicle = {24}
    assertions = {
        "ship_candidate_equals_p40": _canonical(candidate, ship)
        == _canonical(baseline, ship),
        "vehicle_candidate_equals_p40": _canonical(candidate, vehicle)
        == _canonical(baseline, vehicle),
        "aircraft_candidate_equals_v3_d4_reference": _canonical(candidate, aircraft)
        == _canonical(reference, aircraft),
        "ship_p40_equals_v3_primary_reference": _canonical(baseline, ship)
        == _canonical(reference, ship),
    }
    if not all(assertions.values()):
        failed = [key for key, passed in assertions.items() if not passed]
        raise AssertionError(f"D4 ownership/parity assertion failed: {failed}")
    baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    candidate_summary = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    base_time = float(baseline_summary["mean_image_seconds"])
    candidate_time = float(candidate_summary["mean_image_seconds"])
    incremental = candidate_time - base_time
    projected = args.official_p40_latency + incremental
    protocol = parse_evaluation_protocol(load_config(Path("configs/project.yaml")))
    payload = {
        "status": "complete",
        "protocol": "p40_aircraft_d4_only_exact_runtime_audit_v1",
        "assertions": assertions,
        "metrics": {
            "baseline": _metrics(gt, baseline, protocol, base_time),
            "candidate": _metrics(gt, candidate, protocol, candidate_time),
        },
        "latency": {
            "proxy_baseline_mean_seconds": base_time,
            "proxy_candidate_mean_seconds": candidate_time,
            "proxy_incremental_seconds": incremental,
            "official_p40_seconds": args.official_p40_latency,
            "projected_official_seconds_using_proxy_increment": projected,
            "break_even_seconds": 5.8023,
            "target_seconds": 4.5,
            "break_even_projection_pass": projected < 5.8023,
            "target_projection_pass": projected < 4.5,
            "projection_is_diagnostic_not_official_measurement": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
