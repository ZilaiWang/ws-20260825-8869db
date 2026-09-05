#!/usr/bin/env python3
"""Audit exact class ownership, quality, and latency of a Sprint20 runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from sprint20.evaluation import evaluate


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline-pred", type=Path, required=True)
    parser.add_argument("--candidate-pred", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--alternative-labels", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-p40-d4-latency", type=float, default=4.95)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    alternative = set(args.alternative_labels)
    if not alternative or len(alternative) != len(args.alternative_labels):
        raise ValueError("alternative labels must be unique and nonempty")

    gt = load_coco_ground_truth(args.gt)
    baseline = load_coco_predictions(args.baseline_pred)
    candidate = load_coco_predictions(args.candidate_pred)
    protected = set(range(25)) - alternative
    protected_exact = _canonical(baseline, protected) == _canonical(candidate, protected)
    if not protected_exact:
        raise AssertionError("Sprint20 changed labels outside its declared ownership")
    baseline_summary = json.loads(args.baseline_summary.read_text(encoding="utf-8"))
    candidate_summary = json.loads(args.candidate_summary.read_text(encoding="utf-8"))
    baseline_time = float(baseline_summary["mean_image_seconds"])
    candidate_time = float(candidate_summary["mean_image_seconds"])
    increment = candidate_time - baseline_time
    projected = args.official_p40_d4_latency + increment
    baseline_metrics = evaluate(gt, baseline, baseline_time)
    candidate_metrics = evaluate(gt, candidate, candidate_time)
    payload = {
        "status": "complete",
        "protocol": "sprint20_class_disjoint_exact_runtime_audit_v1",
        "alternative_labels": args.alternative_labels,
        "protected_labels_exact": protected_exact,
        "metrics": {"baseline": baseline_metrics, "candidate": candidate_metrics},
        "delta_score_at_proxy_latency": (
            candidate_metrics["score"]["total_score"]
            - baseline_metrics["score"]["total_score"]
        ),
        "latency": {
            "proxy_baseline_mean_seconds": baseline_time,
            "proxy_candidate_mean_seconds": candidate_time,
            "proxy_incremental_seconds": increment,
            "official_p40_d4_seconds_assumption": args.official_p40_d4_latency,
            "projected_official_seconds": projected,
            "projection_is_diagnostic_not_official_measurement": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
