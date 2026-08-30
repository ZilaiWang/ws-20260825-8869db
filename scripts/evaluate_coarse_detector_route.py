#!/usr/bin/env python3
"""Evaluate a fixed coarse-class route between two detectors.

This is a diagnostic, not a threshold tuner: the primary detector supplies ship
and aircraft predictions while the specialist supplies vehicle predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import EvaluationProtocol, parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_predictions(
    primary: dict[int, list[dict[str, Any]]],
    specialist: dict[int, list[dict[str, Any]]],
    *,
    category_mapping: dict[int, str],
    primary_threshold: float,
    specialist_threshold: float,
    specialist_coarse: str = "vehicle",
) -> dict[int, list[dict[str, Any]]]:
    """Route one coarse class to a specialist without score fusion."""
    image_ids = set(primary) | set(specialist)
    routed: dict[int, list[dict[str, Any]]] = {}
    for image_id in image_ids:
        retained = [
            item
            for item in primary.get(image_id, [])
            if category_mapping[int(item["category_id"])] != specialist_coarse
            and float(item["score"]) >= primary_threshold
        ]
        retained.extend(
            item
            for item in specialist.get(image_id, [])
            if category_mapping[int(item["category_id"])] == specialist_coarse
            and float(item["score"]) >= specialist_threshold
        )
        routed[image_id] = retained
    return routed


def _metric_payload(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    protocol: EvaluationProtocol,
) -> dict[str, Any]:
    metrics = evaluate_predictions(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "tp": metrics.details["tp"],
        "fp": metrics.details["fp"],
        "fn": metrics.details["fn"],
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
        "per_coarse": {
            name: {
                "recall": item.recall,
                "fdr": item.fdr,
                "tp": item.tp,
                "fp": item.fp,
                "fn": item.fn,
            }
            for name, item in metrics.per_class.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--primary-pred", type=Path, required=True)
    parser.add_argument("--specialist-pred", type=Path, required=True)
    parser.add_argument("--primary-threshold", type=float, required=True)
    parser.add_argument("--specialist-threshold", type=float, required=True)
    parser.add_argument("--specialist-coarse", default="vehicle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if not 0.0 <= args.primary_threshold <= 1.0:
        raise ValueError("primary threshold must be within [0, 1]")
    if not 0.0 <= args.specialist_threshold <= 1.0:
        raise ValueError("specialist threshold must be within [0, 1]")

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    if args.specialist_coarse not in set(protocol.category_mapping.values()):
        raise ValueError(f"unknown specialist coarse class: {args.specialist_coarse}")
    gt = load_coco_ground_truth(args.gt)
    primary = load_coco_predictions(args.primary_pred)
    specialist = load_coco_predictions(args.specialist_pred)
    routed = route_predictions(
        primary,
        specialist,
        category_mapping=protocol.category_mapping,
        primary_threshold=args.primary_threshold,
        specialist_threshold=args.specialist_threshold,
        specialist_coarse=args.specialist_coarse,
    )
    primary_filtered = {
        image_id: [
            item
            for item in primary.get(image_id, [])
            if float(item["score"]) >= args.primary_threshold
        ]
        for image_id in gt
    }
    baseline = _metric_payload(gt, primary_filtered, protocol)
    candidate = _metric_payload(gt, routed, protocol)
    result = {
        "status": "complete",
        "protocol": "fixed_coarse_detector_route_diagnostic_v1",
        "warning": "heldout thresholds are diagnostic only; CV3 cross-fitting is required",
        "route": {
            "primary_coarse": sorted(
                set(protocol.category_mapping.values()) - {args.specialist_coarse}
            ),
            "specialist_coarse": args.specialist_coarse,
            "primary_threshold": args.primary_threshold,
            "specialist_threshold": args.specialist_threshold,
        },
        "primary_baseline": baseline,
        "routed_candidate": candidate,
        "delta": {
            "recall": candidate["recall"] - baseline["recall"],
            "fdr": candidate["fdr"] - baseline["fdr"],
            "macro_recall": candidate["macro_recall"] - baseline["macro_recall"],
            "macro_fdr": candidate["macro_fdr"] - baseline["macro_fdr"],
            "per_coarse": {
                name: {
                    "recall": candidate["per_coarse"][name]["recall"]
                    - baseline["per_coarse"][name]["recall"],
                    "fdr": candidate["per_coarse"][name]["fdr"]
                    - baseline["per_coarse"][name]["fdr"],
                }
                for name in baseline["per_coarse"]
            },
        },
        "input_sha256": {
            "gt": _sha256(args.gt),
            "primary_pred": _sha256(args.primary_pred),
            "specialist_pred": _sha256(args.specialist_pred),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
