#!/usr/bin/env python3
"""Diagnostic official-matching frontier for a single held-out split."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import build_threshold_curve
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(metrics: Any, ranking: Any, protocol: Any) -> dict[str, Any]:
    platform = platform_metrics_payload(
        build_platform_observed_metrics(
            ranking,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
            latency_seconds=None,
            latency_max_seconds=protocol.latency_max_seconds,
        )
    )
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "pooled_recall": metrics.recall,
        "pooled_fdr": metrics.fdr,
        "tp": metrics.details["tp"],
        "fp": metrics.details["fp"],
        "fn": metrics.details["fn"],
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
        "fine25_macro_recall": ranking.overall_recall,
        "fine25_macro_fdr": ranking.overall_fdr,
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
        "ranking_per_coarse": {
            name: {
                "macro_recall": item.macro_recall,
                "macro_fdr": item.macro_fdr,
                "pooled_recall": item.pooled_recall,
                "pooled_fdr": item.pooled_fdr,
                "fine_count": item.fine_count,
                "fine_ids": item.fine_ids,
            }
            for name, item in ranking.per_coarse.items()
        },
        "platform": platform,
        "platform_gate_recall": platform["gate_recall"],
        "platform_gate_fdr": platform["gate_fdr"],
    }


def _select_platform_point(
    curve: list[dict[str, Any]], target_fdr: float
) -> dict[str, Any]:
    """Select a diagnostic point using the observed three-coarse platform gate."""

    feasible = [
        point for point in curve if float(point["platform_gate_fdr"]) <= target_fdr
    ]
    pool = feasible or curve
    if feasible:
        return max(
            pool,
            key=lambda point: (
                float(point["platform_gate_recall"]),
                -float(point["platform_gate_fdr"]),
                float(point["threshold"]),
            ),
        )
    return min(
        pool,
        key=lambda point: (
            float(point["platform_gate_fdr"]),
            -float(point["platform_gate_recall"]),
            -float(point["threshold"]),
        ),
    )


def _select_quality_point(curve: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the best six-quality-subscore point for diagnosis only."""

    return max(
        curve,
        key=lambda point: (
            float(point["platform_quality_score"]),
            float(point["platform_gate_recall"]),
            -float(point["platform_gate_fdr"]),
            float(point["threshold"]),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step", type=float, default=0.005)
    parser.add_argument(
        "--fdr-levels", type=float, nargs="+", default=(0.10, 0.12, 0.15, 0.20)
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if not 0.0 < args.step <= 1.0:
        raise ValueError("step must be within (0, 1]")

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    thresholds = build_threshold_grid(0.001, 1.0, args.step)
    curve, trace_audit = build_threshold_curve(
        gt,
        pred,
        thresholds=thresholds,
        protocol=protocol,
    )

    def evaluate_threshold(threshold: float) -> dict[str, Any]:
        filtered = {
            image_id: [
                row
                for row in pred.get(image_id, [])
                if float(row["score"]) >= threshold
            ]
            for image_id in gt
        }
        metrics = evaluate_predictions(
            gt,
            filtered,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        ranking = evaluate_ranking_metrics(
            gt,
            filtered,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        return _payload(metrics, ranking, protocol)

    frontiers: dict[str, Any] = {}
    for fdr in args.fdr_levels:
        selected = _select_platform_point(curve, float(fdr))
        threshold = float(selected["threshold"])
        frontiers[f"{fdr:.3f}"] = {
            "threshold": threshold,
            **evaluate_threshold(threshold),
        }
    floor_threshold = float(curve[0]["threshold"])
    quality_selected = _select_quality_point(curve)
    quality_threshold = float(quality_selected["threshold"])
    result = {
        "status": "complete_diagnostic_only",
        "protocol": "single_heldout_split_platform_observed_oracle_frontier_v2",
        "warning": "The held-out labels are used for this diagnostic frontier.",
        "selection_metric": (
            "platform_observed_20260831 three-coarse mean macro Recall under "
            "three-coarse mean macro FDR constraint"
        ),
        "legacy_fields": (
            "recall/fdr and per_coarse are pooled diagnostics; macro_recall/macro_fdr "
            "are the 25-fine average and are not the platform gate"
        ),
        "threshold_grid": {"start": 0.001, "step": args.step},
        "score_prefix_trace_audit": trace_audit,
        "score_floor_metrics": {
            "threshold": floor_threshold,
            **evaluate_threshold(floor_threshold),
        },
        "quality_oracle": {
            "threshold": quality_threshold,
            "selection_quality_score": float(
                quality_selected["platform_quality_score"]
            ),
            **evaluate_threshold(quality_threshold),
        },
        "frontiers": frontiers,
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
