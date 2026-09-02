#!/usr/bin/env python3
"""Evaluate COCO predictions at one externally frozen global score threshold."""

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
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _metrics_payload(metrics: Any, ranking: Any, protocol: Any) -> dict[str, Any]:
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    raw = load_coco_predictions(args.pred)
    pred = {
        image_id: [row for row in raw.get(image_id, []) if float(row["score"]) >= args.threshold]
        for image_id in gt
    }
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
    payload = {
        "status": "complete",
        "protocol": "fixed_external_global_score_threshold_platform_observed_v2",
        "threshold": args.threshold,
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
        "legacy_fields": (
            "recall/fdr and per_coarse are pooled diagnostics; macro_recall/macro_fdr "
            "are the 25-fine average and are not the platform gate"
        ),
        **_metrics_payload(metrics, ranking, protocol),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
