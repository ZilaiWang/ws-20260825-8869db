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

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(metrics: Any, ranking: Any) -> dict[str, Any]:
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
    points: list[tuple[float, dict[str, Any]]] = []
    threshold = 0.001
    while threshold <= 1.0 + 1e-12:
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
        points.append((threshold, _payload(metrics, ranking)))
        threshold += args.step
    frontiers: dict[str, Any] = {}
    for fdr in args.fdr_levels:
        feasible = [point for point in points if point[1]["fdr"] <= fdr]
        selected = max(
            feasible,
            key=lambda point: (
                point[1]["recall"],
                -point[1]["fdr"],
                point[0],
            ),
        )
        frontiers[f"{fdr:.3f}"] = {"threshold": selected[0], **selected[1]}
    result = {
        "status": "complete_diagnostic_only",
        "protocol": "single_heldout_split_oracle_frontier_not_for_threshold_selection_v1",
        "warning": "The held-out labels are used for this diagnostic frontier.",
        "threshold_grid": {"start": 0.001, "step": args.step},
        "score_floor_metrics": {"threshold": points[0][0], **points[0][1]},
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
