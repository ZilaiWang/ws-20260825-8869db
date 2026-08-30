#!/usr/bin/env python3
"""Evaluate one prediction file at an explicitly frozen coarse workpoint."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--threshold-ship", type=float, required=True)
    parser.add_argument("--threshold-aircraft", type=float, required=True)
    parser.add_argument("--threshold-vehicle", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    thresholds = {
        "ship": args.threshold_ship,
        "aircraft": args.threshold_aircraft,
        "vehicle": args.threshold_vehicle,
    }
    if any(not 0.0 <= value <= 1.0 for value in thresholds.values()):
        raise ValueError("all thresholds must be within [0, 1]")

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    filtered = {
        image_id: [
            item
            for item in pred.get(image_id, [])
            if float(item["score"])
            >= thresholds[protocol.category_mapping[int(item["category_id"])]]
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
    result: dict[str, Any] = {
        "status": "complete",
        "protocol": "fixed_per_coarse_workpoint_official_match_v1",
        "thresholds": thresholds,
        "metrics": {
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
        },
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
