#!/usr/bin/env python3
"""Evaluate a target pseudo set with per-coarse thresholds frozen elsewhere.

The source frontier is produced by
``analyze_cv3_oof_pseudo_coarse_thresholds.py``.  For every target fold and
coarse category, this script applies the threshold that was selected without
the corresponding source fold.  No target labels are used for selection.
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
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _thresholds(
    frontier: dict[str, Any], fdr_level: float
) -> dict[int, dict[str, float]]:
    key = f"{fdr_level:.3f}"
    raw = frontier.get("frontiers", {}).get(key, {}).get("crossfit_thresholds")
    if not isinstance(raw, dict):
        raise ValueError(f"source frontier does not contain FDR level {key}")
    output: dict[int, dict[str, float]] = {}
    for fold_text, coarse_values in raw.items():
        if not isinstance(coarse_values, dict):
            raise ValueError("source frontier must contain per-coarse thresholds")
        output[int(fold_text)] = {
            coarse: float(value["threshold"])
            for coarse, value in coarse_values.items()
        }
    if set(output) != {0, 1, 2}:
        raise ValueError("source frontier must provide folds 0,1,2")
    required = {"ship", "aircraft", "vehicle"}
    if any(set(values) != required for values in output.values()):
        raise ValueError("each fold must provide ship, aircraft and vehicle")
    return output


def _metrics(metrics: Any, ranking: Any) -> dict[str, Any]:
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
    parser.add_argument("--source-frontier", type=Path, required=True)
    parser.add_argument("--fdr-level", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(item["id"]): int(item["fold"]) for item in raw_gt["images"]}
    source = json.loads(args.source_frontier.read_text(encoding="utf-8"))
    thresholds = _thresholds(source, args.fdr_level)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    filtered: dict[int, list[dict[str, Any]]] = {}
    for image_id in gt:
        fold = fold_by_image[image_id]
        filtered[image_id] = [
            item
            for item in pred.get(image_id, [])
            if float(item["score"])
            >= thresholds[fold][protocol.category_mapping[int(item["category_id"])]]
        ]
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
    payload = {
        "status": "complete",
        "protocol": "source_per_coarse_frontier_thresholds_frozen_target_pseudo10k_v1",
        "source_fdr_level": args.fdr_level,
        "thresholds": {str(fold): values for fold, values in thresholds.items()},
        "metrics": _metrics(metrics, ranking),
        "input_sha256": {
            "gt": _sha256(args.gt),
            "pred": _sha256(args.pred),
            "source_frontier": _sha256(args.source_frontier),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
