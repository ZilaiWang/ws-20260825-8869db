#!/usr/bin/env python3
"""Evaluate a new pseudo-10K set with thresholds frozen on another set."""

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


def thresholds_from_frontier(frontier: dict[str, Any], fdr_level: float) -> dict[int, float]:
    key = f"{fdr_level:.3f}"
    if key not in frontier.get("frontiers", {}):
        raise ValueError(f"source frontier does not contain FDR level {key}")
    raw = frontier["frontiers"][key]["crossfit_thresholds"]
    thresholds = {int(fold): float(value) for fold, value in raw.items()}
    if set(thresholds) != {0, 1, 2}:
        raise ValueError("source frontier must provide folds 0,1,2")
    return thresholds


def absolute_score_thresholds_from_frontier(frontier: dict[str, Any]) -> dict[int, float]:
    try:
        raw = frontier["absolute_score_crossfit"]["crossfit_thresholds"]
    except (KeyError, TypeError) as error:
        raise ValueError("source frontier lacks absolute-score crossfit thresholds") from error
    thresholds = {int(fold): float(value) for fold, value in raw.items()}
    if set(thresholds) != {0, 1, 2}:
        raise ValueError("source frontier must provide folds 0,1,2")
    return thresholds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--source-frontier", type=Path, required=True)
    parser.add_argument("--fdr-level", type=float)
    parser.add_argument(
        "--selection-mode", choices=("fdr_level", "absolute_score"), default="fdr_level"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(item["id"]): int(item["fold"]) for item in raw_gt["images"]}
    if set(fold_by_image.values()) != {0, 1, 2}:
        raise ValueError("target ground truth must contain folds 0,1,2")
    source = json.loads(args.source_frontier.read_text(encoding="utf-8"))
    if args.selection_mode == "fdr_level":
        if args.fdr_level is None:
            parser.error("--fdr-level is required with --selection-mode=fdr_level")
        thresholds = thresholds_from_frontier(source, args.fdr_level)
    else:
        thresholds = absolute_score_thresholds_from_frontier(source)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    filtered = {
        image_id: [
            item
            for item in pred.get(image_id, [])
            if float(item["score"]) >= thresholds[fold_by_image[image_id]]
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
    payload = {
        "status": "complete",
        "protocol": "source_frontier_thresholds_frozen_target_pseudo10k_v2",
        "selection_mode": args.selection_mode,
        "source_fdr_level": args.fdr_level,
        "thresholds": {str(key): value for key, value in thresholds.items()},
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
        "input_sha256": {
            "gt": _sha256(args.gt),
            "pred": _sha256(args.pred),
            "source_frontier": _sha256(args.source_frontier),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
