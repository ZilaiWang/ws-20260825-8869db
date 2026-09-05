#!/usr/bin/env python3
"""Compare native-source inference with inference after pseudo-10K composition."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions_with_trace
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _coarse(category_id: int) -> str:
    if category_id < 4:
        return "ship"
    if category_id < 24:
        return "aircraft"
    return "vehicle"


def _evaluate(gt: Any, predictions: Any, protocol: Any) -> tuple[dict[str, Any], Any]:
    metrics, trace = evaluate_predictions_with_trace(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "per_coarse": {
            name: {
                "tp": row.tp,
                "fp": row.fp,
                "fn": row.fn,
                "recall": row.recall,
                "fdr": row.fdr,
            }
            for name, row in metrics.per_class.items()
        },
        **metrics.details,
    }, trace


def _paired(trace_a: Any, trace_b: Any, gt: Any) -> dict[str, dict[str, int]]:
    hits_a = {(row.image_id, row.ground_truth_index) for row in trace_a.matches}
    hits_b = {(row.image_id, row.ground_truth_index) for row in trace_b.matches}
    counts: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for image_id, rows in gt.items():
        for index, row in enumerate(rows):
            a = (image_id, index) in hits_a
            b = (image_id, index) in hits_b
            outcome = "both" if a and b else "source_only" if a else "pseudo_only" if b else "neither"
            counts["all"][outcome] += 1
            counts[_coarse(int(row["category_id"]))][outcome] += 1
    return {name: dict(row) for name, row in sorted(counts.items())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--source-projected", type=Path, required=True)
    parser.add_argument("--pseudo-tiled", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gt = load_coco_ground_truth(args.ground_truth)
    source = load_coco_predictions(args.source_projected)
    pseudo = load_coco_predictions(args.pseudo_tiled)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    source_metrics, source_trace = _evaluate(gt, source, protocol)
    pseudo_metrics, pseudo_trace = _evaluate(gt, pseudo, protocol)
    result = {
        "status": "complete",
        "role": "paired_native_source_vs_pseudo_composition_diagnostic",
        "input_sha256": {
            "ground_truth": _sha256(args.ground_truth),
            "source_projected": _sha256(args.source_projected),
            "pseudo_tiled": _sha256(args.pseudo_tiled),
        },
        "native_source_projected": source_metrics,
        "pseudo_composed_and_tiled": pseudo_metrics,
        "pseudo_minus_source": {
            "recall": float(pseudo_metrics["recall"]) - float(source_metrics["recall"]),
            "fdr": float(pseudo_metrics["fdr"]) - float(source_metrics["fdr"]),
            "tp": int(pseudo_metrics["tp"]) - int(source_metrics["tp"]),
            "fp": int(pseudo_metrics["fp"]) - int(source_metrics["fp"]),
            "fn": int(pseudo_metrics["fn"]) - int(source_metrics["fn"]),
            "per_coarse": {
                name: {
                    field: float(pseudo_metrics["per_coarse"][name][field])
                    - float(source_metrics["per_coarse"][name][field])
                    for field in ("tp", "fp", "fn", "recall", "fdr")
                }
                for name in protocol.class_names
            },
        },
        "paired_gt_outcomes": _paired(source_trace, pseudo_trace, gt),
        "limits": [
            "The detector was trained on these sources; only the within-source paired difference is interpretable.",
            "The pseudo path is 2.34375 percent smaller at the network input than direct source inference.",
            "The remaining difference combines scale, artificial neighbours, artificial seams, and tile fusion.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
