#!/usr/bin/env python3
"""Compare exact whole-image and production-tiled predictions on native images."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import (
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.tiling.slicer import generate_tiles
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


def _metrics(gt: Any, predictions: Any, protocol: Any) -> dict[str, Any]:
    pooled, trace = evaluate_predictions_with_trace(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        # This deliberately small implementation probe does not contain every
        # fine class.  The payload remains useful only for paired deltas; the
        # pooled counts and per-GT outcomes are the primary diagnostics.
        require_complete_taxonomy=False,
    )
    return {
        "pooled": {
            "recall": pooled.recall,
            "fdr": pooled.fdr,
            **pooled.details,
        },
        "supported_fine_macro": {
            "per_coarse": {
                name: {
                    "macro_recall": row.macro_recall,
                    "macro_fdr": row.macro_fdr,
                    "pooled_recall": row.pooled_recall,
                    "pooled_fdr": row.pooled_fdr,
                    "fine_ids": row.fine_ids,
                }
                for name, row in ranking.per_coarse.items()
            },
            "not_a_platform_gate_because_taxonomy_is_incomplete": True,
        },
        "trace": trace,
    }


def _distance_to_internal_tile_edge(
    box: list[float], width: int, height: int, tile_size: int, overlap: int
) -> float:
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    tiles = generate_tiles(width, height, tile_size, overlap)
    x_lines = {
        value
        for tile in tiles
        for value in (tile.x_offset, tile.x_offset + tile.width)
        if 0 < value < width
    }
    y_lines = {
        value
        for tile in tiles
        for value in (tile.y_offset, tile.y_offset + tile.height)
        if 0 < value < height
    }
    return min(
        [abs(center_x - value) for value in x_lines]
        + [abs(center_y - value) for value in y_lines]
        + [math.inf]
    )


def _bucket(distance: float) -> str:
    if distance <= 8:
        return "le_8px"
    if distance <= 32:
        return "9_32px"
    if distance <= 128:
        return "33_128px"
    return "gt_128px_or_no_internal_edge"


def _paired_outcomes(
    document: dict[str, Any], gt: Any, direct_trace: Any, tiled_trace: Any
) -> dict[str, Any]:
    direct_hits = {
        (event.image_id, event.ground_truth_index) for event in direct_trace.matches
    }
    tiled_hits = {
        (event.image_id, event.ground_truth_index) for event in tiled_trace.matches
    }
    images = {
        int(row["id"]): (int(row["width"]), int(row["height"]))
        for row in document["images"]
    }
    counters: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    distances: dict[str, list[float]] = collections.defaultdict(list)
    for image_id, rows in gt.items():
        width, height = images[image_id]
        for index, row in enumerate(rows):
            key = (image_id, index)
            direct = key in direct_hits
            tiled = key in tiled_hits
            outcome = (
                "both"
                if direct and tiled
                else "direct_only"
                if direct
                else "tiled_only"
                if tiled
                else "neither"
            )
            distance = _distance_to_internal_tile_edge(
                row["bbox_xyxy"], width, height, 1024, 256
            )
            coarse = _coarse(int(row["category_id"]))
            counters[coarse][outcome] += 1
            counters["all"][outcome] += 1
            counters[f"distance:{_bucket(distance)}"][outcome] += 1
            if math.isfinite(distance):
                distances[outcome].append(distance)
    return {
        "counts": {name: dict(row) for name, row in sorted(counters.items())},
        "internal_edge_distance_median_px": {
            name: statistics.median(values) for name, values in sorted(distances.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--direct", type=Path, required=True)
    parser.add_argument("--tiled", type=Path, required=True)
    parser.add_argument("--direct-summary", type=Path)
    parser.add_argument("--tiled-summary", type=Path)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    document = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    gt = load_coco_ground_truth(args.ground_truth)
    direct = load_coco_predictions(args.direct)
    tiled = load_coco_predictions(args.tiled)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    direct_metrics = _metrics(gt, direct, protocol)
    tiled_metrics = _metrics(gt, tiled, protocol)
    paired = _paired_outcomes(
        document, gt, direct_metrics.pop("trace"), tiled_metrics.pop("trace")
    )
    direct_pooled = direct_metrics["pooled"]
    tiled_pooled = tiled_metrics["pooled"]
    result: dict[str, Any] = {
        "status": "complete",
        "role": "native_continuous_whole_vs_production_tiling_diagnostic",
        "inputs_sha256": {
            "ground_truth": _sha256(args.ground_truth),
            "direct": _sha256(args.direct),
            "tiled": _sha256(args.tiled),
        },
        "inventory": {
            "images": len(document["images"]),
            "annotations": len(document["annotations"]),
        },
        "direct_whole_1280": direct_metrics,
        "production_safe_1024_overlap256": tiled_metrics,
        "tiled_minus_direct": {
            "pooled_recall": float(tiled_pooled["recall"]) - float(direct_pooled["recall"]),
            "pooled_fdr": float(tiled_pooled["fdr"]) - float(direct_pooled["fdr"]),
            "tp": int(tiled_pooled["tp"]) - int(direct_pooled["tp"]),
            "fp": int(tiled_pooled["fp"]) - int(direct_pooled["fp"]),
            "fn": int(tiled_pooled["fn"]) - int(direct_pooled["fn"]),
        },
        "paired_gt_outcomes": paired,
        "limits": [
            "This is a small public-data implementation probe, not a hidden-score predictor.",
            "The subset has no Vehicle ground truth, so it cannot validate Vehicle tiling behaviour.",
            "Direct and tiled runs also differ in detector input scale for images above 1024 pixels.",
        ],
    }
    for name, path in (
        ("direct_runtime", args.direct_summary),
        ("tiled_runtime", args.tiled_summary),
    ):
        if path is not None:
            result[name] = json.loads(path.read_text(encoding="utf-8"))
            result["inputs_sha256"][name] = _sha256(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
