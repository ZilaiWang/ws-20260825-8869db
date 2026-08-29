#!/usr/bin/env python3
"""Cross-fit one score threshold per coarse category on pseudo-10K.

The competition reports ship, aircraft and vehicle independently.  A single
global threshold is therefore an unnecessarily restrictive deployment rule.
For each held-out fold and coarse category this script selects its threshold
using only the other two folds, then evaluates the stitched held-out outputs
with the exact official matcher.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config

COARSE = ("ship", "aircraft", "vehicle")


def _filter(
    values: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    coarse: str,
    mapping: dict[int, str],
    threshold: float | None = None,
) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = {}
    for image_id in sorted(image_ids):
        result[image_id] = [
            item
            for item in values.get(image_id, [])
            if mapping[int(item["category_id"])] == coarse
            and (threshold is None or float(item["score"]) >= threshold)
        ]
    return result


def _sweep_thresholds(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    thresholds: list[float],
    *,
    coarse: str,
    mapping: dict[int, str],
    iou_threshold: float,
) -> list[tuple[float, Any]]:
    coarse_mapping = {key: value for key, value in mapping.items() if value == coarse}
    points: list[tuple[float, Any]] = []
    for threshold in thresholds:
        filtered = {
            image_id: [item for item in items if float(item["score"]) >= threshold]
            for image_id, items in pred.items()
        }
        metrics = evaluate_predictions(
            gt,
            filtered,
            class_names=[coarse],
            category_mapping=coarse_mapping,
            iou_thresholds={coarse: iou_threshold},
        )
        points.append((threshold, metrics))
    return points


def _select_threshold(
    points: list[tuple[float, Any]], target_fdr: float
) -> dict[str, float | int]:
    feasible = [point for point in points if point[1].fdr <= target_fdr]
    if feasible:
        threshold, metrics = max(
            feasible, key=lambda point: (point[1].recall, -point[1].fdr, point[0])
        )
    else:
        threshold, metrics = min(
            points, key=lambda point: (point[1].fdr, -point[1].recall, -point[0])
        )
    return {
        "threshold": float(threshold),
        "train_recall": float(metrics.recall),
        "train_fdr": float(metrics.fdr),
        "train_tp": int(metrics.details["tp"]),
        "train_fp": int(metrics.details["fp"]),
        "train_fn": int(metrics.details["fn"]),
    }


def _metrics(metrics: Any, ranking: Any) -> dict[str, Any]:
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "tp": metrics.details["tp"],
        "fp": metrics.details["fp"],
        "fn": metrics.details["fn"],
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
        "macro_recall": ranking.overall_recall,
        "macro_fdr": ranking.overall_fdr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--threshold-start", type=float, default=0.001)
    parser.add_argument("--threshold-stop", type=float, default=0.996)
    parser.add_argument("--threshold-step", type=float, default=0.005)
    parser.add_argument(
        "--fdr-levels", type=float, nargs="+", default=(0.10, 0.12, 0.15, 0.17, 0.20)
    )
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_images = {
        fold: {int(item["id"]) for item in raw_gt["images"] if int(item["fold"]) == fold}
        for fold in (0, 1, 2)
    }
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    thresholds = build_threshold_grid(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )
    all_images = set().union(*fold_images.values())
    results: dict[str, Any] = {}

    # Matching is the expensive operation.  A threshold sweep is independent
    # of the requested FDR level, so compute it once per fold/category and
    # reuse it for all operating points.
    training_points: dict[tuple[int, str], list[tuple[float, Any]]] = {}
    for held_out in (0, 1, 2):
        train_images = all_images - fold_images[held_out]
        for coarse in COARSE:
            train_gt = _filter(
                gt,
                image_ids=train_images,
                coarse=coarse,
                mapping=protocol.category_mapping,
            )
            train_pred = _filter(
                pred,
                image_ids=train_images,
                coarse=coarse,
                mapping=protocol.category_mapping,
            )
            training_points[(held_out, coarse)] = _sweep_thresholds(
                train_gt,
                train_pred,
                thresholds,
                coarse=coarse,
                mapping=protocol.category_mapping,
                iou_threshold=protocol.iou_thresholds[coarse],
            )

    for target_fdr in args.fdr_levels:
        selections: dict[str, dict[str, Any]] = {}
        stitched: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in all_images}
        for held_out in (0, 1, 2):
            selections[str(held_out)] = {}
            for coarse in COARSE:
                selected = _select_threshold(
                    training_points[(held_out, coarse)],
                    target_fdr=float(target_fdr),
                )
                selections[str(held_out)][coarse] = selected
                held = _filter(
                    pred,
                    image_ids=fold_images[held_out],
                    coarse=coarse,
                    mapping=protocol.category_mapping,
                    threshold=float(selected["threshold"]),
                )
                for image_id, items in held.items():
                    stitched[image_id].extend(items)

        metrics = evaluate_predictions(
            gt,
            stitched,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        ranking = evaluate_ranking_metrics(
            gt,
            stitched,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        results[f"{float(target_fdr):.3f}"] = {
            "crossfit_thresholds": selections,
            "crossfit": _metrics(metrics, ranking),
        }

    gate = results["0.150"]["crossfit"]
    payload = {
        "status": "complete",
        "protocol": "formal_cv3_two_folds_select_per_coarse_thresholds_v1",
        "threshold_grid": {
            "start": args.threshold_start,
            "stop": args.threshold_stop,
            "step": args.threshold_step,
        },
        "frontiers": results,
        "admission": {
            "target": "overall Recall>=0.90 and every coarse FDR<=0.15",
            "passed": bool(
                gate["recall"] >= 0.90
                and all(item["fdr"] <= 0.15 for item in gate["per_coarse"].values())
            ),
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
