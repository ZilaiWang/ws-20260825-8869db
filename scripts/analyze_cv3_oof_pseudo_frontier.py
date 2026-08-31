#!/usr/bin/env python3
"""Cross-fit score thresholds on three formal-CV3 pseudo-10K folds.

For each held-out fold, the score threshold is selected using only the other
two folds.  The three independently thresholded held-out ledgers are then
evaluated together with the exact official matcher.  A pooled-oracle frontier
is reported separately as an optimistic diagnostic, never as the admission
result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import build_threshold_curve
from rsdet.evaluation.absolute_score import competition_score, score_coarse_interpretations
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config

T = TypeVar("T")


def _scoped(mapping: dict[int, list[T]], image_ids: set[int]) -> dict[int, list[T]]:
    return {image_id: list(mapping.get(image_id, [])) for image_id in sorted(image_ids)}


def _select_threshold(points: list[dict[str, Any]], target_fdr: float) -> dict[str, Any]:
    feasible = [point for point in points if float(point["overall_fdr"]) <= target_fdr]
    if not feasible:
        return min(
            points,
            key=lambda point: (
                float(point["overall_fdr"]),
                -float(point["threshold"]),
            ),
        )
    return max(
        feasible,
        key=lambda point: (
            float(point["overall_recall"]),
            -float(point["overall_fdr"]),
            float(point["threshold"]),
        ),
    )


def _select_absolute_score_threshold(
    points: list[dict[str, Any]],
    *,
    latency_seconds: float,
    minimum_recall: float,
    maximum_fdr: float,
) -> dict[str, Any]:
    feasible = [
        point
        for point in points
        if float(point["overall_recall"]) >= minimum_recall
        and float(point["overall_fdr"]) <= maximum_fdr
    ]
    pool = feasible or points
    selected = max(
        pool,
        key=lambda point: (
            competition_score(
                float(point["overall_recall"]),
                float(point["overall_fdr"]),
                latency_seconds,
            )["total_score"],
            float(point["overall_recall"]),
            -float(point["overall_fdr"]),
            float(point["threshold"]),
        ),
    )
    return {
        **selected,
        "constraint_feasible": bool(feasible),
        "selection_score": competition_score(
            float(selected["overall_recall"]),
            float(selected["overall_fdr"]),
            latency_seconds,
        ),
    }


def _metric_payload(metrics: Any, ranking: Any) -> dict[str, Any]:
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
        "ranking_per_coarse": {
            name: {
                "macro_recall": item.macro_recall,
                "macro_fdr": item.macro_fdr,
            }
            for name, item in ranking.per_coarse.items()
        },
    }


def _admission_payload(
    results: dict[str, Any], fdr_levels: list[float] | tuple[float, ...]
) -> dict[str, Any]:
    """Build admission metadata for any non-empty requested FDR grid."""

    if not fdr_levels:
        raise ValueError("at least one FDR level is required")
    admission_key = f"{min(fdr_levels, key=lambda value: abs(value - 0.15)):.3f}"
    stretch_key = f"{min(fdr_levels):.3f}"
    gate = results[admission_key]["crossfit"]
    stretch = results[stretch_key]["crossfit"]
    return {
        "target": f"crossfit Recall>=0.90 and FDR<={float(admission_key):.3f}",
        "selected_level": float(admission_key),
        "passed": bool(
            gate["recall"] >= 0.90 and gate["fdr"] <= float(admission_key)
        ),
        "stretch_target": (
            f"crossfit Recall>=0.95 and FDR<={float(stretch_key):.3f}"
        ),
        "stretch_selected_level": float(stretch_key),
        "stretch_passed": bool(
            stretch["recall"] >= 0.95 and stretch["fdr"] <= float(stretch_key)
        ),
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
    parser.add_argument("--latency-seconds", type=float)
    parser.add_argument("--score-minimum-recall", type=float, default=0.87)
    parser.add_argument("--score-maximum-fdr", type=float, default=0.18)
    args = parser.parse_args()

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_images = {
        fold: {int(item["id"]) for item in raw_gt["images"] if int(item["fold"]) == fold}
        for fold in (0, 1, 2)
    }
    if any(not image_ids for image_ids in fold_images.values()):
        raise ValueError("ground truth must contain non-empty folds 0, 1 and 2")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    pred = load_coco_predictions(args.pred)
    thresholds = build_threshold_grid(
        args.threshold_start, args.threshold_stop, args.threshold_step
    )

    training_points: dict[int, list[dict[str, Any]]] = {}
    trace_audits: dict[str, Any] = {}
    for held_out in (0, 1, 2):
        train_ids = set().union(
            *(image_ids for fold, image_ids in fold_images.items() if fold != held_out)
        )
        training_points[held_out], trace_audits[f"train_for_fold_{held_out}"] = (
            build_threshold_curve(
                _scoped(gt, train_ids),
                _scoped(pred, train_ids),
                thresholds=thresholds,
                protocol=protocol,
            )
        )
    pooled_points, trace_audits["pooled"] = build_threshold_curve(
        gt, pred, thresholds=thresholds, protocol=protocol
    )

    results: dict[str, Any] = {}
    for target_fdr in args.fdr_levels:
        chosen: dict[int, float] = {}
        crossfit_predictions: dict[int, list[dict[str, Any]]] = {}
        for held_out in (0, 1, 2):
            selected = _select_threshold(training_points[held_out], float(target_fdr))
            chosen[held_out] = float(selected["threshold"])
            for image_id in fold_images[held_out]:
                crossfit_predictions[image_id] = [
                    item
                    for item in pred.get(image_id, [])
                    if float(item["score"]) >= float(selected["threshold"])
                ]

        metrics = evaluate_predictions(
            gt,
            crossfit_predictions,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        ranking = evaluate_ranking_metrics(
            gt,
            crossfit_predictions,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )

        pooled_selected = _select_threshold(pooled_points, float(target_fdr))
        pooled_threshold = float(pooled_selected["threshold"])
        pooled_predictions = {
            image_id: [
                item
                for item in pred.get(image_id, [])
                if float(item["score"]) >= pooled_threshold
            ]
            for image_id in gt
        }
        pooled_ranking = evaluate_ranking_metrics(
            gt,
            pooled_predictions,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        results[f"{target_fdr:.3f}"] = {
            "crossfit_thresholds": {str(key): value for key, value in chosen.items()},
            "crossfit": _metric_payload(metrics, ranking),
            "pooled_oracle": {
                "threshold": pooled_threshold,
                "recall": float(pooled_selected["overall_recall"]),
                "fdr": float(pooled_selected["overall_fdr"]),
                "macro_recall": pooled_ranking.overall_recall,
                "macro_fdr": pooled_ranking.overall_fdr,
            },
        }

    floor_metrics = evaluate_predictions(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    floor_ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    absolute_score_crossfit = None
    if args.latency_seconds is not None:
        chosen: dict[int, float] = {}
        training_selections: dict[str, Any] = {}
        score_predictions: dict[int, list[dict[str, Any]]] = {}
        for held_out in (0, 1, 2):
            selected = _select_absolute_score_threshold(
                training_points[held_out],
                latency_seconds=args.latency_seconds,
                minimum_recall=args.score_minimum_recall,
                maximum_fdr=args.score_maximum_fdr,
            )
            chosen[held_out] = float(selected["threshold"])
            training_selections[str(held_out)] = selected
            for image_id in fold_images[held_out]:
                score_predictions[image_id] = [
                    item
                    for item in pred.get(image_id, [])
                    if float(item["score"]) >= float(selected["threshold"])
                ]
        score_metrics = evaluate_predictions(
            gt,
            score_predictions,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        score_ranking = evaluate_ranking_metrics(
            gt,
            score_predictions,
            class_names=protocol.class_names,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
            require_complete_taxonomy=True,
        )
        score_payload = _metric_payload(score_metrics, score_ranking)
        pooled_selected = _select_absolute_score_threshold(
            pooled_points,
            latency_seconds=args.latency_seconds,
            minimum_recall=args.score_minimum_recall,
            maximum_fdr=args.score_maximum_fdr,
        )
        absolute_score_crossfit = {
            "selection_contract": {
                "nested_two_folds_select_one_fold_apply": True,
                "latency_seconds": args.latency_seconds,
                "minimum_recall": args.score_minimum_recall,
                "maximum_fdr": args.score_maximum_fdr,
                "objective": "maximize published absolute score within constraints",
            },
            "crossfit_thresholds": {str(key): value for key, value in chosen.items()},
            "training_selections": training_selections,
            "crossfit": score_payload,
            "crossfit_score_interpretations": score_coarse_interpretations(
                score_payload["per_coarse"], args.latency_seconds
            ),
            "pooled_oracle": pooled_selected,
            "pooled_oracle_is_not_admissible": True,
        }
    payload = {
        "status": "complete",
        "protocol": "formal_cv3_two_folds_select_one_fold_evaluate_pseudo10k_v1",
        "warning": (
            "Pseudo-10K is a deployment proxy, not an independent benchmark; "
            "pooled_oracle is diagnostic only."
        ),
        "threshold_grid": {
            "start": args.threshold_start,
            "stop": args.threshold_stop,
            "step": args.threshold_step,
        },
        "score_prefix_trace_audits": trace_audits,
        "fold_image_ids": {str(key): sorted(value) for key, value in fold_images.items()},
        "candidate_floor": _metric_payload(floor_metrics, floor_ranking),
        "frontiers": results,
        "admission": _admission_payload(results, args.fdr_levels),
        "absolute_score_crossfit": absolute_score_crossfit,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
