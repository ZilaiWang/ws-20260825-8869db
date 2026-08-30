#!/usr/bin/env python3
"""Cross-fit a safe additive gate for novel ship/vehicle dual-view candidates.

Identity predictions above the frozen incumbent workpoint are never removed.
Only dual-view candidates without same-fine identity support are eligible for
addition.  Addition score thresholds are fitted on the other two folds using
their marginal residual-TP/FP labels, then evaluated with the exact official
matcher on the held-out fold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import (
    compute_iou,
    evaluate_predictions,
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _thresholds(args: argparse.Namespace) -> dict[str, float]:
    values = {
        "ship": args.threshold_ship,
        "aircraft": args.threshold_aircraft,
        "vehicle": args.threshold_vehicle,
    }
    if any(not 0.0 <= value <= 1.0 for value in values.values()):
        raise ValueError("incumbent thresholds must be within [0, 1]")
    return values


def _filter_incumbent(
    pred: dict[int, list[dict[str, Any]]],
    mapping: dict[int, str],
    thresholds: dict[str, float],
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            dict(item)
            for item in rows
            if float(item["score"]) >= thresholds[mapping[int(item["category_id"])]]
        ]
        for image_id, rows in pred.items()
    }


def _view_candidates(
    identity: dict[int, list[dict[str, Any]]],
    dual: dict[int, list[dict[str, Any]]],
    mapping: dict[int, str],
    support_iou: float,
    duplicate_iou: float,
    candidate_mode: str,
) -> dict[int, list[dict[str, Any]]]:
    by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for image_id, rows in identity.items():
        for row in rows:
            by_key[(image_id, int(row["category_id"]))].append(row)
    output: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in dual}
    candidate_index = 0
    for image_id in sorted(dual):
        for row in dual[image_id]:
            coarse = mapping[int(row["category_id"])]
            if coarse == "aircraft":
                continue
            supports = by_key.get((image_id, int(row["category_id"])), ())
            best_iou = max(
                (compute_iou(row["bbox_xyxy"], item["bbox_xyxy"]) for item in supports),
                default=0.0,
            )
            eligible = best_iou < support_iou
            if candidate_mode == "supported":
                eligible = support_iou <= best_iou < duplicate_iou
            if not eligible:
                continue
            best_score = max(
                (
                    float(item["score"])
                    for item in supports
                    if compute_iou(row["bbox_xyxy"], item["bbox_xyxy"]) >= 0.05
                ),
                default=0.0,
            )
            output[image_id].append(
                {
                    **row,
                    "image_id": image_id,
                    "source_prediction_index": candidate_index,
                    "novel_same_fine_iou": best_iou,
                    "nearby_identity_score": best_score,
                    "coarse": coarse,
                }
            )
            candidate_index += 1
    return output


def _residual_labels(
    gt: dict[int, list[dict[str, Any]]],
    incumbent: dict[int, list[dict[str, Any]]],
    candidates: dict[int, list[dict[str, Any]]],
    *,
    mapping: dict[int, str],
    iou_thresholds: dict[str, float],
) -> dict[int, int]:
    _, baseline_trace = evaluate_predictions_with_trace(
        gt,
        incumbent,
        category_mapping=mapping,
        iou_thresholds=iou_thresholds,
    )
    missed = {
        (item.image_id, item.ground_truth_index)
        for item in baseline_trace.unmatched_ground_truths
    }
    residual_gt = {
        image_id: [
            row for index, row in enumerate(rows) if (image_id, index) in missed
        ]
        for image_id, rows in gt.items()
    }
    _, trace = evaluate_predictions_with_trace(
        residual_gt,
        candidates,
        category_mapping=mapping,
        iou_thresholds=iou_thresholds,
    )
    positive = {int(item.prediction_index) for item in trace.matches}
    negative = {int(item.prediction_index) for item in trace.unmatched_predictions}
    all_ids = {
        int(row["source_prediction_index"])
        for rows in candidates.values()
        for row in rows
    }
    if positive & negative or positive | negative != all_ids:
        raise RuntimeError("residual candidate labels do not partition candidates")
    return {index: int(index in positive) for index in all_ids}


def _select_addition_threshold(
    rows: list[dict[str, Any]],
    labels: dict[int, int],
    target_fdr: float,
    *,
    score_key: str = "score",
) -> dict[str, float | int]:
    if not rows:
        return {"threshold": 1.000001, "tp": 0, "fp": 0, "fdr": 0.0}
    scores = sorted({float(row[score_key]) for row in rows}, reverse=True)
    points = []
    for threshold in scores:
        selected = [row for row in rows if float(row[score_key]) >= threshold]
        tp = sum(labels[int(row["source_prediction_index"])] for row in selected)
        fp = len(selected) - tp
        fdr = fp / len(selected) if selected else 0.0
        points.append((threshold, tp, fp, fdr))
    feasible = [point for point in points if point[3] <= target_fdr]
    if not feasible:
        return {"threshold": 1.000001, "tp": 0, "fp": 0, "fdr": 0.0}
    threshold, tp, fp, fdr = max(
        feasible, key=lambda point: (point[1], -point[2], point[0])
    )
    return {"threshold": threshold, "tp": tp, "fp": fp, "fdr": fdr}


def _quality_features(rows: list[dict[str, Any]]) -> np.ndarray:
    output = []
    for row in rows:
        score = float(np.clip(row["score"], 1e-5, 1.0 - 1e-5))
        nearby = float(np.clip(row["nearby_identity_score"], 1e-5, 1.0 - 1e-5))
        x0, y0, x1, y1 = (float(value) for value in row["bbox_xyxy"])
        width = max(x1 - x0, 1e-3)
        height = max(y1 - y0, 1e-3)
        category = int(row["category_id"])
        category_slot = category if category <= 3 else 4
        output.append(
            [
                math.log(score / (1.0 - score)),
                math.log(nearby / (1.0 - nearby)),
                score - nearby,
                float(row["novel_same_fine_iou"]),
                math.log1p(width),
                math.log1p(height),
                math.log(width / height),
                *(1.0 if category_slot == index else 0.0 for index in range(5)),
            ]
        )
    return np.asarray(output, dtype=np.float64)


def _fit_logistic(
    rows: list[dict[str, Any]], labels: dict[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    if not rows:
        return np.zeros(12), np.ones(12), np.zeros(12), -20.0
    matrix = _quality_features(rows)
    target = np.asarray(
        [labels[int(row["source_prediction_index"])] for row in rows], dtype=np.float64
    )
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (matrix - mean) / scale
    positives = float(target.sum())
    if positives == 0.0:
        return mean, scale, np.zeros(matrix.shape[1]), -20.0
    positive_weight = min(30.0, (len(target) - positives) / positives)
    sample_weight = np.where(target > 0.5, positive_weight, 1.0)
    weight = np.zeros(matrix.shape[1], dtype=np.float64)
    bias = math.log((positives + 0.5) / (len(target) - positives + 0.5))
    for step in range(800):
        logit = np.clip(normalized @ weight + bias, -30.0, 30.0)
        probability = 1.0 / (1.0 + np.exp(-logit))
        error = (probability - target) * sample_weight
        learning_rate = 0.08 / math.sqrt(1.0 + step / 80.0)
        weight -= learning_rate * (normalized.T @ error / len(target) + 0.01 * weight)
        bias -= learning_rate * float(error.mean())
    return mean, scale, weight, float(bias)


def _predict_logistic(
    rows: list[dict[str, Any]], model: tuple[np.ndarray, np.ndarray, np.ndarray, float]
) -> np.ndarray:
    if not rows:
        return np.empty(0, dtype=np.float64)
    mean, scale, weight, bias = model
    logit = np.clip(((_quality_features(rows) - mean) / scale) @ weight + bias, -30, 30)
    return 1.0 / (1.0 + np.exp(-logit))


def _nested_quality_scores(
    rows: list[dict[str, Any]],
    labels: dict[int, int],
    fold_by_image: dict[int, int],
    held_out: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    training_folds = [fold for fold in (0, 1, 2) if fold != held_out]
    calibration: list[dict[str, Any]] = []
    for calibration_fold in training_folds:
        fit_rows = [
            row
            for row in rows
            if fold_by_image[int(row["image_id"])]
            not in {held_out, calibration_fold}
        ]
        calibration_rows = [
            row
            for row in rows
            if fold_by_image[int(row["image_id"])] == calibration_fold
        ]
        scores = _predict_logistic(calibration_rows, _fit_logistic(fit_rows, labels))
        calibration.extend(
            {**row, "quality_score": float(score)}
            for row, score in zip(calibration_rows, scores, strict=True)
        )
    fit_rows = [
        row for row in rows if fold_by_image[int(row["image_id"])] != held_out
    ]
    held_rows = [
        row for row in rows if fold_by_image[int(row["image_id"])] == held_out
    ]
    held_scores = _predict_logistic(held_rows, _fit_logistic(fit_rows, labels))
    held = [
        {**row, "quality_score": float(score)}
        for row, score in zip(held_rows, held_scores, strict=True)
    ]
    return calibration, held


def _metric_payload(metrics: Any, ranking: Any) -> dict[str, Any]:
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
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--threshold-ship", type=float, default=0.15)
    parser.add_argument("--threshold-aircraft", type=float, default=0.15)
    parser.add_argument("--threshold-vehicle", type=float, default=0.15)
    parser.add_argument("--support-iou", type=float, default=0.25)
    parser.add_argument("--duplicate-iou", type=float, default=0.50)
    parser.add_argument(
        "--candidate-mode", choices=("novel", "supported"), default="novel"
    )
    parser.add_argument("--ranking-score", choices=("raw", "logistic"), default="raw")
    parser.add_argument(
        "--addition-fdr-levels", type=float, nargs="+", default=(0.10, 0.15, 0.20)
    )
    args = parser.parse_args()
    if not 0.0 <= args.support_iou <= 1.0:
        raise ValueError("support-iou must be within [0, 1]")
    if not args.support_iou < args.duplicate_iou <= 1.0:
        raise ValueError("duplicate-iou must be in (support-iou, 1]")
    if any(not 0.0 <= value <= 1.0 for value in args.addition_fdr_levels):
        raise ValueError("addition FDR levels must be within [0, 1]")

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(item["id"]): int(item["fold"]) for item in raw_gt["images"]}
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    identity = load_coco_predictions(args.identity)
    dual = load_coco_predictions(args.dual)
    incumbent_thresholds = _thresholds(args)
    incumbent = _filter_incumbent(identity, protocol.category_mapping, incumbent_thresholds)
    novel = _view_candidates(
        identity,
        dual,
        protocol.category_mapping,
        args.support_iou,
        args.duplicate_iou,
        args.candidate_mode,
    )
    labels = _residual_labels(
        gt,
        incumbent,
        novel,
        mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )

    base_metrics = evaluate_predictions(
        gt,
        incumbent,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    base_ranking = evaluate_ranking_metrics(
        gt,
        incumbent,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    image_ids = set(gt)
    results: dict[str, Any] = {}
    exported: dict[str, list[dict[str, Any]]] = {}
    for target_fdr in args.addition_fdr_levels:
        thresholds_by_fold: dict[str, dict[str, Any]] = {}
        stitched = {image_id: list(incumbent.get(image_id, ())) for image_id in image_ids}
        selected_count = 0
        for held_out in (0, 1, 2):
            thresholds_by_fold[str(held_out)] = {}
            for coarse in ("ship", "vehicle"):
                all_coarse_rows = [
                    row
                    for image_id, rows in novel.items()
                    for row in rows
                    if row["coarse"] == coarse
                ]
                score_key = "score"
                if args.ranking_score == "logistic":
                    train_rows, held_rows = _nested_quality_scores(
                        all_coarse_rows, labels, fold_by_image, held_out
                    )
                    score_key = "quality_score"
                else:
                    train_rows = [
                        row
                        for row in all_coarse_rows
                        if fold_by_image[int(row["image_id"])] != held_out
                    ]
                    held_rows = [
                        row
                        for row in all_coarse_rows
                        if fold_by_image[int(row["image_id"])] == held_out
                    ]
                selected = _select_addition_threshold(
                    train_rows, labels, target_fdr, score_key=score_key
                )
                thresholds_by_fold[str(held_out)][coarse] = selected
                for image_id in sorted(image_ids):
                    additions = [
                        row
                        for row in held_rows
                        if int(row["image_id"]) == image_id
                        and float(row[score_key]) >= float(selected["threshold"])
                    ]
                    stitched[image_id].extend(additions)
                    selected_count += len(additions)
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
        key = f"{target_fdr:.3f}"
        results[key] = {
            "addition_thresholds": thresholds_by_fold,
            "selected_additions": selected_count,
            "metrics": _metric_payload(metrics, ranking),
        }
        exported[key] = [
            {
                "image_id": image_id,
                "category_id": int(row["category_id"]),
                "bbox": [
                    float(row["bbox_xyxy"][0]),
                    float(row["bbox_xyxy"][1]),
                    float(row["bbox_xyxy"][2] - row["bbox_xyxy"][0]),
                    float(row["bbox_xyxy"][3] - row["bbox_xyxy"][1]),
                ],
                "score": float(row["score"]),
            }
            for image_id in sorted(stitched)
            for row in stitched[image_id]
        ]

    novel_rows = [row for rows in novel.values() for row in rows]
    label_distribution = {
        str(fold): {
            coarse: {
                "candidates": sum(
                    1
                    for image_id, rows in novel.items()
                    if fold_by_image[image_id] == fold
                    for row in rows
                    if row["coarse"] == coarse
                ),
                "residual_tp": sum(
                    labels[int(row["source_prediction_index"])]
                    for image_id, rows in novel.items()
                    if fold_by_image[image_id] == fold
                    for row in rows
                    if row["coarse"] == coarse
                ),
            }
            for coarse in ("ship", "vehicle")
        }
        for fold in (0, 1, 2)
    }
    payload = {
        "status": "complete",
        "protocol": "identity_preserving_crossfit_view_additions_v1",
        "candidate_mode": args.candidate_mode,
        "ranking_score": args.ranking_score,
        "incumbent_thresholds": incumbent_thresholds,
        "support_iou": args.support_iou,
        "duplicate_iou": args.duplicate_iou,
        "baseline": _metric_payload(base_metrics, base_ranking),
        "novel_candidates": len(novel_rows),
        "residual_tp_candidates": sum(labels.values()),
        "residual_fp_candidates": len(labels) - sum(labels.values()),
        "label_distribution": label_distribution,
        "frontiers": results,
        "input_sha256": {
            "gt": _sha256(args.gt),
            "identity": _sha256(args.identity),
            "dual": _sha256(args.dual),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for key, rows in exported.items():
        args.output.with_name(f"{args.output.stem}_fdr{key}.predictions.json").write_text(
            json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
