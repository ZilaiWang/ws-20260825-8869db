#!/usr/bin/env python3
"""Nested dual-view-consistency reranker for frozen identity candidates.

The detector candidate set is not expanded.  Ship and vehicle identity
candidates are rescored from their raw score, geometry, and same-fine support
in the frozen dual-view output.  Aircraft keeps the raw detector score.  For
each held-out fold, inner one-fold-to-one-fold predictions select thresholds;
the final quality model is then fitted on both non-held folds.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_safe_novel_view_additions import (
    _fit_logistic,
    _predict_logistic,
    _select_addition_threshold,
)
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


def _candidate_rows(
    identity: dict[int, list[dict[str, Any]]],
    dual: dict[int, list[dict[str, Any]]],
    mapping: dict[int, str],
    min_score: float,
) -> dict[int, list[dict[str, Any]]]:
    dual_by_key: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for image_id, rows in dual.items():
        for row in rows:
            dual_by_key[(image_id, int(row["category_id"]))].append(row)
    output: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in identity}
    source_index = 0
    for image_id in sorted(identity):
        for row in identity[image_id]:
            if float(row["score"]) < min_score:
                continue
            supports = dual_by_key.get((image_id, int(row["category_id"])), ())
            best_iou = 0.0
            best_score = 0.0
            for support in supports:
                iou = compute_iou(row["bbox_xyxy"], support["bbox_xyxy"])
                if (iou, float(support["score"])) > (best_iou, best_score):
                    best_iou = iou
                    best_score = float(support["score"])
            output[image_id].append(
                {
                    **row,
                    "image_id": image_id,
                    "source_prediction_index": source_index,
                    "nearby_identity_score": best_score,
                    "novel_same_fine_iou": best_iou,
                    "coarse": mapping[int(row["category_id"])],
                }
            )
            source_index += 1
    return output


def _official_labels(
    gt: dict[int, list[dict[str, Any]]],
    candidates: dict[int, list[dict[str, Any]]],
    *,
    mapping: dict[int, str],
    iou_thresholds: dict[str, float],
) -> dict[int, int]:
    _, trace = evaluate_predictions_with_trace(
        gt,
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
        raise RuntimeError("official labels do not partition identity candidates")
    return {index: int(index in positive) for index in all_ids}


def _score_heldout(
    rows: list[dict[str, Any]],
    labels: dict[int, int],
    fold_by_image: dict[int, int],
    held_out: int,
    *,
    model_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    score_key = "score"
    if not model_enabled:
        calibration = [
            row for row in rows if fold_by_image[int(row["image_id"])] != held_out
        ]
        held = [row for row in rows if fold_by_image[int(row["image_id"])] == held_out]
        return calibration, held, score_key

    score_key = "quality_score"
    train_folds = [fold for fold in (0, 1, 2) if fold != held_out]
    calibration: list[dict[str, Any]] = []
    for calibration_fold in train_folds:
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
            {**row, score_key: float(score)}
            for row, score in zip(calibration_rows, scores, strict=True)
        )
    final_train = [
        row for row in rows if fold_by_image[int(row["image_id"])] != held_out
    ]
    held = [row for row in rows if fold_by_image[int(row["image_id"])] == held_out]
    scores = _predict_logistic(held, _fit_logistic(final_train, labels))
    held_scored = [
        {**row, score_key: float(score)}
        for row, score in zip(held, scores, strict=True)
    ]
    return calibration, held_scored, score_key


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
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--min-score", type=float, default=0.001)
    parser.add_argument(
        "--fdr-levels", type=float, nargs="+", default=(0.10, 0.12, 0.15, 0.20)
    )
    args = parser.parse_args()
    if not 0.0 <= args.min_score <= 1.0:
        raise ValueError("min-score must be within [0, 1]")

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(item["id"]): int(item["fold"]) for item in raw_gt["images"]}
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    identity = load_coco_predictions(args.identity)
    dual = load_coco_predictions(args.dual)
    candidates = _candidate_rows(
        identity, dual, protocol.category_mapping, args.min_score
    )
    labels = _official_labels(
        gt,
        candidates,
        mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    rows_by_coarse = {
        coarse: [
            row for rows in candidates.values() for row in rows if row["coarse"] == coarse
        ]
        for coarse in ("ship", "aircraft", "vehicle")
    }
    results: dict[str, Any] = {}
    exports: dict[str, list[dict[str, Any]]] = {}
    for target_fdr in args.fdr_levels:
        stitched: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in gt}
        selections: dict[str, dict[str, Any]] = {}
        for held_out in (0, 1, 2):
            selections[str(held_out)] = {}
            for coarse, coarse_rows in rows_by_coarse.items():
                calibration, held, score_key = _score_heldout(
                    coarse_rows,
                    labels,
                    fold_by_image,
                    held_out,
                    model_enabled=coarse != "aircraft",
                )
                selected = _select_addition_threshold(
                    calibration, labels, target_fdr, score_key=score_key
                )
                selections[str(held_out)][coarse] = selected
                for row in held:
                    if float(row[score_key]) < float(selected["threshold"]):
                        continue
                    stitched[int(row["image_id"])].append(
                        {**row, "score": float(row[score_key])}
                    )
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
        results[key] = {"thresholds": selections, "metrics": _metrics(metrics, ranking)}
        exports[key] = [
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

    distribution = {
        str(fold): {
            coarse: {
                "candidates": sum(
                    1
                    for row in rows_by_coarse[coarse]
                    if fold_by_image[int(row["image_id"])] == fold
                ),
                "official_tp": sum(
                    labels[int(row["source_prediction_index"])]
                    for row in rows_by_coarse[coarse]
                    if fold_by_image[int(row["image_id"])] == fold
                ),
            }
            for coarse in rows_by_coarse
        }
        for fold in (0, 1, 2)
    }
    payload = {
        "status": "complete",
        "protocol": "nested_dual_consistency_identity_rerank_v1",
        "min_score": args.min_score,
        "aircraft_policy": "raw_score_bypass",
        "ship_vehicle_policy": "nested_weighted_logistic_dual_consistency",
        "label_distribution": distribution,
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
    for key, rows in exports.items():
        args.output.with_name(f"{args.output.stem}_fdr{key}.predictions.json").write_text(
            json.dumps(rows, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
