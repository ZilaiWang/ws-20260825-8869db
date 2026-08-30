#!/usr/bin/env python3
"""Train and freeze the Hard-dev dual-consistency identity reranker.

The candidate set is the identity detector output only.  Ship and vehicle
quality models use same-fine support from the frozen identity+rot90 output;
aircraft is an explicit raw-score bypass.  Every development prediction gets
an out-of-fold score before workpoint thresholds are selected.  Final models
are then fitted on all development rows and serialized for label-free target
application.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_dual_consistency_rerank import (
    _candidate_rows,
    _metrics,
    _official_labels,
)
from analyze_safe_novel_view_additions import _select_addition_threshold
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.innovation.dual_consistency import (
    FEATURE_CONTRACTS,
    blend_probability,
    fit_logistic,
    predict_logistic,
    serialize_model,
)
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction_export(predictions: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
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
        for image_id in sorted(predictions)
        for row in predictions[image_id]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--dual", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    parser.add_argument("--target-fdr", type=float, default=0.15)
    parser.add_argument("--min-score", type=float, default=0.001)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument(
        "--feature-contract",
        choices=tuple(FEATURE_CONTRACTS),
        default="quality_features_v1_12d",
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if not 0.0 < args.target_fdr < 1.0:
        raise ValueError("target-fdr must be within (0, 1)")
    if not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError("blend-alpha must be within [0, 1]")

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(item["id"]): int(item["fold"]) for item in raw_gt["images"]}
    if set(fold_by_image.values()) != {0, 1, 2}:
        raise ValueError("development ground truth must contain folds 0, 1 and 2")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    identity = load_coco_predictions(args.identity)
    dual = load_coco_predictions(args.dual)
    candidates = _candidate_rows(identity, dual, protocol.category_mapping, args.min_score)
    labels = _official_labels(
        gt,
        candidates,
        mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    rows_by_coarse = {
        coarse: [
            row
            for rows in candidates.values()
            for row in rows
            if row["coarse"] == coarse
        ]
        for coarse in ("ship", "aircraft", "vehicle")
    }

    stitched: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in gt}
    baseline_stitched: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in gt}
    frozen: dict[str, Any] = {}
    raw_baseline: dict[str, Any] = {}
    threshold_diagnostics: dict[str, Any] = {}
    raw_threshold_diagnostics: dict[str, Any] = {}
    for coarse, rows in rows_by_coarse.items():
        oof_rows: list[dict[str, Any]] = []
        for held_out in (0, 1, 2):
            train = [row for row in rows if fold_by_image[int(row["image_id"])] != held_out]
            held = [row for row in rows if fold_by_image[int(row["image_id"])] == held_out]
            if coarse == "aircraft":
                scores = np.asarray([float(row["score"]) for row in held], dtype=np.float64)
            else:
                quality_scores = predict_logistic(
                    held,
                    fit_logistic(train, labels, args.feature_contract),
                    args.feature_contract,
                )
                scores = blend_probability(
                    np.asarray([float(row["score"]) for row in held]),
                    quality_scores,
                    args.blend_alpha,
                )
            oof_rows.extend(
                {**row, "quality_score": float(score)}
                for row, score in zip(held, scores, strict=True)
            )
        oof_by_image: dict[int, list[dict[str, Any]]] = {
            image_id: [] for image_id in gt
        }
        for row in oof_rows:
            oof_by_image[int(row["image_id"])].append(
                {**row, "score": float(row["quality_score"])}
            )
        quality_labels = _official_labels(
            gt,
            oof_by_image,
            mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        selected = _select_addition_threshold(
            oof_rows,
            quality_labels,
            args.target_fdr,
            score_key="quality_score",
        )
        threshold = float(selected["threshold"])
        threshold_diagnostics[coarse] = selected
        for row in oof_rows:
            if float(row["quality_score"]) >= threshold:
                stitched[int(row["image_id"])].append(
                    {**row, "score": float(row["quality_score"])}
                )
        frozen[coarse] = {
            "policy": "raw_score_bypass" if coarse == "aircraft" else "dual_consistency_logistic",
            "threshold": threshold,
            "model": (
                None
                if coarse == "aircraft"
                else serialize_model(
                    fit_logistic(rows, labels, args.feature_contract)
                )
            ),
            "development_candidates": len(rows),
            "development_official_tp": sum(
                labels[int(row["source_prediction_index"])] for row in rows
            ),
        }
        raw_rows = [{**row, "quality_score": float(row["score"])} for row in rows]
        raw_selected = _select_addition_threshold(
            raw_rows,
            labels,
            args.target_fdr,
            score_key="quality_score",
        )
        raw_threshold = float(raw_selected["threshold"])
        raw_threshold_diagnostics[coarse] = raw_selected
        for row in raw_rows:
            if float(row["quality_score"]) >= raw_threshold:
                baseline_stitched[int(row["image_id"])].append(
                    {**row, "score": float(row["quality_score"])}
                )
        raw_baseline[coarse] = {
            "policy": "raw_score_bypass",
            "threshold": raw_threshold,
            "model": None,
            "development_candidates": len(rows),
            "development_official_tp": sum(
                labels[int(row["source_prediction_index"])] for row in rows
            ),
        }

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
    baseline_metrics = evaluate_predictions(
        gt,
        baseline_stitched,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    baseline_ranking = evaluate_ranking_metrics(
        gt,
        baseline_stitched,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    model_payload = {
        "status": "complete",
        "protocol": "hard_dev_oof_threshold_final_full_fit_dual_consistency_v1",
        "target_fdr": args.target_fdr,
        "min_score": args.min_score,
        "feature_contract": args.feature_contract,
        "blend_alpha": args.blend_alpha,
        "candidate_policy": "identity_only_no_expansion",
        "dual_semantics": "identity_plus_rot90_same_weight_support",
        "coarse": frozen,
        "raw_baseline_coarse": raw_baseline,
        "development_oof_metrics": _metrics(metrics, ranking),
        "development_raw_baseline_metrics": _metrics(
            baseline_metrics, baseline_ranking
        ),
        "threshold_diagnostics": threshold_diagnostics,
        "raw_threshold_diagnostics": raw_threshold_diagnostics,
        "input_sha256": {
            "gt": _sha256(args.gt),
            "identity": _sha256(args.identity),
            "dual": _sha256(args.dual),
        },
    }
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.model_output.write_text(
        json.dumps(model_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.oof_output.parent.mkdir(parents=True, exist_ok=True)
    args.oof_output.write_text(
        json.dumps(_prediction_export(stitched), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(model_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
