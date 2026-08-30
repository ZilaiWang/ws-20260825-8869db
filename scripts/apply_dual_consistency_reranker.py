#!/usr/bin/env python3
"""Apply a frozen Hard-dev dual-consistency reranker without target fitting."""

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

from analyze_dual_consistency_rerank import _candidate_rows, _metrics
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.innovation.dual_consistency import (
    FEATURE_CONTRACTS,
    blend_probability,
    deserialize_model,
    predict_logistic,
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
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=("reranker", "raw-baseline"), default="reranker"
    )
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    frozen = json.loads(args.model.read_text(encoding="utf-8"))
    feature_contract = frozen.get("feature_contract")
    if frozen.get("status") != "complete" or feature_contract not in FEATURE_CONTRACTS:
        raise ValueError("unsupported or incomplete frozen reranker")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    identity = load_coco_predictions(args.identity)
    dual = load_coco_predictions(args.dual)
    candidates = _candidate_rows(
        identity,
        dual,
        protocol.category_mapping,
        float(frozen["min_score"]),
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
    selected: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in gt}
    counts: dict[str, Any] = {}
    policy_root = (
        frozen["coarse"]
        if args.policy == "reranker"
        else frozen["raw_baseline_coarse"]
    )
    for coarse, rows in rows_by_coarse.items():
        policy = policy_root[coarse]
        threshold = float(policy["threshold"])
        if policy["policy"] == "raw_score_bypass":
            scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        elif policy["policy"] == "dual_consistency_logistic":
            quality_scores = predict_logistic(
                rows,
                deserialize_model(policy["model"], feature_contract),
                feature_contract,
            )
            scores = blend_probability(
                np.asarray([float(row["score"]) for row in rows]),
                quality_scores,
                float(frozen.get("blend_alpha", 1.0)),
            )
        else:
            raise ValueError(f"unsupported policy for {coarse}: {policy['policy']}")
        kept = 0
        for row, score in zip(rows, scores, strict=True):
            if float(score) < threshold:
                continue
            selected[int(row["image_id"])].append({**row, "score": float(score)})
            kept += 1
        counts[coarse] = {"candidates": len(rows), "selected": kept, "threshold": threshold}

    metrics = evaluate_predictions(
        gt,
        selected,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        gt,
        selected,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    payload = {
        "status": "complete",
        "protocol": "frozen_hard_dev_dual_consistency_target_application_v1",
        "selection_policy": args.policy,
        "target_label_usage": "evaluation_only_no_fit_no_threshold_selection",
        "selection_counts": counts,
        "metrics": _metrics(metrics, ranking),
        "input_sha256": {
            "gt": _sha256(args.gt),
            "identity": _sha256(args.identity),
            "dual": _sha256(args.dual),
            "model": _sha256(args.model),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
    args.predictions_output.write_text(
        json.dumps(_prediction_export(selected), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
