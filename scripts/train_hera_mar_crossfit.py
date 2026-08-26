#!/usr/bin/env python3
"""Cheap outer-fold MAR proxy used only to decide whether nested stacking is worthwhile."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_frontier import official_fixed_risk_frontier
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.mar_training import (
    MAR_FEATURE_NAMES,
    build_mar_features,
    fit_monotone_mar,
)
from rsdet.utils.config import load_config

COARSE_INDEX = {"aircraft": 0, "ship": 1, "vehicle": 2}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metrics(frontier, *, formal, image_ids, protocol) -> dict:
    points = {
        str(level): {
            "recall": point.recall,
            "fdr": point.fdr,
            "tp": point.tp,
            "fp": point.fp,
        }
        for level, point in frontier.points.items()
    }
    selected = set(frontier.selected_candidate_ids[0.12])
    selected_by_image = {
        image_id: [
            row
            for row in frontier.kept_predictions[image_id]
            if int(row["source_prediction_index"]) in selected
        ]
        for image_id in image_ids
    }
    ranking = evaluate_ranking_metrics(
        {image_id: formal.boxes.get(image_id, []) for image_id in image_ids},
        selected_by_image,
        class_names=list(dict.fromkeys(protocol.category_mapping.values())),
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    coarse = {}
    six = []
    for name in ("ship", "aircraft", "vehicle"):
        item = ranking.per_coarse[name]
        coarse[name] = {
            "macro_recall": item.macro_recall,
            "macro_fdr": item.macro_fdr,
            "pooled_recall": item.pooled_recall,
            "pooled_fdr": item.pooled_fdr,
        }
        six.extend((item.macro_recall, 1.0 - item.macro_fdr))
    return {
        "frontier": points,
        "ranking_at_fdr_0.12": {
            "per_coarse": coarse,
            "six_metric_min": min(six),
            "six_metric_mean": sum(six) / len(six),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--pav-logits", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--rho-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=202625)
    args = parser.parse_args()

    with args.manifest.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    pav = np.load(args.pav_logits, allow_pickle=False)
    logits = {key: pav[key] for key in pav.files}
    features = build_mar_features(rows=rows, predictions=predictions, logits=logits)
    folds = np.asarray([int(row["fold"]) for row in rows], dtype=np.int64)
    roles = np.asarray([row["workpoint_role"] for row in rows], dtype=object)
    target = np.asarray([int(row["target_foreground"]) for row in rows], dtype=np.float32)
    coarse = np.asarray([COARSE_INDEX[str(row["detector_coarse"])] for row in rows], dtype=np.int64)
    base_score = np.asarray([float(row["score"]) for row in predictions], dtype=np.float32)
    if set(np.unique(folds)) != {0, 1, 2}:
        raise ValueError("MAR requires the frozen three formal folds")

    resolved_score = np.empty(len(rows), dtype=np.float64)
    fit_records = []
    for held_out_fold in (0, 1, 2):
        train = folds != held_out_fold
        validation = folds == held_out_fold
        fit = fit_monotone_mar(
            train_base_score=base_score[train],
            train_features=features[train],
            train_target=target[train],
            train_role=roles[train].tolist(),
            train_coarse=coarse[train].tolist(),
            validation_base_score=base_score[validation],
            validation_features=features[validation],
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            rho_max=args.rho_max,
            seed=args.seed + held_out_fold,
        )
        resolved_score[validation] = fit.validation_scores
        fit_records.append(
            {
                "held_out_fold": held_out_fold,
                "n_train": int(train.sum()),
                "n_validation": int(validation.sum()),
                "feature_names": list(MAR_FEATURE_NAMES),
                "weights": fit.constrained_weights.tolist(),
                "rho": fit.rho,
                "bias": fit.bias,
                "final_loss": fit.final_loss,
            }
        )
    if not np.isfinite(resolved_score).all():
        raise ValueError("MAR generated non-finite OOF scores")

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    records = []
    for candidate_id, raw in enumerate(predictions):
        item = dict(raw)
        item["score"] = float(resolved_score[candidate_id])
        item["source_prediction_index"] = candidate_id
        records.append(item)
    baseline_records = []
    for candidate_id, raw in enumerate(predictions):
        item = dict(raw)
        item["source_prediction_index"] = candidate_id
        baseline_records.append(item)

    evaluations = {}
    for scope in (0, 1, 2, "all"):
        scoped_folds = {0, 1, 2} if scope == "all" else {scope}
        image_ids = {
            int(obj.image_id) for obj in formal.objects.values() if int(obj.fold) in scoped_folds
        }
        kwargs = {
            "gt_boxes": formal.boxes,
            "category_mapping": protocol.category_mapping,
            "iou_thresholds": protocol.iou_thresholds,
            "image_ids": image_ids,
            "fdr_levels": (0.15, 0.12, 0.11, 0.10),
            "nms_iou": 0.50,
        }
        baseline = official_fixed_risk_frontier(
            predictions=baseline_records,
            **kwargs,
        )
        mar = official_fixed_risk_frontier(predictions=records, **kwargs)
        baseline_metrics = _metrics(baseline, formal=formal, image_ids=image_ids, protocol=protocol)
        mar_metrics = _metrics(mar, formal=formal, image_ids=image_ids, protocol=protocol)
        evaluations[str(scope)] = {
            "baseline": baseline_metrics,
            "mar": mar_metrics,
            "delta_recall_at_fdr_0.12": (mar.points[0.12].recall - baseline.points[0.12].recall),
            "delta_recall_at_fdr_0.10": (mar.points[0.10].recall - baseline.points[0.10].recall),
            "delta_six_metric_min_at_fdr_0.12": (
                mar_metrics["ranking_at_fdr_0.12"]["six_metric_min"]
                - baseline_metrics["ranking_at_fdr_0.12"]["six_metric_min"]
            ),
        }

    combined = evaluations["all"]
    passes = (
        combined["delta_recall_at_fdr_0.12"] >= 0.002
        and combined["delta_recall_at_fdr_0.10"] >= -0.001
        and combined["delta_six_metric_min_at_fdr_0.12"] >= -0.005
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / "mar_crossfit_proxy_scores.npz"
    np.savez_compressed(
        score_path,
        candidate_id=np.arange(len(rows), dtype=np.int64),
        base_score=base_score,
        mar_score=resolved_score,
        fold=folds,
    )
    summary = {
        "status": "complete_exploratory_crossfit_proxy",
        "formal_admission": False,
        "stacking_protocol": "outer_meta_crossfit_without_inner_pav_oof",
        "fit_records": fit_records,
        "evaluations": evaluations,
        "passes_proxy_gate": passes,
        "next_action": "run_nested_pav_mar" if passes else "stop_learned_mar",
        "input_sha256": {
            "manifest": _sha256(args.manifest),
            "predictions": _sha256(args.predictions),
            "pav_logits": _sha256(args.pav_logits),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
        "scores_sha256": _sha256(score_path),
    }
    (args.output_dir / "mar_crossfit_proxy_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
