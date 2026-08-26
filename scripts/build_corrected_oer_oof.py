#!/usr/bin/env python3
"""Rebuild OER OOF scores from official prediction-first proposal labels.

This is the corrected CPU bridge from historical Y5 candidates to HERA-Guard.
It uses only the frozen CV3 outer folds, never random row folds, and emits a
complete prediction file whose ``score`` is a strictly OOF OER probability.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.grouped_oof import GroupedLedger, iter_outer_splits, split_audit
from rsdet.evaluation.official_frontier import official_fixed_risk_frontier
from rsdet.evaluation.official_metric import compute_iou
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

BASE_FEATURES = (
    "y5_score",
    "crop_top1",
    "crop_margin",
    "crop_entropy",
    "crop_top1_class",
    "detector_crop_agree",
    "w",
    "h",
    "area",
    "short_edge",
    "aspect",
    "local_density",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_assignments(formal) -> dict[int, tuple[int, str]]:
    result: dict[int, tuple[int, str]] = {}
    for obj in formal.objects.values():
        assignment = (int(obj.fold), str(obj.group_id))
        previous = result.setdefault(int(obj.image_id), assignment)
        if previous != assignment:
            raise ValueError(f"inconsistent formal assignment: image {obj.image_id}")
    return result


def _oto_support(predictions: list[dict], oto_dir: Path) -> np.ndarray:
    by_image: dict[int, list[dict]] = defaultdict(list)
    for fold in range(3):
        path = oto_dir / f"a5_oto_fold{fold}.json"
        for raw in json.loads(path.read_text(encoding="utf-8")):
            by_image[int(raw["image_id"])].append(
                {
                    "category_id": int(raw["category_id"]),
                    "score": float(raw["score"]),
                    "bbox_xyxy": [float(value) for value in raw["bbox_xyxy"]],
                }
            )
    support = np.zeros(len(predictions), dtype=np.float32)
    for index, pred in enumerate(predictions):
        for other in by_image.get(int(pred["image_id"]), []):
            if other["category_id"] != int(pred["category_id"]) or other["score"] < 0.5:
                continue
            if compute_iou(pred["bbox_xyxy"], other["bbox_xyxy"]) > 0.5:
                support[index] = 1.0
                break
    return support


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--oto-dir", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from sklearn.ensemble import HistGradientBoostingClassifier

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    assignment = _image_assignments(formal)
    nodes = pd.read_csv(args.nodes).sort_values("idx").reset_index(drop=True)
    if nodes["idx"].astype(int).tolist() != list(range(len(nodes))):
        raise ValueError("node idx must be contiguous and prediction-aligned")
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if len(predictions) != len(nodes):
        raise ValueError("prediction/node count mismatch")
    for raw in predictions:
        raw["image_id"] = int(raw["image_id"])
        raw["category_id"] = int(raw["category_id"])
        raw["score"] = float(raw["score"])
        raw["bbox_xyxy"] = [float(value) for value in raw["bbox_xyxy"]]

    nodes["has_oto"] = _oto_support(predictions, args.oto_dir)
    nodes["d4_support_norm"] = np.asarray(
        [float(raw.get("d4_support", 0.0)) / 8.0 for raw in predictions], dtype=np.float32
    )
    nodes["outer_fold"] = nodes["image_id"].map(lambda value: assignment[int(value)][0])
    nodes["group_id"] = nodes["image_id"].map(lambda value: assignment[int(value)][1])
    ledger = GroupedLedger(
        candidate_ids=nodes["idx"].astype(str).to_numpy(),
        image_ids=nodes["image_id"].to_numpy(),
        outer_folds=nodes["outer_fold"].to_numpy(),
        group_ids=nodes["group_id"].astype(str).to_numpy(),
    )
    outer = iter_outer_splits(ledger)
    features = list(BASE_FEATURES) + ["d4_support_norm", "has_oto"]
    x = nodes[features].to_numpy(dtype=np.float64)
    y = nodes["is_valid"].to_numpy(dtype=np.int64)
    probabilities = np.full(len(nodes), np.nan, dtype=np.float64)
    fold_rows = []
    for split in outer:
        train, validation = split.train_indices, split.validation_indices
        model = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.08,
            max_depth=6,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=2026 + split.held_out_fold,
        )
        model.fit(x[train], y[train])
        probabilities[validation] = model.predict_proba(x[validation])[:, 1]
        fold_rows.append(
            {
                "held_out_fold": split.held_out_fold,
                "n_train": len(train),
                "n_validation": len(validation),
                "positive_rate": float(y[validation].mean()),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF probability coverage is incomplete")

    scored_predictions = []
    score_rows = []
    for candidate_id, (raw, probability) in enumerate(
        zip(predictions, probabilities, strict=True)
    ):
        record = dict(raw)
        record["score"] = float(probability)
        record["source_prediction_index"] = candidate_id
        scored_predictions.append(record)
        score_rows.append(
            {
                "candidate_id": candidate_id,
                "image_id": record["image_id"],
                "fold": int(nodes.iloc[candidate_id]["outer_fold"]),
                "group_id": str(nodes.iloc[candidate_id]["group_id"]),
                "oer_oof_score": float(probability),
                "official_is_valid": int(y[candidate_id]),
            }
        )
    frontier = official_fixed_risk_frontier(
        gt_boxes=formal.boxes,
        predictions=scored_predictions,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        fdr_levels=(0.15, 0.12, 0.11, 0.10),
        nms_iou=0.50,
    )

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "corrected_oer_oof_predictions.json"
    predictions_path.write_text(
        json.dumps(scored_predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    scores_path = output_dir / "corrected_oer_oof_scores.csv"
    with scores_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
        writer.writeheader()
        writer.writerows(score_rows)
    summary = {
        "status": "corrected_oer_oof_ready",
        "protocol": "official_prediction_first_plus_formal_cv3_outer_oof_v1",
        "features": features,
        "n_predictions": len(predictions),
        "positive_count": int(y.sum()),
        "folds": fold_rows,
        "split_audit": split_audit(ledger, outer=outer),
        "frontier": {
            str(level): {
                "recall": frontier.points[level].recall,
                "fdr": frontier.points[level].fdr,
                "tp": frontier.points[level].tp,
                "fp": frontier.points[level].fp,
                "threshold": frontier.points[level].score_threshold,
            }
            for level in (0.15, 0.12, 0.11, 0.10)
        },
        "input_sha256": {
            "nodes": _sha256(args.nodes),
            "predictions": _sha256(args.predictions),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
        "output_sha256": {
            "predictions": _sha256(predictions_path),
            "scores": _sha256(scores_path),
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
