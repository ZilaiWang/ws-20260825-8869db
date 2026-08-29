#!/usr/bin/env python3
"""Fit and audit the deployable 12-feature OER on formal CV3 OOF proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.evaluation.official_frontier import official_fixed_risk_frontier
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.manifest import PAV_METADATA_COLUMNS
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _model(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=6,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    table = pd.read_csv(args.manifest)
    required = set(PAV_METADATA_COLUMNS) | {
        "fold",
        "group_id",
        "image_id",
        "detector_category_id",
        "proposal_x0",
        "proposal_y0",
        "proposal_x1",
        "proposal_y1",
        "target_official_tp",
    }
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"PAV manifest missing columns: {sorted(missing)}")
    if set(table["fold"].astype(int)) != {0, 1, 2}:
        raise ValueError("manifest must contain folds 0, 1 and 2")
    if table[list(PAV_METADATA_COLUMNS)].isna().any().any():
        raise ValueError("OER features contain missing values")

    x = table[list(PAV_METADATA_COLUMNS)].to_numpy(dtype=np.float64)
    y = table["target_official_tp"].to_numpy(dtype=np.int64)
    folds = table["fold"].to_numpy(dtype=np.int64)
    probabilities = np.full(len(table), np.nan, dtype=np.float64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fold_audit: list[dict[str, object]] = []
    for held_out in (0, 1, 2):
        train = folds != held_out
        validation = folds == held_out
        train_groups = set(table.loc[train, "group_id"].astype(str))
        validation_groups = set(table.loc[validation, "group_id"].astype(str))
        overlap = train_groups & validation_groups
        if overlap:
            raise ValueError(f"group leakage for fold {held_out}: {len(overlap)}")
        model = _model(2026 + held_out)
        model.fit(x[train], y[train])
        probabilities[validation] = model.predict_proba(x[validation])[:, 1]
        model_path = args.output_dir / f"oer_heldout_fold{held_out}.joblib"
        joblib.dump(model, model_path, compress=3)
        fold_audit.append(
            {
                "held_out_fold": held_out,
                "n_train": int(train.sum()),
                "n_validation": int(validation.sum()),
                "train_positive_rate": float(y[train].mean()),
                "validation_positive_rate": float(y[validation].mean()),
                "model_sha256": _sha256(model_path),
            }
        )
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF OER coverage is incomplete")

    predictions = []
    for index, row in table.iterrows():
        predictions.append(
            {
                "image_id": int(row["image_id"]),
                "category_id": int(row["detector_category_id"]),
                "bbox_xyxy": [
                    float(row["proposal_x0"]),
                    float(row["proposal_y0"]),
                    float(row["proposal_x1"]),
                    float(row["proposal_y1"]),
                ],
                "score": float(probabilities[index]),
                "source_prediction_index": int(index),
            }
        )
    formal = load_formal_ground_truth(args.formal_crop_manifest)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    frontier = official_fixed_risk_frontier(
        gt_boxes=formal.boxes,
        predictions=predictions,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        fdr_levels=(0.20, 0.17, 0.15, 0.12, 0.10),
        nms_iou=0.50,
    )
    scored_path = args.output_dir / "deployable_oer_oof_predictions.json"
    scored_path.write_text(json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {
        "status": "deployable_oer_models_ready",
        "protocol": "formal_cv3_outer_oof_metadata_only_oer_v1",
        "features": list(PAV_METADATA_COLUMNS),
        "n_rows": len(table),
        "positive_count": int(y.sum()),
        "folds": fold_audit,
        "frontier": {
            str(level): {
                "recall": frontier.points[level].recall,
                "fdr": frontier.points[level].fdr,
                "tp": frontier.points[level].tp,
                "fp": frontier.points[level].fp,
                "threshold": frontier.points[level].score_threshold,
            }
            for level in (0.20, 0.17, 0.15, 0.12, 0.10)
        },
        "input_sha256": {
            "manifest": _sha256(args.manifest),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
        },
        "predictions_sha256": _sha256(scored_path),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
