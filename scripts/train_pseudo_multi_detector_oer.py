#!/usr/bin/env python3
"""Cross-fit Y5/M3 risk models directly in the fold-heldout pseudo-10K domain.

The pseudo benchmark contains two mosaics per formal CV3 fold.  For every
held-out fold this script fits only on the other two folds and writes scores
only for the held-out fold.  No full-data detector and no held-out GT is used
to produce a score.  The result is a deployment-domain diagnostic, not a claim
about the hidden official set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.multi_detector_oer import (
    MULTI_DETECTOR_AGREEMENT_COLUMNS,
    MULTI_DETECTOR_BASE_COLUMNS,
    MULTI_DETECTOR_GEOMETRY_COLUMNS,
    build_multi_detector_features,
    candidate_validity_labels,
    class_aware_nms_records,
)
from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_records(
    path: Path,
    *,
    model_key: str,
    score_floor: float,
    stable_offset: int,
) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    result: list[dict[str, Any]] = []
    for source_order, raw in enumerate(rows):
        score = float(raw["score"])
        if score < score_floor:
            continue
        x, y, width, height = (float(value) for value in raw["bbox"])
        if width <= 0.0 or height <= 0.0:
            continue
        fold = int(raw["source_fold"])
        if fold not in {0, 1, 2}:
            raise ValueError(f"invalid source_fold={fold}")
        result.append(
            {
                "image_id": int(raw["image_id"]),
                "fold": fold,
                "category_id": int(raw["category_id"]),
                "bbox_xyxy": [x, y, x + width, y + height],
                "score": score,
                "detector_score": score,
                "model_key": model_key,
                "stable_order": stable_offset + source_order,
            }
        )
    return result


def _to_coco(records: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for item in records:
        x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
        output.append(
            {
                "image_id": int(item["image_id"]),
                "category_id": int(item["category_id"]),
                "bbox": [x0, y0, x1 - x0, y1 - y0],
                "score": float(item["score"]),
                "source_fold": int(item["fold"]),
                "source_model": str(item["model_key"]),
            }
        )
    return output


def _model(seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=250,
        learning_rate=0.08,
        max_depth=6,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=seed,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--y5-predictions", type=Path, required=True)
    parser.add_argument("--m3-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--y5-floor", type=float, default=0.001)
    parser.add_argument("--m3-floor", type=float, default=0.03)
    parser.add_argument("--pre-nms-iou", type=float, default=0.70)
    parser.add_argument("--post-nms-iou", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    raw_gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    image_fold = {int(item["id"]): int(item["fold"]) for item in raw_gt["images"]}
    if set(image_fold.values()) != {0, 1, 2}:
        raise ValueError("ground truth must cover folds 0, 1 and 2")
    gt = load_coco_ground_truth(args.ground_truth)

    y5_raw = _load_records(
        args.y5_predictions,
        model_key="Y5",
        score_floor=args.y5_floor,
        stable_offset=0,
    )
    m3_raw = _load_records(
        args.m3_predictions,
        model_key="M3",
        score_floor=args.m3_floor,
        stable_offset=2_000_000,
    )
    y5 = class_aware_nms_records(y5_raw, iou_threshold=args.pre_nms_iou)
    m3 = class_aware_nms_records(m3_raw, iou_threshold=args.pre_nms_iou)
    records = sorted(y5 + m3, key=lambda item: int(item["stable_order"]))
    for item in records:
        expected = image_fold.get(int(item["image_id"]))
        if expected is None or expected != int(item["fold"]):
            raise ValueError("prediction fold does not match ground-truth image fold")

    features, all_columns = build_multi_detector_features(
        records, category_mapping=protocol.category_mapping
    )
    labels = candidate_validity_labels(
        records,
        gt_boxes=gt,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    folds = np.asarray([int(item["fold"]) for item in records], dtype=np.int64)
    variants = {
        "score_model": MULTI_DETECTOR_BASE_COLUMNS,
        "score_geometry": MULTI_DETECTOR_GEOMETRY_COLUMNS,
        "score_geometry_agreement": MULTI_DETECTOR_AGREEMENT_COLUMNS,
    }
    column_index = {name: index for index, name in enumerate(all_columns)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_union = class_aware_nms_records(records, iou_threshold=args.post_nms_iou)
    raw_union_path = args.output_dir / "candidate_union_predictions.json"
    raw_union_path.write_text(
        json.dumps(_to_coco(raw_union), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "status": "complete",
        "protocol": "pseudo10k_formal_cv3_fold_heldout_y5_m3_oer_v1",
        "warning": "deployment-domain proxy only; not an independent hidden-set estimate",
        "candidate_floors": {"Y5": args.y5_floor, "M3": args.m3_floor},
        "nms": {"pre": args.pre_nms_iou, "post": args.post_nms_iou},
        "counts": {
            "Y5_raw": len(y5_raw),
            "Y5_after_pre_nms": len(y5),
            "M3_raw": len(m3_raw),
            "M3_after_pre_nms": len(m3),
            "combined": len(records),
            "candidate_union_after_post_nms": len(raw_union),
            "valid_candidates": int(labels.sum()),
        },
        "inputs": {
            "ground_truth": _sha256(args.ground_truth),
            "Y5": _sha256(args.y5_predictions),
            "M3": _sha256(args.m3_predictions),
        },
        "candidate_union_predictions_sha256": _sha256(raw_union_path),
        "variants": {},
    }

    for variant_index, (name, columns) in enumerate(variants.items()):
        indices = [column_index[column] for column in columns]
        x = features[:, indices]
        probabilities = np.full(len(records), np.nan, dtype=np.float64)
        fold_rows: list[dict[str, Any]] = []
        for held_out in (0, 1, 2):
            train = folds != held_out
            validation = folds == held_out
            model = _model(20260829 + variant_index * 10 + held_out)
            model.fit(x[train], labels[train])
            probabilities[validation] = model.predict_proba(x[validation])[:, 1]
            path = args.output_dir / f"{name}_heldout_fold{held_out}.joblib"
            joblib.dump(
                {"model": model, "columns": list(columns), "all_columns": list(all_columns)},
                path,
                compress=3,
            )
            fold_rows.append(
                {
                    "held_out_fold": held_out,
                    "n_train": int(train.sum()),
                    "n_validation": int(validation.sum()),
                    "train_positive_rate": float(labels[train].mean()),
                    "validation_positive_rate": float(labels[validation].mean()),
                    "model_sha256": _sha256(path),
                }
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"incomplete OOF probabilities for {name}")
        scored: list[dict[str, Any]] = []
        for item, probability in zip(records, probabilities, strict=True):
            output = dict(item)
            output["score"] = float(probability)
            scored.append(output)
        scored = class_aware_nms_records(scored, iou_threshold=args.post_nms_iou)
        path = args.output_dir / f"{name}_predictions.json"
        path.write_text(
            json.dumps(_to_coco(scored), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["variants"][name] = {
            "columns": list(columns),
            "fold_models": fold_rows,
            "predictions": len(scored),
            "predictions_sha256": _sha256(path),
        }

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
