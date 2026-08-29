#!/usr/bin/env python3
"""Cross-fit a deployable Y5/M3 proposal risk resolver on formal CV3 OOF.

This is an integration experiment, not a new detector training run.  M3 is
first truncated at a documented low score and both models receive same-fine
NMS.  Three scalar-only risk models are compared: score/model identity,
geometry/density, and cross-model agreement.  Every probability is produced by
a model that did not train on that proposal's fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable

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
    prediction_ledger,
)
from rsdet.analysis.oof_detection import build_threshold_curve, load_formal_ground_truth
from rsdet.evaluation.official_metric import evaluate_predictions, evaluate_ranking_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.calibration import build_threshold_grid
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_image_metadata(path: Path) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = int(row["image_id"])
            if image_id in result:
                raise ValueError(f"duplicate image_id={image_id} in {path}")
            result[image_id] = row
    return result


def _load_candidates(
    aggregate: Path,
    *,
    model_key: str,
    score_floor: float,
    stable_offset: int,
) -> list[dict[str, Any]]:
    images = _load_image_metadata(aggregate / "oof_images.csv")
    result: list[dict[str, Any]] = []
    with (aggregate / "oof_proposals.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for source_order, row in enumerate(csv.DictReader(handle)):
            score = float(row["score"])
            if score < score_floor:
                continue
            image_id = int(row["image_id"])
            if image_id not in images:
                raise ValueError(f"proposal references unknown image_id={image_id}")
            x0, y0 = float(row["x"]), float(row["y"])
            width, height = float(row["width"]), float(row["height"])
            result.append(
                {
                    "proposal_uid": row["proposal_uid"],
                    "image_id": image_id,
                    "fold": int(row["fold"]),
                    "group_id": images[image_id]["group_id"],
                    "category_id": int(row["category_id"]),
                    "bbox_xyxy": [x0, y0, x0 + width, y0 + height],
                    "score": score,
                    "detector_score": score,
                    "model_key": model_key,
                    "stable_order": stable_offset + source_order,
                }
            )
    return result


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


def _scope(
    mapping: dict[int, list[dict[str, Any]]], image_ids: set[int]
) -> dict[int, list[dict[str, Any]]]:
    return {image_id: list(mapping.get(image_id, ())) for image_id in sorted(image_ids)}


def _metrics_payload(metrics: Any, ranking: Any) -> dict[str, Any]:
    return {
        "recall": metrics.recall,
        "fdr": metrics.fdr,
        "tp": int(metrics.details["tp"]),
        "fp": int(metrics.details["fp"]),
        "fn": int(metrics.details["fn"]),
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
        "ranking_per_coarse": {
            name: {
                "macro_recall": item.macro_recall,
                "macro_fdr": item.macro_fdr,
            }
            for name, item in ranking.per_coarse.items()
        },
    }


def _best_under_fdr(curve: Iterable[dict[str, Any]], level: float) -> dict[str, Any]:
    feasible = [row for row in curve if float(row["overall_fdr"]) <= level]
    if not feasible:
        return min(curve, key=lambda row: (float(row["overall_fdr"]), -float(row["threshold"])))
    return max(
        feasible,
        key=lambda row: (
            float(row["overall_recall"]),
            -float(row["overall_fdr"]),
            float(row["threshold"]),
        ),
    )


def _crossfit_frontier(
    *,
    formal: Any,
    predictions: list[dict[str, Any]],
    protocol: Any,
    image_fold: dict[int, int],
    levels: tuple[float, ...],
) -> dict[str, Any]:
    gt = formal.boxes
    # The official trace exposes an index within each image's input list.
    # ``build_threshold_curve`` intentionally keys its prefix events the same
    # way, so a global candidate ID must not be passed here.
    pred_by_image = prediction_ledger(predictions, formal.image_ids)
    fold_images = {
        fold: {image_id for image_id, value in image_fold.items() if value == fold}
        for fold in (0, 1, 2)
    }
    thresholds = build_threshold_grid(0.001, 0.996, 0.005)
    train_curves: dict[int, list[dict[str, Any]]] = {}
    for held_out in (0, 1, 2):
        train_ids = set().union(
            *(ids for fold, ids in fold_images.items() if fold != held_out)
        )
        train_curves[held_out], _ = build_threshold_curve(
            _scope(gt, train_ids),
            _scope(pred_by_image, train_ids),
            thresholds=thresholds,
            protocol=protocol,
        )
    result: dict[str, Any] = {}
    for level in levels:
        chosen = {
            held_out: float(_best_under_fdr(train_curves[held_out], level)["threshold"])
            for held_out in (0, 1, 2)
        }
        selected = {
            image_id: [
                item
                for item in pred_by_image.get(image_id, ())
                if float(item["score"]) >= chosen[image_fold[image_id]]
            ]
            for image_id in formal.image_ids
        }
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
        result[f"{level:.3f}"] = {
            "crossfit_thresholds": {str(key): value for key, value in chosen.items()},
            "crossfit": _metrics_payload(metrics, ranking),
        }
    floor_metrics = evaluate_predictions(
        gt,
        pred_by_image,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    floor_ranking = evaluate_ranking_metrics(
        gt,
        pred_by_image,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return {"candidate_floor": _metrics_payload(floor_metrics, floor_ranking), "levels": result}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--y5-aggregate", type=Path, required=True)
    parser.add_argument("--m3-aggregate", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--y5-floor", type=float, default=0.001)
    parser.add_argument("--m3-floor", type=float, default=0.03)
    parser.add_argument("--pre-nms-iou", type=float, default=0.50)
    parser.add_argument("--post-nms-iou", type=float, default=0.50)
    args = parser.parse_args()
    for floor in (args.y5_floor, args.m3_floor):
        if not 0.0 <= floor <= 1.0:
            raise ValueError("candidate floors must be in [0, 1]")

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    image_fold: dict[int, int] = {}
    image_group: dict[int, str] = {}
    for item in formal.objects.values():
        previous = image_fold.setdefault(item.image_id, item.fold)
        previous_group = image_group.setdefault(item.image_id, item.group_id)
        if previous != item.fold or previous_group != item.group_id:
            raise ValueError("formal image has inconsistent fold/group metadata")
    if set(image_fold) != set(formal.image_ids):
        raise ValueError("formal fold coverage mismatch")

    y5_raw = _load_candidates(
        args.y5_aggregate,
        model_key="Y5",
        score_floor=args.y5_floor,
        stable_offset=0,
    )
    m3_raw = _load_candidates(
        args.m3_aggregate,
        model_key="M3",
        score_floor=args.m3_floor,
        stable_offset=2_000_000,
    )
    y5 = class_aware_nms_records(y5_raw, iou_threshold=args.pre_nms_iou)
    m3 = class_aware_nms_records(m3_raw, iou_threshold=args.pre_nms_iou)
    records = y5 + m3
    records.sort(key=lambda item: int(item["stable_order"]))
    features, all_columns = build_multi_detector_features(
        records, category_mapping=protocol.category_mapping
    )
    labels = candidate_validity_labels(
        records,
        gt_boxes=formal.boxes,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    folds = np.asarray([image_fold[int(item["image_id"])] for item in records], dtype=np.int64)
    if set(folds.tolist()) != {0, 1, 2}:
        raise ValueError("candidate fold coverage mismatch")

    variants = {
        "score_model": MULTI_DETECTOR_BASE_COLUMNS,
        "score_geometry": MULTI_DETECTOR_GEOMETRY_COLUMNS,
        "score_geometry_agreement": MULTI_DETECTOR_AGREEMENT_COLUMNS,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "complete",
        "protocol": "formal_cv3_fold_heldout_y5_m3_scalar_oer_v1",
        "candidate_floors": {"Y5": args.y5_floor, "M3": args.m3_floor},
        "counts": {
            "Y5_raw": len(y5_raw),
            "Y5_after_pre_nms": len(y5),
            "M3_raw": len(m3_raw),
            "M3_after_pre_nms": len(m3),
            "combined": len(records),
            "valid_candidates": int(labels.sum()),
        },
        "features": {},
        "variants": {},
        "input_sha256": {
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
            "Y5_oof_proposals": _sha256(args.y5_aggregate / "oof_proposals.csv"),
            "M3_oof_proposals": _sha256(args.m3_aggregate / "oof_proposals.csv"),
        },
    }
    column_index = {name: index for index, name in enumerate(all_columns)}
    for variant_index, (name, columns) in enumerate(variants.items()):
        indices = [column_index[column] for column in columns]
        x = features[:, indices]
        probabilities = np.full(len(records), np.nan, dtype=np.float64)
        fold_models = []
        for held_out in (0, 1, 2):
            train = folds != held_out
            validation = folds == held_out
            train_groups = {
                image_group[int(records[index]["image_id"])]
                for index in np.flatnonzero(train)
            }
            validation_groups = {
                image_group[int(records[index]["image_id"])]
                for index in np.flatnonzero(validation)
            }
            if train_groups & validation_groups:
                raise ValueError(f"group leakage in held-out fold {held_out}")
            model = _model(20260829 + variant_index * 10 + held_out)
            model.fit(x[train], labels[train])
            probabilities[validation] = model.predict_proba(x[validation])[:, 1]
            model_path = args.output_dir / f"{name}_heldout_fold{held_out}.joblib"
            joblib.dump(
                {"model": model, "columns": list(columns), "all_columns": list(all_columns)},
                model_path,
                compress=3,
            )
            fold_models.append(
                {
                    "held_out_fold": held_out,
                    "n_train": int(train.sum()),
                    "n_validation": int(validation.sum()),
                    "train_positive_rate": float(labels[train].mean()),
                    "validation_positive_rate": float(labels[validation].mean()),
                    "model_sha256": _sha256(model_path),
                }
            )
        if not np.isfinite(probabilities).all():
            raise RuntimeError(f"incomplete OOF scores for {name}")
        scored = []
        for index, (item, probability) in enumerate(zip(records, probabilities, strict=True)):
            output = dict(item)
            output["score"] = float(probability)
            output["source_prediction_index"] = index
            scored.append(output)
        post_nms = class_aware_nms_records(scored, iou_threshold=args.post_nms_iou)
        for index, item in enumerate(post_nms):
            item["source_prediction_index"] = index
        prediction_path = args.output_dir / f"{name}_oof_predictions.json"
        prediction_path.write_text(
            json.dumps(post_nms, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        frontier = _crossfit_frontier(
            formal=formal,
            predictions=post_nms,
            protocol=protocol,
            image_fold=image_fold,
            levels=(0.10, 0.12, 0.15, 0.17, 0.20),
        )
        summary["features"][name] = list(columns)
        summary["variants"][name] = {
            "fold_models": fold_models,
            "after_post_nms": len(post_nms),
            "predictions_sha256": _sha256(prediction_path),
            "frontier": frontier,
        }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
