#!/usr/bin/env python3
"""Cross-fit a pixel-evidence risk head over pseudo-10K detector proposals.

The input is produced by ``rerank_cv3_pseudo_with_crop.py`` and therefore each
candidate carries predictions from the P03 classifier that was held out on the
same formal fold.  Two modes are evaluated:

``identity`` keeps the detector fine class. ``dual_hypothesis`` additionally
creates the P03 top-1 fine class when it belongs to the same official coarse
class.  Both hypotheses remain observable; a fold-heldout risk model ranks
them.  Vehicle is naturally unchanged because it has only one fine class.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.multi_detector_oer import (
    candidate_validity_labels,
    class_aware_nms_records,
)
from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

PIXEL_COLUMNS = (
    "detector_score",
    "model_y5",
    "model_m3",
    "coarse_ship",
    "coarse_aircraft",
    "coarse_vehicle",
    "log_short_edge",
    "log_area",
    "log_aspect",
    "foreground_probability",
    "crop_hypothesis_probability",
    "crop_conditional_class_probability",
    "crop_top1",
    "crop_margin",
    "crop_entropy_normalized",
    "detector_crop_agree",
    "hypothesis_is_relabel",
    "support_y5_rot_max_iou",
    "support_y5_800_max_iou",
    "support_m3_id_max_iou",
    "support_coph_max_iou",
    "source_support_count",
    "source_support_score_sum",
    "heterogeneous_support",
    "coarse_foreground_probability",
    "dino_foreground_probability",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalize(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for stable_order, raw in enumerate(rows):
        x, y, width, height = (float(value) for value in raw["bbox"])
        if width <= 0.0 or height <= 0.0:
            continue
        fold = int(raw["source_fold"])
        if fold not in {0, 1, 2}:
            raise ValueError(f"invalid source_fold={fold}")
        model = str(raw.get("source_model", "UNKNOWN")).upper()
        if model not in {"Y5", "M3"}:
            raise ValueError(f"invalid source_model={model}")
        result.append(
            {
                **dict(raw),
                "fold": fold,
                "model_key": model,
                "bbox_xyxy": [x, y, x + width, y + height],
                "stable_order": stable_order,
            }
        )
    return result


def expand_hypotheses(
    records: list[Mapping[str, Any]],
    *,
    category_mapping: Mapping[int, str],
    dual_hypothesis: bool,
) -> list[dict[str, Any]]:
    """Create the original and optional same-coarse P03 top-1 hypotheses."""

    output: list[dict[str, Any]] = []
    for item in records:
        original = dict(item)
        original["hypothesis_is_relabel"] = 0
        original["crop_hypothesis_probability"] = float(
            item["crop_class_probability"]
        )
        output.append(original)
        current = int(item["category_id"])
        top1 = int(item["crop_top1_class"])
        if (
            dual_hypothesis
            and top1 != current
            and category_mapping[top1] == category_mapping[current]
        ):
            alternative = dict(item)
            alternative["category_id"] = top1
            alternative["hypothesis_is_relabel"] = 1
            alternative["crop_hypothesis_probability"] = float(
                item.get("crop_top1_absolute", item["crop_top1"])
            )
            alternative["stable_order"] = int(item["stable_order"]) + 10_000_000
            output.append(alternative)
    return sorted(output, key=lambda item: int(item["stable_order"]))


def build_pixel_features(
    records: list[Mapping[str, Any]], *, category_mapping: Mapping[int, str]
) -> np.ndarray:
    matrix = np.zeros((len(records), len(PIXEL_COLUMNS)), dtype=np.float64)
    for index, item in enumerate(records):
        category = int(item["category_id"])
        coarse = category_mapping[category]
        x0, y0, x1, y1 = (float(value) for value in item["bbox_xyxy"])
        width, height = x1 - x0, y1 - y0
        short_edge = min(width, height)
        aspect = max(width / height, height / width)
        model = str(item["model_key"])
        matrix[index] = (
            float(item.get("detector_score", item["score"])),
            float(model == "Y5"),
            float(model == "M3"),
            float(coarse == "ship"),
            float(coarse == "aircraft"),
            float(coarse == "vehicle"),
            math.log1p(short_edge),
            math.log1p(width * height),
            math.log(aspect),
            float(item.get("foreground_probability", 1.0)),
            float(item["crop_hypothesis_probability"]),
            float(
                item.get(
                    "crop_conditional_class_probability",
                    item["crop_hypothesis_probability"],
                )
            ),
            float(item["crop_top1"]),
            float(item["crop_margin"]),
            float(item["crop_entropy"]) / math.log(25.0),
            float(item["detector_crop_agree"]),
            float(item["hypothesis_is_relabel"]),
            float(item.get("support_y5_rot_max_iou", 0.0)),
            float(item.get("support_y5_800_max_iou", 0.0)),
            float(item.get("support_m3_id_max_iou", 0.0)),
            float(item.get("support_coph_max_iou", 0.0)),
            float(item.get("source_support_count", 0.0)),
            float(item.get("source_support_score_sum", 0.0)),
            float(item.get("heterogeneous_support", 0.0)),
            float(item.get("coarse_foreground_probability", 0.0)),
            float(item.get("dino_foreground_probability", 0.0)),
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("pixel feature matrix contains NaN/Inf")
    return matrix


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
                "hypothesis_is_relabel": int(item["hypothesis_is_relabel"]),
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
    parser.add_argument("--crop-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--nms-iou", type=float, default=0.70)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.ground_truth)
    raw = json.loads(args.crop_predictions.read_text(encoding="utf-8"))
    normalized = _normalize(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "status": "complete",
        "protocol": "pseudo10k_fold_heldout_p03_pixel_oer_v1",
        "warning": "deployment-domain proxy only; not an independent hidden-set estimate",
        "columns": list(PIXEL_COLUMNS),
        "input_sha256": {
            "ground_truth": _sha256(args.ground_truth),
            "crop_predictions": _sha256(args.crop_predictions),
        },
        "variants": {},
    }
    for variant_index, dual in enumerate((False, True)):
        name = "identity" if not dual else "dual_hypothesis"
        records = expand_hypotheses(
            normalized,
            category_mapping=protocol.category_mapping,
            dual_hypothesis=dual,
        )
        records = class_aware_nms_records(records, iou_threshold=args.nms_iou)
        features = build_pixel_features(records, category_mapping=protocol.category_mapping)
        labels = candidate_validity_labels(
            records,
            gt_boxes=gt,
            category_mapping=protocol.category_mapping,
            iou_thresholds=protocol.iou_thresholds,
        )
        folds = np.asarray([int(item["fold"]) for item in records], dtype=np.int64)
        probabilities = np.full(len(records), np.nan, dtype=np.float64)
        fold_rows = []
        for held_out in (0, 1, 2):
            train = folds != held_out
            validation = folds == held_out
            model = _model(20260829 + variant_index * 10 + held_out)
            model.fit(features[train], labels[train])
            probabilities[validation] = model.predict_proba(features[validation])[:, 1]
            path = args.output_dir / f"{name}_heldout_fold{held_out}.joblib"
            joblib.dump({"model": model, "columns": list(PIXEL_COLUMNS)}, path, compress=3)
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
        scored = []
        for item, probability in zip(records, probabilities, strict=True):
            row = dict(item)
            row["score"] = float(probability)
            scored.append(row)
        scored = class_aware_nms_records(scored, iou_threshold=args.nms_iou)
        path = args.output_dir / f"{name}_predictions.json"
        path.write_text(
            json.dumps(_to_coco(scored), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        summary["variants"][name] = {
            "dual_hypothesis": dual,
            "input_candidates": len(records),
            "valid_candidates": int(labels.sum()),
            "output_candidates": len(scored),
            "fold_models": fold_rows,
            "predictions_sha256": _sha256(path),
        }
    path = args.output_dir / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
