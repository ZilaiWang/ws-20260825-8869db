#!/usr/bin/env python3
"""Export NMS-kept OMQ OOF scores and a fold-aware formal COCO ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import load_formal_ground_truth
from rsdet.postprocess.nms import class_aware_nms_predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyxy_to_xywh(box: list[float]) -> list[float]:
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path)
    parser.add_argument("--score-source", choices=("detector", "quality"), required=True)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.score_source == "quality" and args.score_dir is None:
        raise ValueError("quality score export requires --score-dir")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("nms-iou must be within (0, 1]")

    with np.load(args.cache, allow_pickle=False) as cache:
        detector_score = cache["detector_score"].astype(np.float32)
        folds = cache["fold"].astype(np.int64)
        image_ids = cache["image_id"].astype(np.int64)
        category_ids = cache["category_id"].astype(np.int64)
        boxes = cache["bbox_xyxy"].astype(np.float32)
    n = len(detector_score)
    scores = detector_score.copy()
    score_inputs: list[str] = []
    if args.score_source == "quality":
        scores[:] = np.nan
        coverage = np.zeros(n, dtype=np.uint8)
        for fold in range(3):
            path = args.score_dir / f"quality_fold{fold}.scores.npz"
            with np.load(path, allow_pickle=False) as payload:
                index = payload["candidate_index"].astype(np.int64)
                value = payload["score"].astype(np.float32)
            if len(index) != len(value) or np.any(index < 0) or np.any(index >= n):
                raise ValueError(f"invalid score rows in {path}")
            if np.any(coverage[index]):
                raise ValueError("quality score files overlap")
            if np.any(folds[index] != fold):
                raise ValueError(f"quality score file {path} contains the wrong outer fold")
            scores[index] = value
            coverage[index] = 1
            score_inputs.append(str(path))
        if not coverage.all() or not np.isfinite(scores).all():
            raise ValueError("quality OOF coverage is incomplete")

    raw_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    if len(raw_predictions) != n:
        raise ValueError("cache/prediction count mismatch")
    by_image: dict[int, list[dict]] = {}
    for index in range(n):
        raw = raw_predictions[index]
        if (
            int(raw["image_id"]) != int(image_ids[index])
            or int(raw["category_id"]) != int(category_ids[index])
        ):
            raise ValueError(f"cache/prediction identity mismatch at row {index}")
        row = {
            "image_id": int(image_ids[index]),
            "category_id": int(category_ids[index]),
            "bbox_xyxy": [float(value) for value in boxes[index]],
            "score": float(scores[index]),
            "source_prediction_index": index,
        }
        by_image.setdefault(int(image_ids[index]), []).append(row)
    kept = class_aware_nms_predictions(by_image, args.nms_iou)
    exported_predictions = [
        {
            "image_id": image_id,
            "category_id": int(row["category_id"]),
            "bbox": _xyxy_to_xywh([float(value) for value in row["bbox_xyxy"]]),
            "score": float(row["score"]),
            "source_prediction_index": int(row["source_prediction_index"]),
        }
        for image_id in sorted(kept)
        for row in kept[image_id]
    ]

    formal = load_formal_ground_truth(args.formal_crop_manifest)
    fold_by_image: dict[int, int] = {}
    for obj in formal.objects.values():
        previous = fold_by_image.setdefault(int(obj.image_id), int(obj.fold))
        if previous != int(obj.fold):
            raise ValueError(f"inconsistent formal fold for image {obj.image_id}")
    annotations = []
    annotation_id = 1
    for image_id in sorted(formal.boxes):
        for row in formal.boxes[image_id]:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": int(image_id),
                    "category_id": int(row["category_id"]),
                    "bbox": _xyxy_to_xywh([float(value) for value in row["bbox_xyxy"]]),
                }
            )
            annotation_id += 1
    ground_truth = {
        "images": [
            {"id": image_id, "fold": fold_by_image[image_id]}
            for image_id in sorted(formal.boxes)
        ],
        "annotations": annotations,
        "categories": [{"id": index, "name": str(index)} for index in range(25)],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred_path = args.output_dir / f"{args.score_source}_oof_predictions.json"
    gt_path = args.output_dir / "formal_cv3_ground_truth.json"
    pred_path.write_text(
        json.dumps(exported_predictions, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    gt_path.write_text(json.dumps(ground_truth, ensure_ascii=False) + "\n", encoding="utf-8")
    audit = {
        "status": "complete",
        "score_source": args.score_source,
        "input_candidates": n,
        "nms_kept": len(exported_predictions),
        "images": len(formal.boxes),
        "annotations": len(annotations),
        "fold_candidate_counts": {
            str(fold): int((folds == fold).sum()) for fold in range(3)
        },
        "score_inputs": score_inputs,
        "sha256": {
            "cache": _sha256(args.cache),
            "predictions": _sha256(args.predictions),
            "formal_crop_manifest": _sha256(args.formal_crop_manifest),
            "output_predictions": _sha256(pred_path),
            "output_ground_truth": _sha256(gt_path),
        },
    }
    audit_path = args.output_dir / f"{args.score_source}_export_audit.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
