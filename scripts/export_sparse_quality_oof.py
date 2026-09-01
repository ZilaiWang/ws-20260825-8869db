#!/usr/bin/env python3
"""Join sparse coarse-route quality scores with identity scores and export OOF."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.postprocess.nms import class_aware_nms_predictions


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _export(
    *,
    image_ids: np.ndarray,
    category_ids: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_iou: float,
) -> list[dict[str, object]]:
    by_image: dict[int, list[dict[str, object]]] = {}
    for index in range(len(scores)):
        image_id = int(image_ids[index])
        by_image.setdefault(image_id, []).append(
            {
                "image_id": image_id,
                "category_id": int(category_ids[index]),
                "bbox_xyxy": [float(value) for value in boxes[index]],
                "score": float(scores[index]),
                "source_prediction_index": index,
            }
        )
    kept = class_aware_nms_predictions(by_image, nms_iou)
    return [
        {
            "image_id": image_id,
            "category_id": int(row["category_id"]),
            "bbox": [
                float(row["bbox_xyxy"][0]),
                float(row["bbox_xyxy"][1]),
                float(row["bbox_xyxy"][2]) - float(row["bbox_xyxy"][0]),
                float(row["bbox_xyxy"][3]) - float(row["bbox_xyxy"][1]),
            ],
            "score": float(row["score"]),
            "source_prediction_index": int(row["source_prediction_index"]),
        }
        for image_id in sorted(kept)
        for row in kept[image_id]
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--score", type=Path, action="append", required=True)
    parser.add_argument("--route", choices=("ship", "aircraft", "vehicle"), required=True)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.score) != 3:
        raise ValueError("exactly three fold-held-out score files are required")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("nms-iou must be within (0, 1]")
    with np.load(args.cache, allow_pickle=False) as cache:
        detector_score = cache["detector_score"].astype(np.float32)
        folds = cache["fold"].astype(np.int64)
        coarse_ids = cache["coarse_id"].astype(np.int64)
        image_ids = cache["image_id"].astype(np.int64)
        category_ids = cache["category_id"].astype(np.int64)
        boxes = cache["bbox_xyxy"].astype(np.float32)
    n = len(detector_score)
    if any(len(array) != n for array in (folds, coarse_ids, image_ids, category_ids, boxes)):
        raise ValueError("cache arrays are not aligned")
    if not np.isfinite(detector_score).all() or not np.isfinite(boxes).all():
        raise ValueError("cache contains NaN/Inf")

    route_id = {"ship": 0, "aircraft": 1, "vehicle": 2}[args.route]
    route_mask = coarse_ids == route_id
    candidate_score = detector_score.copy()
    coverage = np.zeros(n, dtype=np.uint8)
    score_inputs: list[dict[str, object]] = []
    for path in args.score:
        with np.load(path, allow_pickle=False) as payload:
            index = payload["candidate_index"].astype(np.int64)
            score = payload["score"].astype(np.float32)
        if len(index) != len(score) or np.any(index < 0) or np.any(index >= n):
            raise ValueError(f"invalid sparse score rows in {path}")
        if np.any(coverage[index]):
            raise ValueError("sparse score files overlap")
        if len(set(int(value) for value in folds[index])) != 1:
            raise ValueError(f"score file mixes held-out folds: {path}")
        fold = int(folds[index][0]) if len(index) else -1
        if fold not in (0, 1, 2):
            raise ValueError(f"invalid or empty held-out score file: {path}")
        if np.any(coarse_ids[index] != route_id):
            raise ValueError(f"score file escapes the declared {args.route} route")
        if not np.isfinite(score).all() or np.any((score < 0.0) | (score > 1.0)):
            raise ValueError(f"score file contains invalid probabilities: {path}")
        candidate_score[index] = score
        coverage[index] = 1
        score_inputs.append({"fold": fold, "path": str(path), "sha256": _sha256(path)})
    if not np.all(coverage[route_mask] == 1) or np.any(coverage[~route_mask]):
        raise ValueError("sparse scores must cover the entire declared route and nothing else")

    baseline = _export(
        image_ids=image_ids,
        category_ids=category_ids,
        boxes=boxes,
        scores=detector_score,
        nms_iou=args.nms_iou,
    )
    candidate = _export(
        image_ids=image_ids,
        category_ids=category_ids,
        boxes=boxes,
        scores=candidate_score,
        nms_iou=args.nms_iou,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = args.output_dir / "baseline_identity_oof_predictions.json"
    candidate_path = args.output_dir / f"{args.route}_quality_oof_predictions.json"
    baseline_path.write_text(json.dumps(baseline) + "\n", encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate) + "\n", encoding="utf-8")
    audit = {
        "version": "sparse_quality_oof_export_v1",
        "status": "complete",
        "route": args.route,
        "non_route_policy": "identity_score_and_geometry",
        "cache_rows": n,
        "route_rows": int(route_mask.sum()),
        "covered_route_rows": int(coverage.sum()),
        "baseline_nms_rows": len(baseline),
        "candidate_nms_rows": len(candidate),
        "nms_iou": args.nms_iou,
        "score_inputs": sorted(score_inputs, key=lambda row: int(row["fold"])),
        "sha256": {
            "cache": _sha256(args.cache),
            "baseline": _sha256(baseline_path),
            "candidate": _sha256(candidate_path),
        },
    }
    (args.output_dir / "export_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
