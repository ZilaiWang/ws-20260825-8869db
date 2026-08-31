#!/usr/bin/env python3
"""Attach OOF D-FINE same-fine support targets to an aligned OMQ cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.cross_detector_agreement import best_same_fine_support
from rsdet.evaluation.coco import load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--fold-root", type=Path, required=True)
    parser.add_argument("--specialist-name", default="dfine_predictions.json")
    parser.add_argument("--coarse", default="vehicle")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    category_ids = {
        category_id
        for category_id, coarse in protocol.category_mapping.items()
        if coarse == args.coarse
    }
    if not category_ids:
        raise ValueError(f"coarse class has no categories: {args.coarse}")
    iou_threshold = float(protocol.iou_thresholds[args.coarse])

    with np.load(args.cache, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    required = {
        "features",
        "detector_score",
        "fold",
        "image_id",
        "category_id",
        "bbox_xyxy",
    }
    missing = required - set(arrays)
    if missing:
        raise ValueError(f"cache is missing arrays: {sorted(missing)}")
    n = int(arrays["features"].shape[0])
    if arrays["features"].ndim != 2 or any(value.shape[0] != n for value in arrays.values()):
        raise ValueError("cache arrays are not row aligned")
    if not np.isfinite(arrays["features"]).all() or not np.isfinite(
        arrays["detector_score"]
    ).all():
        raise ValueError("cache contains NaN/Inf")

    support_score = np.zeros(n, dtype=np.float32)
    support_iou = np.zeros(n, dtype=np.float32)
    input_sha: dict[str, str] = {"cache": _sha256(args.cache)}
    coarse_mask = np.isin(arrays["category_id"].astype(np.int64), sorted(category_ids))
    for fold in (0, 1, 2):
        path = args.fold_root / f"fold_{fold}" / args.specialist_name
        specialist_by_image = load_coco_predictions(path)
        specialist = [
            dict(item, image_id=image_id)
            for image_id, rows in specialist_by_image.items()
            for item in rows
            if int(item["category_id"]) in category_ids
        ]
        index = np.flatnonzero(coarse_mask & (arrays["fold"].astype(np.int64) == fold))
        primary = [
            {
                "image_id": int(arrays["image_id"][row]),
                "category_id": int(arrays["category_id"][row]),
                "bbox_xyxy": [float(value) for value in arrays["bbox_xyxy"][row]],
                "score": float(arrays["detector_score"][row]),
            }
            for row in index
        ]
        evidence = best_same_fine_support(
            primary,
            specialist,
            iou_threshold=iou_threshold,
        )
        support_score[index] = np.asarray(
            [item["support_score"] for item in evidence], dtype=np.float32
        )
        support_iou[index] = np.asarray(
            [item["support_iou"] for item in evidence], dtype=np.float32
        )
        input_sha[f"specialist_fold{fold}"] = _sha256(path)

    agreement_product = arrays["detector_score"].astype(np.float32) * support_score
    if not np.isfinite(agreement_product).all():
        raise RuntimeError("teacher agreement contains NaN/Inf")
    arrays.update(
        {
            "teacher_support_score": support_score,
            "teacher_support_iou": support_iou,
            "teacher_agreement_product": agreement_product,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    summary = {
        "status": "complete_diagnostic_teacher_cache",
        "protocol": "oof_dfine_same_fine_support_on_aligned_y5_omq_v1",
        "warning": (
            "Teacher rows are OOF with respect to each row, but an outer-fold student using "
            "the other folds is still diagnostic until nested teacher inference is generated."
        ),
        "rows": n,
        "feature_dim": int(arrays["features"].shape[1]),
        "coarse": args.coarse,
        "coarse_rows": int(coarse_mask.sum()),
        "supported_rows": int((support_score[coarse_mask] > 0).sum()),
        "mean_support": float(support_score[coarse_mask].mean()),
        "mean_agreement_product": float(agreement_product[coarse_mask].mean()),
        "input_sha256": input_sha,
        "output_sha256": _sha256(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
