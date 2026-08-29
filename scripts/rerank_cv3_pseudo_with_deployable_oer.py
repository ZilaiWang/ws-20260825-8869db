#!/usr/bin/env python3
"""Apply fold-heldout deployable OER models to pseudo-10K P03 evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rerank_cv3_pseudo_with_crop import aircraft_same_class_nms

from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.hera_guard.manifest import PAV_METADATA_COLUMNS, metadata_from_node
from rsdet.utils.config import load_config


def _local_density(
    predictions: list[dict[str, Any]], coarse_mapping: dict[int, str], radius: float
) -> list[int]:
    """Approximate an 800px source image with a local 1024px neighborhood."""

    result = [0] * len(predictions)
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, item in enumerate(predictions):
        by_image[int(item["image_id"])].append(index)
    radius_sq = radius * radius
    for indices in by_image.values():
        centers = []
        for index in indices:
            x, y, w, h = (float(value) for value in predictions[index]["bbox"])
            centers.append((x + w / 2.0, y + h / 2.0))
        for offset, index in enumerate(indices):
            coarse = coarse_mapping[int(predictions[index]["category_id"])]
            cx, cy = centers[offset]
            result[index] = sum(
                1
                for other_offset, other_index in enumerate(indices)
                if coarse_mapping[int(predictions[other_index]["category_id"])] == coarse
                and (centers[other_offset][0] - cx) ** 2
                + (centers[other_offset][1] - cy) ** 2
                <= radius_sq
            )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--model-pattern", required=True, help="must contain {fold}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--density-radius", type=float, default=1024.0)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    args = parser.parse_args()
    if "{fold}" not in args.model_pattern:
        raise ValueError("--model-pattern must contain {fold}")
    if args.density_radius <= 0:
        raise ValueError("--density-radius must be positive")

    gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(item["id"]): int(item["fold"]) for item in gt["images"]}
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    density = _local_density(predictions, protocol.category_mapping, args.density_radius)
    models = {
        fold: joblib.load(Path(args.model_pattern.format(fold=fold))) for fold in (0, 1, 2)
    }

    rows: list[list[float]] = []
    for index, item in enumerate(predictions):
        x, y, width, height = (float(value) for value in item["bbox"])
        short_edge = min(width, height)
        aspect = max(width / height, height / width)
        coarse = protocol.category_mapping[int(item["category_id"])]
        node = {
            "y5_score": float(item["detector_score"]),
            "crop_top1": float(item["crop_top1"]),
            "crop_margin": float(item["crop_margin"]),
            "crop_entropy": float(item["crop_entropy"]),
            "detector_crop_agree": float(item["detector_crop_agree"]),
            "short_edge": short_edge,
            "area": width * height,
            "aspect": aspect,
            "local_density": density[index],
        }
        metadata = metadata_from_node(node, coarse_name=coarse)
        rows.append([metadata[name] for name in PAV_METADATA_COLUMNS])
    matrix = np.asarray(rows, dtype=np.float64)
    if matrix.shape != (len(predictions), len(PAV_METADATA_COLUMNS)) or not np.isfinite(
        matrix
    ).all():
        raise RuntimeError("invalid pseudo OER feature matrix")

    scored: list[dict[str, Any]] = []
    for fold in (0, 1, 2):
        indices = [
            index
            for index, item in enumerate(predictions)
            if fold_by_image[int(item["image_id"])] == fold
        ]
        probabilities = models[fold].predict_proba(matrix[indices])[:, 1]
        for index, probability in zip(indices, probabilities, strict=True):
            item = dict(predictions[index])
            item["p03_fused_score"] = float(item["score"])
            item["score"] = float(probability)
            item["oer_feature_contract"] = "pav_metadata_12_local1024_v1"
            scored.append(item)
    scored.sort(key=lambda item: int(item["proposal_index"]))
    if len(scored) != len(predictions):
        raise RuntimeError("OER output coverage mismatch")
    nms = aircraft_same_class_nms(scored, args.nms_iou)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "predictions_oer.json"
    nms_path = args.output_dir / "predictions_oer_aircraft_nms.json"
    raw_path.write_text(json.dumps(scored, ensure_ascii=False) + "\n", encoding="utf-8")
    nms_path.write_text(json.dumps(nms, ensure_ascii=False) + "\n", encoding="utf-8")
    payload = {
        "status": "complete",
        "protocol": "formal_fold_heldout_deployable_oer_on_pseudo10k_v1",
        "features": list(PAV_METADATA_COLUMNS),
        "density_radius": args.density_radius,
        "counts": {"input": len(predictions), "after_aircraft_nms": len(nms)},
        "outputs": {"raw": str(raw_path), "aircraft_nms": str(nms_path)},
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
