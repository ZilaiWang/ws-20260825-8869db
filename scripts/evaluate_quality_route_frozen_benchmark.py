#!/usr/bin/env python3
"""Apply a Normal-CV3-frozen quality route to a labelled pressure benchmark.

The route decision, fold-specific thresholds, and fold-specific TorchScript
heads are immutable inputs.  Benchmark labels are used only after inference to
measure the frozen route; they never select a threshold or a coarse bypass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.data.xh_dataset import coarse_name
from rsdet.evaluation.coco import load_coco_ground_truth
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    COARSE_ORDER,
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.postprocess.nms import class_aware_nms_predictions
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _local_density(
    predictions: list[dict[str, Any]], coarse_mapping: dict[int, str], radius: float
) -> np.ndarray:
    output = np.zeros(len(predictions), dtype=np.float32)
    by_image: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(predictions):
        by_image[int(row["image_id"])].append(index)
    radius_sq = float(radius) ** 2
    for indices in by_image.values():
        centers: dict[int, tuple[float, float]] = {}
        cells: dict[tuple[str, int, int], list[int]] = defaultdict(list)
        for index in indices:
            x, y, width, height = (float(value) for value in predictions[index]["bbox"])
            center = (x + width / 2.0, y + height / 2.0)
            centers[index] = center
            coarse = coarse_mapping[int(predictions[index]["category_id"])]
            cells[(coarse, math.floor(center[0] / radius), math.floor(center[1] / radius))].append(index)
        for index in indices:
            coarse = coarse_mapping[int(predictions[index]["category_id"])]
            cx, cy = centers[index]
            cell_x, cell_y = math.floor(cx / radius), math.floor(cy / radius)
            neighbours = (
                other_index
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                for other_index in cells.get((coarse, cell_x + dx, cell_y + dy), ())
            )
            output[index] = sum(
                1
                for other_index in neighbours
                if (centers[other_index][0] - cx) ** 2
                + (centers[other_index][1] - cy) ** 2
                <= radius_sq
            )
    return output


def build_base_crop_features(
    predictions: list[dict[str, Any]],
    *,
    score_field: str,
    coarse_mapping: dict[int, str],
    density_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the registered 63-D ``base_crop`` feature contract."""

    density = _local_density(predictions, coarse_mapping, density_radius)
    rows: list[np.ndarray] = []
    detector_scores: list[float] = []
    coarse_lookup = {"ship": 0, "aircraft": 1, "vehicle": 2}
    for index, item in enumerate(predictions):
        missing = {
            "crop_top1",
            "crop_margin",
            "crop_entropy",
            "crop_top1_class",
            "detector_crop_agree",
        } - set(item)
        if missing:
            raise ValueError(f"prediction[{index}] lacks crop evidence: {sorted(missing)}")
        category = int(item["category_id"])
        crop_category = int(item["crop_top1_class"])
        if not 0 <= category < 25 or not 0 <= crop_category < 25:
            raise ValueError("fine categories must be in [0, 24]")
        x, y, width, height = (float(value) for value in item["bbox"])
        del x, y
        if width <= 0.0 or height <= 0.0:
            raise ValueError("bbox width/height must be positive")
        detector_score = float(item[score_field])
        detector_scores.append(detector_score)
        numerical = np.asarray(
            [
                detector_score,
                float(item["crop_top1"]),
                float(item["crop_margin"]),
                float(item["crop_entropy"]) / math.log(25.0),
                float(item["detector_crop_agree"]),
                math.log1p(width),
                math.log1p(height),
                math.log1p(width * height),
                math.log(max(width / height, height / width)),
                math.log1p(float(density[index])),
            ],
            dtype=np.float32,
        )
        coarse = coarse_lookup[coarse_mapping[category]]
        row = np.concatenate(
            (
                numerical,
                np.eye(3, dtype=np.float32)[coarse],
                np.eye(25, dtype=np.float32)[category],
                np.eye(25, dtype=np.float32)[crop_category],
            )
        )
        rows.append(row)
    features = np.stack(rows).astype(np.float32)
    scores = np.asarray(detector_scores, dtype=np.float32)
    if features.shape != (len(predictions), 63) or not np.isfinite(features).all():
        raise RuntimeError("base_crop feature construction failed")
    if not np.isfinite(scores).all() or np.any((scores < 0.0) | (scores > 1.0)):
        raise RuntimeError("baseline scores must be finite probabilities")
    return features, scores


def _nms_threshold(
    predictions: list[dict[str, Any]],
    *,
    thresholds: dict[tuple[int, str], float],
    fold_by_image: dict[int, int],
    nms_iou: float,
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {
        image_id: [] for image_id in fold_by_image
    }
    for item in predictions:
        row = dict(item)
        x, y, width, height = (float(value) for value in row["bbox"])
        row["bbox_xyxy"] = [x, y, x + width, y + height]
        grouped[int(row["image_id"])].append(row)
    kept = class_aware_nms_predictions(grouped, nms_iou)
    return [
        row
        for image_id in sorted(kept)
        for row in kept[image_id]
        if float(row["score"])
        >= thresholds[(fold_by_image[image_id], coarse_name(int(row["category_id"])))]
    ]


def _nms_all(
    predictions: list[dict[str, Any]], *, image_ids: set[int], nms_iou: float
) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    for item in predictions:
        row = dict(item)
        x, y, width, height = (float(value) for value in row["bbox"])
        row["bbox_xyxy"] = [x, y, x + width, y + height]
        grouped[int(row["image_id"])].append(row)
    kept = class_aware_nms_predictions(grouped, nms_iou)
    return [row for image_id in sorted(kept) for row in kept[image_id]]


def _metrics(
    gt_path: Path,
    predictions: list[dict[str, Any]],
    *,
    protocol: Any,
    latency: float,
) -> dict[str, Any]:
    gt = load_coco_ground_truth(gt_path)
    grouped: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in gt}
    for row in predictions:
        grouped[int(row["image_id"])].append(row)
    ranking = evaluate_ranking_metrics(
        gt,
        grouped,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return platform_metrics_payload(
        build_platform_observed_metrics(
            ranking,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
            latency_seconds=latency,
            latency_max_seconds=protocol.latency_max_seconds,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--route-decision", type=Path, required=True)
    parser.add_argument("--model-pattern", required=True, help="must contain {fold}")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    parser.add_argument("--score-field", default="score")
    parser.add_argument("--quantile-calibrators", type=Path)
    parser.add_argument("--density-radius", type=float, default=1024.0)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    parser.add_argument("--latency-baseline", type=float, default=2.473167)
    parser.add_argument("--latency-candidate", type=float, default=2.623167)
    parser.add_argument("--maximum-coarse-recall-drop", type=float, default=0.005)
    args = parser.parse_args()
    if "{fold}" not in args.model_pattern:
        raise ValueError("--model-pattern must contain {fold}")

    import torch

    gt_payload = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(row["id"]): int(row["fold"]) for row in gt_payload["images"]}
    if set(fold_by_image.values()) != {0, 1, 2}:
        raise ValueError("benchmark must provide folds 0/1/2")
    predictions = json.loads(args.pred.read_text(encoding="utf-8"))
    route = json.loads(args.route_decision.read_text(encoding="utf-8"))
    if route.get("metric_protocol") != "platform_observed_20260831":
        raise ValueError("route decision uses the wrong metric protocol")
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    features, detector_scores = build_base_crop_features(
        predictions,
        score_field=args.score_field,
        coarse_mapping=protocol.category_mapping,
        density_radius=args.density_radius,
    )

    models = {
        fold: torch.jit.load(args.model_pattern.format(fold=fold), map_location="cpu").eval()
        for fold in (0, 1, 2)
    }
    calibrators = (
        None
        if args.quantile_calibrators is None
        else json.loads(args.quantile_calibrators.read_text(encoding="utf-8"))
    )
    if calibrators is not None and calibrators.get("uses_labels") is not False:
        raise ValueError("quantile calibrator must be explicitly label-free")
    candidate = [dict(item) for item in predictions]
    route_audit: dict[str, Any] = {}
    with torch.inference_mode():
        for fold in (0, 1, 2):
            indices = np.asarray(
                [i for i, row in enumerate(predictions) if fold_by_image[int(row["image_id"])] == fold],
                dtype=np.int64,
            )
            output = models[fold](
                torch.from_numpy(features[indices]), torch.from_numpy(detector_scores[indices])
            )[:, 0].numpy()
            if calibrators is not None:
                calibrated = output.copy()
                category = np.asarray(
                    [int(predictions[int(index)]["category_id"]) for index in indices]
                )
                coarse_id = np.where(category <= 3, 0, np.where(category <= 23, 1, 2))
                for value in (0, 1, 2):
                    mask = coarse_id == value
                    row = calibrators["calibrators"][str(fold)][str(value)]
                    from calibrate_quality_score_quantiles import quantile_map

                    calibrated[mask] = quantile_map(
                        output[mask],
                        np.asarray(row["quality_knots"]),
                        np.asarray(row["detector_knots"]),
                    )
                output = calibrated
            for source_index, score in zip(indices, output, strict=True):
                candidate[int(source_index)]["quality_score"] = float(score)
            route_audit[str(fold)] = {}

    baseline_thresholds: dict[tuple[int, str], float] = {}
    candidate_thresholds: dict[tuple[int, str], float] = {}
    baseline_threshold = float(route["baseline_threshold"])
    for fold in (0, 1, 2):
        for coarse in COARSE_ORDER:
            selected = route["folds"][str(fold)]["selected_routes"][coarse]
            use_quality = selected["route"] == "quality"
            threshold = float(selected["threshold"] if use_quality else baseline_threshold)
            baseline_thresholds[(fold, coarse)] = baseline_threshold
            candidate_thresholds[(fold, coarse)] = threshold
            route_audit[str(fold)][coarse] = {
                "route": selected["route"],
                "threshold": threshold,
            }

    for row in candidate:
        fold = fold_by_image[int(row["image_id"])]
        coarse = coarse_name(int(row["category_id"]))
        if route_audit[str(fold)][coarse]["route"] == "quality":
            row["score"] = float(row["quality_score"])
        else:
            row["score"] = float(row[args.score_field])
    baseline_rows = [dict(row, score=float(row[args.score_field])) for row in predictions]
    baseline_ranked = _nms_all(
        baseline_rows, image_ids=set(fold_by_image), nms_iou=args.nms_iou
    )
    candidate_ranked = _nms_all(
        candidate, image_ids=set(fold_by_image), nms_iou=args.nms_iou
    )
    baseline_kept = _nms_threshold(
        baseline_rows,
        thresholds=baseline_thresholds,
        fold_by_image=fold_by_image,
        nms_iou=args.nms_iou,
    )
    candidate_kept = _nms_threshold(
        candidate,
        thresholds=candidate_thresholds,
        fold_by_image=fold_by_image,
        nms_iou=args.nms_iou,
    )
    baseline_metrics = _metrics(
        args.gt, baseline_kept, protocol=protocol, latency=args.latency_baseline
    )
    candidate_metrics = _metrics(
        args.gt, candidate_kept, protocol=protocol, latency=args.latency_candidate
    )
    coarse_drops = {
        coarse: float(baseline_metrics["per_coarse"][coarse]["macro_recall"])
        - float(candidate_metrics["per_coarse"][coarse]["macro_recall"])
        for coarse in COARSE_ORDER
    }
    delta = {
        "gate_recall": candidate_metrics["gate_recall"] - baseline_metrics["gate_recall"],
        "gate_fdr": candidate_metrics["gate_fdr"] - baseline_metrics["gate_fdr"],
        "absolute_score": candidate_metrics["absolute_score"]
        - baseline_metrics["absolute_score"],
        "per_coarse_recall_drop": coarse_drops,
        "max_coarse_recall_drop": max(coarse_drops.values()),
    }
    admitted = (
        delta["absolute_score"] > 0.0
        and delta["gate_fdr"] <= 0.0
        and delta["max_coarse_recall_drop"] <= args.maximum_coarse_recall_drop
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": "frozen_quality_route_pressure_evaluation_v1",
        "metric_protocol": protocol.metric_protocol,
        "benchmark_labels_used_for_selection": False,
        "route_decision_sha256": _sha256(args.route_decision),
        "quantile_calibrators_sha256": (
            None
            if args.quantile_calibrators is None
            else _sha256(args.quantile_calibrators)
        ),
        "input_sha256": {"gt": _sha256(args.gt), "pred": _sha256(args.pred)},
        "feature_contract": "base_crop_63d_v1",
        "route": route_audit,
        "counts": {
            "raw": len(predictions),
            "baseline_after_nms": len(baseline_ranked),
            "candidate_after_nms": len(candidate_ranked),
            "baseline_after_nms_threshold": len(baseline_kept),
            "candidate_after_nms_threshold": len(candidate_kept),
        },
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "delta_vs_baseline": delta,
        "pressure_admission": admitted,
    }
    (args.output_dir / "analysis.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.output_dir / "baseline_predictions.json").write_text(
        json.dumps(baseline_kept) + "\n", encoding="utf-8"
    )
    (args.output_dir / "candidate_predictions.json").write_text(
        json.dumps(candidate_kept) + "\n", encoding="utf-8"
    )
    (args.output_dir / "baseline_ranked_nms.json").write_text(
        json.dumps(baseline_ranked) + "\n", encoding="utf-8"
    )
    (args.output_dir / "candidate_ranked_nms.json").write_text(
        json.dumps(candidate_ranked) + "\n", encoding="utf-8"
    )
    (args.output_dir / "baseline_ranked_raw.json").write_text(
        json.dumps(baseline_rows) + "\n", encoding="utf-8"
    )
    (args.output_dir / "candidate_ranked_raw.json").write_text(
        json.dumps(candidate) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
