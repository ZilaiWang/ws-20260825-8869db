#!/usr/bin/env python3
"""Bounded replay for P40 high-confidence Vehicle fallback into hierarchy.

The incumbent owns all non-Vehicle detections plus the hierarchy Vehicle
branch.  P40 may only append fine-24 boxes above a frozen threshold which do
not overlap incumbent/earlier-rescue Vehicle boxes.  The grid is deliberately
small and the selected point must improve both fixed Hard and Sentinel-B.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.contracts import Prediction
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.experiments.fixed_proxy import quality_contribution
from rsdet.submission.class_resolution_router import (
    PrimaryLabelRescue,
    ResolutionLabelRoute,
    compose_routed_predictions,
)
from rsdet.utils.config import load_config

HARD_GT_SHA256 = "e02e139e5b7f440ee1107fa49c1673d9dc5dc40ed2b3e92de74af6460b885195"
SENTINEL_GT_SHA256 = "eb1f8850624d77252b568b3515b1953d57adbd6234eed1dec6b467ff5f48c211"
VEHICLE_LABEL = 24


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _grid(text: str, *, lower_open: bool = False) -> tuple[float, ...]:
    values = tuple(float(token.strip()) for token in text.split(",") if token.strip())
    if not values or len(values) != len(set(values)):
        raise ValueError("grid must contain unique values")
    if any(not (0.0 < value <= 1.0) if lower_open else not (0.0 <= value <= 1.0)
           for value in values):
        raise ValueError("grid values outside valid probability/IoU range")
    return values


def _prediction(image_id: int, rows: list[dict[str, Any]]) -> Prediction:
    return Prediction(
        image_id=image_id,
        boxes_xyxy=[[float(value) for value in row["bbox_xyxy"]] for row in rows],
        scores=[float(row["score"]) for row in rows],
        labels=[int(row["category_id"]) for row in rows],
    )


def _coco_rows(predictions: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = [
        {
            "image_id": image_id,
            "category_id": int(row["category_id"]),
            "bbox": [
                float(row["bbox_xyxy"][0]),
                float(row["bbox_xyxy"][1]),
                float(row["bbox_xyxy"][2] - row["bbox_xyxy"][0]),
                float(row["bbox_xyxy"][3] - row["bbox_xyxy"][1]),
            ],
            "score": float(row["score"]),
        }
        for image_id in sorted(predictions)
        for row in predictions[image_id]
    ]
    rows.sort(
        key=lambda row: (
            int(row["image_id"]),
            -float(row["score"]),
            int(row["category_id"]),
            tuple(float(value) for value in row["bbox"]),
        )
    )
    return rows


def _mapping(prediction: Prediction) -> list[dict[str, Any]]:
    return [
        {
            "bbox_xyxy": [float(value) for value in box],
            "score": float(score),
            "category_id": int(label),
        }
        for box, score, label in zip(
            prediction.boxes_xyxy,
            prediction.scores,
            prediction.labels,
            strict=True,
        )
    ]


def _platform(gt: Any, predictions: Any, protocol: Any) -> dict[str, Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    return platform_metrics_payload(build_platform_observed_metrics(ranking))


def _fingerprint(rows: dict[int, list[dict[str, Any]]], *, exclude_vehicle: bool) -> list[Any]:
    return sorted(
        (
            int(image_id),
            int(row["category_id"]),
            float(row["score"]),
            tuple(float(value) for value in row["bbox_xyxy"]),
        )
        for image_id, records in rows.items()
        for row in records
        if not exclude_vehicle or int(row["category_id"]) != VEHICLE_LABEL
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for split in ("hard", "sentinel"):
        parser.add_argument(f"--{split}-gt", type=Path, required=True)
        parser.add_argument(f"--{split}-primary", type=Path, required=True)
        parser.add_argument(f"--{split}-incumbent", type=Path, required=True)
    parser.add_argument("--rescue-thresholds", default="0.60,0.70,0.80")
    parser.add_argument("--dedup-ious", default="0.35,0.50,0.70")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    if args.output_root.exists():
        raise FileExistsError(args.output_root)
    thresholds = _grid(args.rescue_thresholds)
    dedup_ious = _grid(args.dedup_ious, lower_open=True)
    expected_gt = {"hard": HARD_GT_SHA256, "sentinel": SENTINEL_GT_SHA256}
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    route = ResolutionLabelRoute(
        primary_labels=frozenset(range(24)),
        expert_labels=frozenset({VEHICLE_LABEL}),
        primary_threshold=0.0,
        expert_threshold=0.0,
    )

    inputs: dict[str, dict[str, Any]] = {}
    for split in ("hard", "sentinel"):
        gt_path = getattr(args, f"{split}_gt")
        primary_path = getattr(args, f"{split}_primary")
        incumbent_path = getattr(args, f"{split}_incumbent")
        if _sha256(gt_path) != expected_gt[split]:
            raise ValueError(f"{split} GT SHA differs from the frozen proxy")
        gt = load_coco_ground_truth(gt_path)
        primary_raw = load_coco_predictions(primary_path)
        incumbent_raw = load_coco_predictions(incumbent_path)
        if set(primary_raw) - set(gt) or set(incumbent_raw) - set(gt):
            raise ValueError(f"{split} predictions contain image ids outside GT")
        primary = {image_id: list(primary_raw.get(image_id, [])) for image_id in gt}
        incumbent = {image_id: list(incumbent_raw.get(image_id, [])) for image_id in gt}
        if _fingerprint(primary, exclude_vehicle=True) != _fingerprint(
            incumbent, exclude_vehicle=True
        ):
            raise ValueError(f"{split} incumbent changed non-Vehicle P40 predictions")
        inputs[split] = {
            "gt": gt,
            "primary": primary,
            "incumbent": incumbent,
            "baseline_platform": _platform(gt, incumbent, protocol),
            "sha256": {
                "gt": _sha256(gt_path),
                "primary": _sha256(primary_path),
                "incumbent": _sha256(incumbent_path),
            },
        }

    candidates: list[dict[str, Any]] = []
    prediction_cache: dict[tuple[float, float, str], dict[int, list[dict[str, Any]]]] = {}
    for threshold in thresholds:
        for dedup_iou in dedup_ious:
            result: dict[str, Any] = {
                "rescue_threshold": threshold,
                "dedup_iou": dedup_iou,
                "splits": {},
            }
            deltas: list[float] = []
            for split in ("hard", "sentinel"):
                source = inputs[split]
                composed: dict[int, list[dict[str, Any]]] = {}
                for image_id in source["gt"]:
                    primary = _prediction(image_id, source["primary"][image_id])
                    expert = _prediction(
                        image_id,
                        [
                            row
                            for row in source["incumbent"][image_id]
                            if int(row["category_id"]) == VEHICLE_LABEL
                        ],
                    )
                    merged = compose_routed_predictions(
                        primary,
                        expert,
                        route=route,
                        primary_rescue=PrimaryLabelRescue(
                            labels=frozenset({VEHICLE_LABEL}),
                            threshold=threshold,
                            dedup_iou=dedup_iou,
                        ),
                    )
                    composed[image_id] = _mapping(merged)
                if _fingerprint(composed, exclude_vehicle=True) != _fingerprint(
                    source["incumbent"], exclude_vehicle=True
                ):
                    raise AssertionError("rescue changed Ship/Aircraft predictions")
                platform = _platform(source["gt"], composed, protocol)
                delta = quality_contribution(platform) - quality_contribution(
                    source["baseline_platform"]
                )
                added = sum(map(len, composed.values())) - sum(
                    map(len, source["incumbent"].values())
                )
                result["splits"][split] = {
                    "quality_delta": delta,
                    "added_vehicle": added,
                    "platform": platform,
                }
                deltas.append(delta)
                prediction_cache[(threshold, dedup_iou, split)] = composed
            result["worst_quality_delta"] = min(deltas)
            result["mean_quality_delta"] = sum(deltas) / len(deltas)
            result["both_positive"] = all(delta > 0.0 for delta in deltas)
            candidates.append(result)

    eligible = [candidate for candidate in candidates if candidate["both_positive"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["worst_quality_delta"],
            row["mean_quality_delta"],
            -row["rescue_threshold"],
            -row["dedup_iou"],
        ),
        default=None,
    )
    args.output_root.mkdir(parents=True)
    if selected is not None:
        threshold = float(selected["rescue_threshold"])
        dedup_iou = float(selected["dedup_iou"])
        selected_dir = args.output_root / "selected"
        selected_dir.mkdir()
        for split in ("hard", "sentinel"):
            output = selected_dir / f"{split}_predictions.json"
            output.write_text(
                json.dumps(
                    _coco_rows(prediction_cache[(threshold, dedup_iou, split)]),
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
        next_action = "exact_runtime_validation"
    else:
        next_action = "keep_incumbent"
    summary = {
        "status": "complete",
        "experiment": "p40_hierarchy_vehicle_primary_rescue_bounded_v1",
        "metric_protocol": "platform_observed_20260831",
        "role": "fixed_proxy_bounded_selection_not_hidden_score_prediction",
        "frozen_grid": {
            "rescue_thresholds": list(thresholds),
            "dedup_ious": list(dedup_ious),
        },
        "selection_rule": "maximize min(Hard,Sentinel) quality delta; both strictly positive",
        "selected": selected,
        "next_action": next_action,
        "ship_aircraft_bitwise_unchanged": True,
        "input_sha256": {split: inputs[split]["sha256"] for split in inputs},
        "candidates": candidates,
    }
    summary_path = args.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.output_root / "status.txt").write_text("complete\n", encoding="utf-8")
    checksums = []
    for path in sorted(args.output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{_sha256(path)}  {path.relative_to(args.output_root)}")
    (args.output_root / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "selected": selected,
                "next_action": next_action,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
