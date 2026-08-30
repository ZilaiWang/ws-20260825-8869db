#!/usr/bin/env python3
"""Build official-match-quality labels from a frozen prediction workpoint.

The script imports the repository's exact prediction-first evaluator.  It does
not use the V3 geometry-first support shortcut.  Same-fine, same-coarse and any
GT support are computed independently, which prevents a nearby wrong-class GT
from hiding a valid same-fine association.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.official_metric import compute_iou, evaluate_predictions_with_trace
from rsdet.postprocess.nms import class_aware_nms_predictions


def coarse_of(category_id: int) -> str:
    if 0 <= category_id <= 3:
        return "ship"
    if 4 <= category_id <= 23:
        return "aircraft"
    if category_id == 24:
        return "vehicle"
    raise ValueError(f"unknown category_id={category_id}")


def xywh_to_xyxy(values: Sequence[float]) -> list[float]:
    x, y, width, height = (float(value) for value in values)
    if width <= 0.0 or height <= 0.0:
        raise ValueError("bbox extent must be positive")
    return [x, y, x + width, y + height]


def best_supports(
    prediction: Mapping[str, Any],
    gt_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    predicted_category = int(prediction["category_id"])
    predicted_coarse = coarse_of(predicted_category)
    box = prediction["bbox_xyxy"]
    best_same_fine = (0.0, None)
    best_same_coarse = (0.0, None)
    best_any = (0.0, None)
    for index, gt in enumerate(gt_rows):
        gt_category = int(gt["category_id"])
        overlap = compute_iou(box, gt["bbox_xyxy"])
        if overlap > best_any[0]:
            best_any = (overlap, index)
        if coarse_of(gt_category) == predicted_coarse and overlap > best_same_coarse[0]:
            best_same_coarse = (overlap, index)
        if gt_category == predicted_category and overlap > best_same_fine[0]:
            best_same_fine = (overlap, index)
    return {
        "best_same_fine_iou": best_same_fine[0],
        "best_same_fine_gt_index": best_same_fine[1],
        "best_same_coarse_iou": best_same_coarse[0],
        "best_same_coarse_gt_index": best_same_coarse[1],
        "best_any_iou": best_any[0],
        "best_any_gt_index": best_any[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workpoint-threshold", type=float)
    parser.add_argument("--workpoint-threshold-ship", type=float)
    parser.add_argument("--workpoint-threshold-aircraft", type=float)
    parser.add_argument("--workpoint-threshold-vehicle", type=float)
    parser.add_argument("--background-iou", type=float, default=0.05)
    parser.add_argument("--nms-iou", type=float, default=0.50)
    args = parser.parse_args()
    coarse_values = {
        "ship": args.workpoint_threshold_ship,
        "aircraft": args.workpoint_threshold_aircraft,
        "vehicle": args.workpoint_threshold_vehicle,
    }
    if args.workpoint_threshold is not None and any(
        value is not None for value in coarse_values.values()
    ):
        raise ValueError("use either one global threshold or all three coarse thresholds")
    if args.workpoint_threshold is not None:
        workpoint_thresholds = {
            name: float(args.workpoint_threshold) for name in coarse_values
        }
    elif all(value is not None for value in coarse_values.values()):
        workpoint_thresholds = {
            name: float(value) for name, value in coarse_values.items()
        }
    else:
        raise ValueError("provide one global threshold or all three coarse thresholds")
    if any(not 0.0 <= value <= 1.0 for value in workpoint_thresholds.values()):
        raise ValueError("workpoint thresholds must be within [0, 1]")
    if not 0.0 <= args.background_iou < 0.35:
        raise ValueError("background-iou must be in [0, 0.35)")
    if not 0.0 < args.nms_iou <= 1.0:
        raise ValueError("nms-iou must be in (0, 1]")

    raw_gt = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    raw_predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    image_ids = [int(item["id"]) for item in raw_gt["images"]]
    gt: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    for annotation in raw_gt["annotations"]:
        gt[int(annotation["image_id"])].append(
            {
                "category_id": int(annotation["category_id"]),
                "bbox_xyxy": xywh_to_xyxy(annotation["bbox"]),
            }
        )

    normalized: list[dict[str, Any]] = []
    for candidate_id, raw in enumerate(raw_predictions):
        score = float(raw["score"])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid score at candidate {candidate_id}")
        normalized.append(
            {
                **dict(raw),
                "candidate_id": candidate_id,
                "source_prediction_index": candidate_id,
                "image_id": int(raw["image_id"]),
                "category_id": int(raw["category_id"]),
                "score": score,
                "bbox_xyxy": (
                    [float(value) for value in raw["bbox_xyxy"]]
                    if "bbox_xyxy" in raw
                    else xywh_to_xyxy(raw["bbox"])
                ),
            }
        )

    active_rows = [
        row
        for row in normalized
        if row["score"] >= workpoint_thresholds[coarse_of(int(row["category_id"]))]
    ]
    pred_ledger: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in image_ids}
    for row in active_rows:
        image_id = int(row["image_id"])
        pred_ledger[image_id].append(
            {
                "bbox_xyxy": row["bbox_xyxy"],
                "score": row["score"],
                "category_id": row["category_id"],
                "source_prediction_index": row["candidate_id"],
            }
        )

    pred_ledger = class_aware_nms_predictions(pred_ledger, args.nms_iou)
    active_candidate_ids = {
        int(row["source_prediction_index"])
        for records in pred_ledger.values()
        for row in records
    }
    category_mapping = {
        category_id: coarse_of(category_id) for category_id in range(25)
    }
    iou_thresholds = {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35}
    _metrics, trace = evaluate_predictions_with_trace(
        gt,
        pred_ledger,
        category_mapping=category_mapping,
        iou_thresholds=iou_thresholds,
    )
    protected = {
        int(match.prediction_index)
        for match in trace.matches
    }
    active_fp = {
        int(item.prediction_index)
        for item in trace.unmatched_predictions
    }

    output: list[dict[str, Any]] = []
    for row in normalized:
        candidate_id = int(row["candidate_id"])
        supports = best_supports(row, gt.get(int(row["image_id"]), ()))
        coarse = coarse_of(int(row["category_id"]))
        threshold = iou_thresholds[coarse]
        intrinsic_match = supports["best_same_fine_iou"] >= threshold
        is_active = candidate_id in active_candidate_ids
        if candidate_id in protected:
            role = "protected_tp"
        elif candidate_id in active_fp:
            if intrinsic_match:
                role = "active_duplicate_or_conflict"
            elif supports["best_same_coarse_iou"] >= threshold:
                role = "active_wrong_fine"
            elif supports["best_any_iou"] > args.background_iou:
                role = "active_localization_or_near_object"
            else:
                role = "active_background_fp"
        elif intrinsic_match:
            role = "inactive_matchable_tp"
        elif supports["best_same_coarse_iou"] >= threshold:
            role = "inactive_wrong_fine"
        elif supports["best_any_iou"] > args.background_iou:
            role = "inactive_near_object"
        else:
            role = "inactive_background"
        output.append(
            {
                **row,
                **supports,
                "coarse_id": {"ship": 0, "aircraft": 1, "vehicle": 2}[coarse],
                "official_iou_threshold": threshold,
                "intrinsic_match": int(intrinsic_match),
                "active_mask": int(is_active),
                "protected_tp": int(candidate_id in protected),
                "active_fp": int(candidate_id in active_fp),
                "quality_role": role,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in output) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidates": len(output),
                "active_before_nms": len(active_rows),
                "active_after_nms": len(active_candidate_ids),
                "protected_tp": len(protected),
                "active_fp": len(active_fp),
                "workpoint_thresholds": workpoint_thresholds,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
