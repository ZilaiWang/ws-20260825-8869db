#!/usr/bin/env python3
"""Decompose a pseudo-10K cross-fit workpoint with the official matcher."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import (
    FormalGroundTruth,
    GroundTruthObject,
    decompose_official_errors,
)
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def apply_fold_thresholds(
    predictions: Mapping[int, Sequence[Mapping[str, Any]]],
    image_folds: Mapping[int, int],
    thresholds: Mapping[int, float],
) -> dict[int, list[dict[str, Any]]]:
    if set(thresholds) != {0, 1, 2}:
        raise ValueError("thresholds must define folds 0, 1 and 2")
    return {
        image_id: [
            dict(item)
            for item in predictions.get(image_id, ())
            if float(item["score"]) >= float(thresholds[image_folds[image_id]])
        ]
        for image_id in sorted(image_folds)
    }


def _formal_ground_truth(
    boxes: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    image_folds: Mapping[int, int],
    category_mapping: Mapping[int, str],
) -> FormalGroundTruth:
    objects: dict[tuple[int, int], GroundTruthObject] = {}
    count = 0
    for image_id in sorted(boxes):
        for index, item in enumerate(boxes[image_id]):
            category = int(item["category_id"])
            objects[(image_id, index)] = GroundTruthObject(
                annotation_uid=f"pseudo-i{image_id}-g{index:04d}",
                image_id=image_id,
                ground_truth_index=index,
                fold=int(image_folds[image_id]),
                group_id=f"pseudo-fold-{image_folds[image_id]}",
                category_id=category,
                class_name=category_mapping[category],
                bbox_xyxy=tuple(float(value) for value in item["bbox_xyxy"]),
            )
            count += 1
    return FormalGroundTruth(
        boxes={key: list(value) for key, value in boxes.items()},
        objects=objects,
        image_ids=frozenset(boxes),
        annotation_count=count,
    )


def _size_bin(case: Mapping[str, Any]) -> str:
    x0, y0, x1, y1 = (float(value) for value in str(case["bbox_xyxy"]).split())
    short = min(x1 - x0, y1 - y0)
    if short < 32:
        return "tiny"
    if short < 96:
        return "small"
    if short < 256:
        return "medium"
    return "large"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--frontier", type=Path, required=True)
    parser.add_argument("--fdr-level", default="0.150")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    image_folds = {int(item["id"]): int(item["fold"]) for item in raw_gt["images"]}
    frontier = json.loads(args.frontier.read_text(encoding="utf-8"))
    thresholds = {
        int(key): float(value)
        for key, value in frontier["frontiers"][args.fdr_level][
            "crossfit_thresholds"
        ].items()
    }
    gt_boxes = load_coco_ground_truth(args.gt)
    predictions = load_coco_predictions(args.pred)
    filtered = apply_fold_thresholds(predictions, image_folds, thresholds)
    formal = _formal_ground_truth(
        gt_boxes,
        image_folds=image_folds,
        category_mapping=protocol.category_mapping,
    )
    summary, cases, _ = decompose_official_errors(
        formal,
        filtered,
        threshold=0.0,
        protocol=protocol,
        model_key="PSEUDO_TIGHT_IDENTITY",
        include_cases=True,
    )
    reason_size: dict[str, Counter[str]] = defaultdict(Counter)
    reason_fine: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        reason_size[str(case["reason"])][_size_bin(case)] += 1
        reason_fine[str(case["reason"])][str(case["category_id"])] += 1
    payload = {
        "status": "complete",
        "protocol": "pseudo10k_crossfit_workpoint_error_decomposition_v1",
        "warning": "Diagnostic attribution after exact official matching; not a second metric.",
        "fdr_level": args.fdr_level,
        "crossfit_thresholds": {str(key): value for key, value in thresholds.items()},
        "summary": summary,
        "reason_by_size": {key: dict(value) for key, value in sorted(reason_size.items())},
        "reason_by_fine": {
            key: dict(value.most_common()) for key, value in sorted(reason_fine.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
