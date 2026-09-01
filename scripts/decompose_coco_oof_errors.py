#!/usr/bin/env python3
"""Decompose a COCO OOF ledger with the frozen official error hierarchy."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import (
    FormalGroundTruth,
    GroundTruthObject,
    decompose_official_errors,
)
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--pred", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.15)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-key", default="Y5-OOF")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    raw = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(row["id"]): int(row.get("fold", -1)) for row in raw["images"]}
    group_by_image = {
        int(row["id"]): str(row.get("group_id", f"image-{row['id']}"))
        for row in raw["images"]
    }
    gt = load_coco_ground_truth(args.gt)
    objects: dict[tuple[int, int], GroundTruthObject] = {}
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    for image_id, rows in gt.items():
        for index, row in enumerate(rows):
            category = int(row["category_id"])
            objects[(image_id, index)] = GroundTruthObject(
                annotation_uid=f"coco-i{image_id}-g{index:04d}",
                image_id=image_id,
                ground_truth_index=index,
                fold=fold_by_image[image_id],
                group_id=group_by_image[image_id],
                category_id=category,
                class_name=protocol.category_mapping[category],
                bbox_xyxy=tuple(float(value) for value in row["bbox_xyxy"]),
            )
    formal = FormalGroundTruth(
        boxes=gt,
        objects=objects,
        image_ids=frozenset(fold_by_image),
        annotation_count=len(objects),
    )
    summary, cases, _ = decompose_official_errors(
        formal,
        load_coco_predictions(args.pred),
        threshold=args.threshold,
        protocol=protocol,
        model_key=args.model_key,
        include_cases=True,
    )
    summary["metric_protocol"] = protocol.metric_protocol
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "error_decomposition.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "error_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader()
        writer.writerows(cases)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
