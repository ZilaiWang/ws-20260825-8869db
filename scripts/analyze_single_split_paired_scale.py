#!/usr/bin/env python3
"""Paired diagnosis for two detectors on one held-out split.

The score thresholds are read from already-produced single-split frontiers.
This script does not select a new workpoint.  It reports platform-observed
metrics, the frozen error hierarchy, per-fine deltas and paired GT recovery by
native object size.  Its results are diagnostic because the frontier
thresholds were selected with labels from the same held-out split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import (
    FormalGroundTruth,
    GroundTruthObject,
    decompose_official_errors,
)
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import (
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

SIZE_BINS = (
    ("lt48", 0.0, 48.0),
    ("48to80", 48.0, 80.0),
    ("80to128", 80.0, 128.0),
    ("ge128", 128.0, math.inf),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _threshold(frontier: dict[str, Any], level: str) -> float:
    try:
        return float(frontier["frontiers"][level]["threshold"])
    except KeyError as exc:
        raise ValueError(f"frontier lacks level {level}") from exc


def _filter(
    predictions: dict[int, list[dict[str, Any]]], threshold: float
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row for row in rows if float(row["score"]) >= threshold
        ]
        for image_id, rows in predictions.items()
    }


def _formal_ground_truth(
    raw: dict[str, Any], gt: dict[int, list[dict[str, Any]]], protocol: Any
) -> FormalGroundTruth:
    image_rows = {int(row["id"]): row for row in raw["images"]}
    objects: dict[tuple[int, int], GroundTruthObject] = {}
    for image_id, rows in gt.items():
        image = image_rows[image_id]
        for index, row in enumerate(rows):
            category_id = int(row["category_id"])
            objects[(image_id, index)] = GroundTruthObject(
                annotation_uid=f"coco-i{image_id}-g{index:04d}",
                image_id=image_id,
                ground_truth_index=index,
                fold=0,
                group_id=str(image.get("group_id", f"image-{image_id}")),
                category_id=category_id,
                class_name=protocol.category_mapping[category_id],
                bbox_xyxy=tuple(float(value) for value in row["bbox_xyxy"]),
            )
    return FormalGroundTruth(
        boxes=gt,
        objects=objects,
        image_ids=frozenset(image_rows),
        annotation_count=len(objects),
    )


def _platform_payload(
    gt: dict[int, list[dict[str, Any]]],
    predictions: dict[int, list[dict[str, Any]]],
    protocol: Any,
) -> tuple[dict[str, Any], Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    platform = platform_metrics_payload(
        build_platform_observed_metrics(
            ranking,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
            latency_seconds=None,
            latency_max_seconds=protocol.latency_max_seconds,
        )
    )
    return platform, ranking


def _size_bin(bbox_xyxy: tuple[float, float, float, float] | list[float]) -> str:
    width = max(0.0, float(bbox_xyxy[2]) - float(bbox_xyxy[0]))
    height = max(0.0, float(bbox_xyxy[3]) - float(bbox_xyxy[1]))
    scale = math.sqrt(width * height)
    return next(name for name, lower, upper in SIZE_BINS if lower <= scale < upper)


def _paired_recovery(
    formal_gt: FormalGroundTruth,
    baseline_trace: Any,
    candidate_trace: Any,
) -> dict[str, Any]:
    baseline_hits = {
        (item.image_id, item.ground_truth_index) for item in baseline_trace.matches
    }
    candidate_hits = {
        (item.image_id, item.ground_truth_index) for item in candidate_trace.matches
    }
    accumulators: dict[str, dict[str, Counter[str]]] = {
        "by_size": defaultdict(Counter),
        "by_coarse": defaultdict(Counter),
        "by_fine": defaultdict(Counter),
    }
    for key, obj in formal_gt.objects.items():
        baseline_hit = key in baseline_hits
        candidate_hit = key in candidate_hits
        transition = (
            "both"
            if baseline_hit and candidate_hit
            else "baseline_only"
            if baseline_hit
            else "candidate_only"
            if candidate_hit
            else "neither"
        )
        labels = {
            "by_size": _size_bin(obj.bbox_xyxy),
            "by_coarse": obj.class_name,
            "by_fine": str(obj.category_id),
        }
        for dimension, label in labels.items():
            row = accumulators[dimension][label]
            row["gt"] += 1
            row[transition] += 1
            row["baseline_tp"] += int(baseline_hit)
            row["candidate_tp"] += int(candidate_hit)

    output: dict[str, Any] = {}
    for dimension, grouped in accumulators.items():
        output[dimension] = {}
        for label, counts in sorted(grouped.items()):
            gt_count = counts["gt"]
            output[dimension][label] = {
                **dict(counts),
                "baseline_recall": counts["baseline_tp"] / gt_count,
                "candidate_recall": counts["candidate_tp"] / gt_count,
                "delta_recall_pp": 100.0
                * (counts["candidate_tp"] - counts["baseline_tp"])
                / gt_count,
                "net_recovered_gt": counts["candidate_only"]
                - counts["baseline_only"],
            }
    return output


def _prediction_error_size(cases: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        if case["case_side"] != "prediction":
            continue
        bbox = [float(value) for value in str(case["bbox_xyxy"]).split()]
        grouped[_size_bin(bbox)][str(case["reason"])] += 1
    return {name: dict(counts) for name, counts in sorted(grouped.items())}


def _fine_rows(
    baseline_ranking: Any,
    candidate_ranking: Any,
    baseline_errors: dict[str, Any],
    candidate_errors: dict[str, Any],
    category_names: dict[int, str],
) -> list[dict[str, Any]]:
    fine_count = {
        name: item.fine_count for name, item in baseline_ranking.per_coarse.items()
    }
    rows: list[dict[str, Any]] = []
    for category_id in sorted(baseline_ranking.per_fine):
        baseline = baseline_ranking.per_fine[category_id]
        candidate = candidate_ranking.per_fine[category_id]
        coarse = baseline.coarse_class
        baseline_error = baseline_errors["per_fine_category"][str(category_id)]
        candidate_error = candidate_errors["per_fine_category"][str(category_id)]
        recall_delta = candidate.recall - baseline.recall
        fdr_delta = candidate.fdr - baseline.fdr
        rows.append(
            {
                "category_id": category_id,
                "category_name": category_names.get(category_id, str(category_id)),
                "coarse_class": coarse,
                "gt": baseline.tp + baseline.fn,
                "baseline_tp": baseline.tp,
                "candidate_tp": candidate.tp,
                "delta_tp": candidate.tp - baseline.tp,
                "baseline_fp": baseline.fp,
                "candidate_fp": candidate.fp,
                "delta_fp": candidate.fp - baseline.fp,
                "baseline_recall": baseline.recall,
                "candidate_recall": candidate.recall,
                "delta_recall_pp": 100.0 * recall_delta,
                "baseline_fdr": baseline.fdr,
                "candidate_fdr": candidate.fdr,
                "delta_fdr_pp": 100.0 * fdr_delta,
                "platform_recall_contribution_pp": (
                    100.0 * recall_delta / (3 * fine_count[coarse])
                ),
                "platform_fdr_contribution_pp": (
                    100.0 * fdr_delta / (3 * fine_count[coarse])
                ),
                "delta_fp_bg": candidate_error["FP_BG"] - baseline_error["FP_BG"],
                "delta_fp_cls": candidate_error["FP_CLS"] - baseline_error["FP_CLS"],
                "delta_fp_loc": candidate_error["FP_LOC"] - baseline_error["FP_LOC"],
                "delta_fp_dup": candidate_error["FP_DUP"] - baseline_error["FP_DUP"],
                "delta_fn_miss": candidate_error["FN_MISS"] - baseline_error["FN_MISS"],
                "delta_fn_cls": candidate_error["FN_CLS"] - baseline_error["FN_CLS"],
                "delta_fn_loc": candidate_error["FN_LOC"] - baseline_error["FN_LOC"],
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-frontier", type=Path, required=True)
    parser.add_argument("--candidate-frontier", type=Path, required=True)
    parser.add_argument("--fdr-level", default="0.150")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    protocol = parse_evaluation_protocol(load_config(args.project_config))
    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    category_names = {
        int(row["id"]): str(row.get("name", row["id"]))
        for row in raw_gt["categories"]
    }
    gt = load_coco_ground_truth(args.gt)
    formal_gt = _formal_ground_truth(raw_gt, gt, protocol)
    baseline_raw = load_coco_predictions(args.baseline)
    candidate_raw = load_coco_predictions(args.candidate)
    baseline_threshold = _threshold(
        json.loads(args.baseline_frontier.read_text(encoding="utf-8")),
        args.fdr_level,
    )
    candidate_threshold = _threshold(
        json.loads(args.candidate_frontier.read_text(encoding="utf-8")),
        args.fdr_level,
    )
    baseline_predictions = _filter(baseline_raw, baseline_threshold)
    candidate_predictions = _filter(candidate_raw, candidate_threshold)

    baseline_platform, baseline_ranking = _platform_payload(
        gt, baseline_predictions, protocol
    )
    candidate_platform, candidate_ranking = _platform_payload(
        gt, candidate_predictions, protocol
    )
    _, baseline_trace = evaluate_predictions_with_trace(
        gt,
        baseline_predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    _, candidate_trace = evaluate_predictions_with_trace(
        gt,
        candidate_predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    baseline_errors, baseline_cases, _ = decompose_official_errors(
        formal_gt,
        baseline_predictions,
        threshold=0.0,
        protocol=protocol,
        model_key="s1024",
        include_cases=True,
    )
    candidate_errors, candidate_cases, _ = decompose_official_errors(
        formal_gt,
        candidate_predictions,
        threshold=0.0,
        protocol=protocol,
        model_key="s1280",
        include_cases=True,
    )
    fine_rows = _fine_rows(
        baseline_ranking,
        candidate_ranking,
        baseline_errors,
        candidate_errors,
        category_names,
    )
    payload = {
        "schema_version": "single_split_paired_scale_diagnosis_v1",
        "metric_protocol": protocol.metric_protocol,
        "diagnostic_only": True,
        "selection_uses_same_split_labels": True,
        "warning": (
            "Both thresholds are oracle points selected on this same held-out split; "
            "do not copy either threshold into deployment."
        ),
        "fdr_level": args.fdr_level,
        "baseline_threshold": baseline_threshold,
        "candidate_threshold": candidate_threshold,
        "baseline": baseline_platform,
        "candidate": candidate_platform,
        "delta": {
            "gate_recall_pp": 100.0
            * (candidate_platform["gate_recall"] - baseline_platform["gate_recall"]),
            "gate_fdr_pp": 100.0
            * (candidate_platform["gate_fdr"] - baseline_platform["gate_fdr"]),
            "per_coarse": {
                name: {
                    "macro_recall_pp": 100.0
                    * (
                        candidate_platform["per_coarse"][name]["macro_recall"]
                        - baseline_platform["per_coarse"][name]["macro_recall"]
                    ),
                    "macro_fdr_pp": 100.0
                    * (
                        candidate_platform["per_coarse"][name]["macro_fdr"]
                        - baseline_platform["per_coarse"][name]["macro_fdr"]
                    ),
                }
                for name in protocol.class_names
            },
        },
        "paired_gt_recovery": _paired_recovery(
            formal_gt, baseline_trace, candidate_trace
        ),
        "baseline_error_decomposition": baseline_errors,
        "candidate_error_decomposition": candidate_errors,
        "prediction_error_by_size": {
            "baseline": _prediction_error_size(baseline_cases),
            "candidate": _prediction_error_size(candidate_cases),
        },
        "fine_rows": fine_rows,
        "largest_recall_gains": sorted(
            fine_rows,
            key=lambda row: (-row["platform_recall_contribution_pp"], row["category_id"]),
        )[:10],
        "largest_fdr_costs": sorted(
            fine_rows,
            key=lambda row: (-row["platform_fdr_contribution_pp"], row["category_id"]),
        )[:10],
        "input_sha256": {
            "gt": _sha256(args.gt),
            "baseline": _sha256(args.baseline),
            "candidate": _sha256(args.candidate),
            "baseline_frontier": _sha256(args.baseline_frontier),
            "candidate_frontier": _sha256(args.candidate_frontier),
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fine_rows[0]))
        writer.writeheader()
        writer.writerows(fine_rows)
    print(
        json.dumps(
            {
                "delta": payload["delta"],
                "paired_gt_recovery_by_size": payload["paired_gt_recovery"]["by_size"],
                "largest_recall_gains": payload["largest_recall_gains"][:5],
                "largest_fdr_costs": payload["largest_fdr_costs"][:5],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
