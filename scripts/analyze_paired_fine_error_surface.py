#!/usr/bin/env python3
"""Compare two CV3 prediction ledgers at their frozen cross-fit workpoints.

The report keeps pooled counts for debugging, but all ranking and contribution
fields follow ``platform_observed_20260831``: fine-class metrics are averaged
inside each coarse class, then Ship/Aircraft/Vehicle are averaged equally.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.analysis.oof_detection import (
    FormalGroundTruth,
    GroundTruthObject,
    decompose_official_errors,
)
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import evaluate_ranking_metrics
from rsdet.evaluation.platform_protocol import (
    build_platform_observed_metrics,
    platform_metrics_payload,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _crossfit_thresholds(frontier: dict[str, Any], level: str) -> dict[int, float]:
    try:
        raw = frontier["frontiers"][level]["crossfit_thresholds"]
    except KeyError as exc:
        raise ValueError(f"frontier lacks crossfit thresholds for {level}") from exc
    thresholds = {int(fold): float(value) for fold, value in raw.items()}
    if set(thresholds) != {0, 1, 2}:
        raise ValueError("crossfit thresholds must contain folds 0, 1 and 2")
    return thresholds


def _filter_crossfit(
    predictions: dict[int, list[dict[str, Any]]],
    fold_by_image: dict[int, int],
    thresholds: dict[int, float],
) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [
            row
            for row in predictions.get(image_id, ())
            if float(row["score"]) >= thresholds[fold]
        ]
        for image_id, fold in fold_by_image.items()
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
                fold=int(image["fold"]),
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


def _ranking_payload(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    protocol: Any,
) -> tuple[dict[str, Any], Any]:
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
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


def _fine_rows(
    baseline_ranking: Any,
    candidate_ranking: Any,
    baseline_errors: dict[str, Any],
    candidate_errors: dict[str, Any],
    category_names: dict[int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fine_count = {
        name: item.fine_count for name, item in baseline_ranking.per_coarse.items()
    }
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
                "baseline_fn": baseline.fn,
                "candidate_fn": candidate.fn,
                "delta_fn": candidate.fn - baseline.fn,
                "baseline_recall": baseline.recall,
                "candidate_recall": candidate.recall,
                "delta_recall": recall_delta,
                "baseline_fdr": baseline.fdr,
                "candidate_fdr": candidate.fdr,
                "delta_fdr": fdr_delta,
                "platform_gate_recall_contribution_pp": (
                    100.0 * recall_delta / (3 * fine_count[coarse])
                ),
                "platform_gate_fdr_contribution_pp": (
                    100.0 * fdr_delta / (3 * fine_count[coarse])
                ),
                "baseline_errors": baseline_error,
                "candidate_errors": candidate_error,
                "delta_fp_bg": (
                    candidate_error["FP_BG"] - baseline_error["FP_BG"]
                ),
                "delta_fn_miss": (
                    candidate_error["FN_MISS"] - baseline_error["FN_MISS"]
                ),
                "delta_fp_cls": (
                    candidate_error["FP_CLS"] - baseline_error["FP_CLS"]
                ),
                "delta_fn_cls": (
                    candidate_error["FN_CLS"] - baseline_error["FN_CLS"]
                ),
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

    raw_gt = json.loads(args.gt.read_text(encoding="utf-8"))
    fold_by_image = {int(row["id"]): int(row["fold"]) for row in raw_gt["images"]}
    if set(fold_by_image.values()) != {0, 1, 2}:
        raise ValueError("ground truth must contain folds 0, 1 and 2")
    category_names = {
        int(row["id"]): str(row.get("name", row["id"]))
        for row in raw_gt["categories"]
    }
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    gt = load_coco_ground_truth(args.gt)
    formal_gt = _formal_ground_truth(raw_gt, gt, protocol)
    baseline_frontier = json.loads(args.baseline_frontier.read_text(encoding="utf-8"))
    candidate_frontier = json.loads(args.candidate_frontier.read_text(encoding="utf-8"))
    baseline_thresholds = _crossfit_thresholds(baseline_frontier, args.fdr_level)
    candidate_thresholds = _crossfit_thresholds(candidate_frontier, args.fdr_level)
    baseline_predictions = _filter_crossfit(
        load_coco_predictions(args.baseline), fold_by_image, baseline_thresholds
    )
    candidate_predictions = _filter_crossfit(
        load_coco_predictions(args.candidate), fold_by_image, candidate_thresholds
    )

    baseline_platform, baseline_ranking = _ranking_payload(
        gt, baseline_predictions, protocol
    )
    candidate_platform, candidate_ranking = _ranking_payload(
        gt, candidate_predictions, protocol
    )
    baseline_errors, _, _ = decompose_official_errors(
        formal_gt,
        baseline_predictions,
        threshold=0.0,
        protocol=protocol,
        model_key="baseline",
        include_cases=False,
    )
    candidate_errors, _, _ = decompose_official_errors(
        formal_gt,
        candidate_predictions,
        threshold=0.0,
        protocol=protocol,
        model_key="candidate",
        include_cases=False,
    )
    fine_rows = _fine_rows(
        baseline_ranking,
        candidate_ranking,
        baseline_errors,
        candidate_errors,
        category_names,
    )
    ranked_harm = sorted(
        fine_rows,
        key=lambda row: (
            row["platform_gate_recall_contribution_pp"]
            - row["platform_gate_fdr_contribution_pp"],
            row["category_id"],
        ),
    )
    payload = {
        "schema_version": "paired_fine_error_surface_platform_observed_v1",
        "metric_protocol": protocol.metric_protocol,
        "selection_uses_held_out_labels": False,
        "fdr_level": args.fdr_level,
        "baseline_threshold_by_fold": baseline_thresholds,
        "candidate_threshold_by_fold": candidate_thresholds,
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
                for name in ("ship", "aircraft", "vehicle")
            },
        },
        "baseline_error_decomposition": baseline_errors,
        "candidate_error_decomposition": candidate_errors,
        "fine_rows": fine_rows,
        "largest_harmful_fine_classes": ranked_harm[:10],
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
    flat_rows = [
        {key: value for key, value in row.items() if not isinstance(value, dict)}
        for row in fine_rows
    ]
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    print(json.dumps({key: payload[key] for key in ("delta", "largest_harmful_fine_classes")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
