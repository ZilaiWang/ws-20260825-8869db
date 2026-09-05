#!/usr/bin/env python3
"""Strict outer-policy CV and robustness audit for a Vehicle task vector."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
from rsdet.experiments.task_vector_policy import (
    percentile,
    score_from_fine_counts,
    select_conservative_alpha,
    stress_incremental_vehicle_fp,
)
from rsdet.utils.config import load_config


def _filtered(path: Path, threshold: float) -> dict[int, list[dict[str, Any]]]:
    return {
        image_id: [row for row in rows if float(row["score"]) >= threshold]
        for image_id, rows in load_coco_predictions(path).items()
    }


def _merge(mappings: list[dict[int, list[dict[str, Any]]]]) -> dict[int, list[dict[str, Any]]]:
    output: dict[int, list[dict[str, Any]]] = {}
    for mapping in mappings:
        overlap = set(output) & set(mapping)
        if overlap:
            raise ValueError(f"fold image IDs overlap: {sorted(overlap)[:10]}")
        output.update(mapping)
    return output


def _canonical_nonvehicle(rows: dict[int, list[dict[str, Any]]]) -> list[tuple[Any, ...]]:
    result = []
    for image_id, items in rows.items():
        for row in items:
            if int(row["category_id"]) == 24:
                continue
            result.append(
                (
                    image_id,
                    int(row["category_id"]),
                    tuple(float(value) for value in row["bbox_xyxy"]),
                    float(row["score"]),
                )
            )
    return sorted(result)


def _compose_vehicle_only(
    baseline: dict[int, list[dict[str, Any]]],
    task_vector: dict[int, list[dict[str, Any]]],
) -> dict[int, list[dict[str, Any]]]:
    """Keep P40 for 0..23 and take only class 24 from the task vector."""

    return {
        image_id: [
            *(
                row
                for row in baseline.get(image_id, [])
                if int(row["category_id"]) != 24
            ),
            *(
                row
                for row in task_vector.get(image_id, [])
                if int(row["category_id"]) == 24
            ),
        ]
        for image_id in sorted(set(baseline) | set(task_vector))
    }


def _evaluate(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    protocol: Any,
) -> tuple[dict[str, Any], dict[int, dict[str, int]]]:
    ranking = evaluate_ranking_metrics(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    platform = build_platform_observed_metrics(ranking, latency_seconds=0.0)
    counts = {
        int(fine_id): {"tp": row.tp, "fp": row.fp, "fn": row.fn}
        for fine_id, row in ranking.per_fine.items()
    }
    return platform_metrics_payload(platform), counts


def _events_by_group(
    gt: dict[int, list[dict[str, Any]]],
    pred: dict[int, list[dict[str, Any]]],
    group_of_image: dict[int, str],
    protocol: Any,
) -> dict[str, dict[int, dict[str, int]]]:
    _, trace = evaluate_predictions_with_trace(
        gt,
        pred,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ledger: dict[str, dict[int, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )
    for event in trace.matches:
        ledger[group_of_image[event.image_id]][event.category_id]["tp"] += 1
    for event in trace.unmatched_predictions:
        ledger[group_of_image[event.image_id]][event.category_id]["fp"] += 1
    for event in trace.unmatched_ground_truths:
        ledger[group_of_image[event.image_id]][event.category_id]["fn"] += 1
    return ledger


def _sum_sampled_groups(
    ledger: dict[str, dict[int, dict[str, int]]], sampled: list[str]
) -> dict[int, dict[str, int]]:
    output: dict[int, dict[str, int]] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    for group in sampled:
        for fine_id, row in ledger.get(group, {}).items():
            for name in ("tp", "fp", "fn"):
                output[fine_id][name] += int(row[name])
    return dict(output)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image-groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.546)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    folds = sorted(manifest["folds"], key=lambda row: int(row["fold"]))
    if [int(row["fold"]) for row in folds] != [0, 1, 2]:
        raise ValueError("manifest must contain folds 0, 1, 2 exactly once")
    alphas = sorted(float(value) for value in manifest["alphas"])
    if alphas != [0.0, 0.125, 0.25, 0.5]:
        raise ValueError("frozen alpha grid is 0, 0.125, 0.25, 0.5")
    protocol = parse_evaluation_protocol(load_config(args.project_config))

    fold_gt: dict[int, dict[int, list[dict[str, Any]]]] = {}
    fold_base: dict[int, dict[int, list[dict[str, Any]]]] = {}
    fold_candidates: dict[int, dict[float, dict[int, list[dict[str, Any]]]]] = {}
    raw_parity: dict[str, bool] = {}
    composed_parity: dict[str, bool] = {}
    for row in folds:
        fold = int(row["fold"])
        fold_gt[fold] = load_coco_ground_truth(Path(row["gt"]))
        fold_base[fold] = _filtered(Path(row["baseline"]), args.threshold)
        fold_candidates[fold] = {}
        for raw_alpha, path in row["candidates"].items():
            alpha = float(raw_alpha)
            raw_candidate = _filtered(Path(path), args.threshold)
            candidate = _compose_vehicle_only(fold_base[fold], raw_candidate)
            fold_candidates[fold][alpha] = candidate
            key = f"fold_{fold}_alpha_{alpha:g}"
            raw_parity[key] = _canonical_nonvehicle(
                raw_candidate
            ) == _canonical_nonvehicle(fold_base[fold])
            composed_parity[key] = _canonical_nonvehicle(candidate) == _canonical_nonvehicle(
                fold_base[fold]
            )
    if not all(composed_parity.values()):
        failed = [key for key, passed in composed_parity.items() if not passed]
        raise AssertionError(f"class-disjoint composition failed: {failed}")

    train_scores: dict[int, dict[float, float]] = {}
    selected: dict[int, float] = {}
    for heldout in range(3):
        training = [fold for fold in range(3) if fold != heldout]
        gt = _merge([fold_gt[fold] for fold in training])
        scores: dict[float, float] = {}
        for alpha in alphas:
            pred = _merge([fold_candidates[fold][alpha] for fold in training])
            platform, _ = _evaluate(gt, pred, protocol)
            scores[alpha] = float(platform["absolute_score"])
        train_scores[heldout] = scores
        selected[heldout] = select_conservative_alpha(scores)

    fold_tests: dict[int, dict[str, Any]] = {}
    for heldout in range(3):
        baseline, _ = _evaluate(fold_gt[heldout], fold_base[heldout], protocol)
        candidate, _ = _evaluate(
            fold_gt[heldout], fold_candidates[heldout][selected[heldout]], protocol
        )
        fold_tests[heldout] = {
            "selected_alpha": selected[heldout],
            "baseline": baseline,
            "candidate": candidate,
            "score_delta": float(candidate["absolute_score"])
            - float(baseline["absolute_score"]),
            "vehicle_recall_delta": candidate["per_coarse"]["vehicle"]["macro_recall"]
            - baseline["per_coarse"]["vehicle"]["macro_recall"],
        }

    aggregate_gt = _merge([fold_gt[fold] for fold in range(3)])
    aggregate_base = _merge([fold_base[fold] for fold in range(3)])
    aggregate_candidate = _merge(
        [fold_candidates[fold][selected[fold]] for fold in range(3)]
    )
    base_platform, base_counts = _evaluate(aggregate_gt, aggregate_base, protocol)
    candidate_platform, candidate_counts = _evaluate(
        aggregate_gt, aggregate_candidate, protocol
    )
    score_delta = float(candidate_platform["absolute_score"]) - float(
        base_platform["absolute_score"]
    )

    groups_payload = json.loads(args.image_groups.read_text(encoding="utf-8"))
    group_of_image = {
        int(row["image_id"]): str(row["group_id"])
        for row in groups_payload["samples"]
    }
    missing_groups = set(aggregate_gt) - set(group_of_image)
    if missing_groups:
        raise ValueError(f"missing image groups: {sorted(missing_groups)[:10]}")
    base_events = _events_by_group(
        aggregate_gt, aggregate_base, group_of_image, protocol
    )
    candidate_events = _events_by_group(
        aggregate_gt, aggregate_candidate, group_of_image, protocol
    )
    groups = sorted({group_of_image[image_id] for image_id in aggregate_gt})
    rng = random.Random(args.seed)
    bootstrap_deltas: list[float] = []
    for _ in range(args.bootstrap_iterations):
        sampled = [rng.choice(groups) for _ in groups]
        sampled_base = _sum_sampled_groups(base_events, sampled)
        sampled_candidate = _sum_sampled_groups(candidate_events, sampled)
        base_score = score_from_fine_counts(
            sampled_base, protocol.category_mapping
        )["total_score"]
        candidate_score = score_from_fine_counts(
            sampled_candidate, protocol.category_mapping
        )["total_score"]
        bootstrap_deltas.append(float(candidate_score) - float(base_score))

    stressed_counts = stress_incremental_vehicle_fp(
        base_counts, candidate_counts, multiplier=6.0
    )
    stressed_score = score_from_fine_counts(
        stressed_counts, protocol.category_mapping
    )["total_score"]
    base_count_score = score_from_fine_counts(
        base_counts, protocol.category_mapping
    )["total_score"]
    stress_delta = float(stressed_score) - float(base_count_score)
    gates = {
        "every_outer_fold_score_delta_positive": all(
            row["score_delta"] > 0.0 for row in fold_tests.values()
        ),
        "max_vehicle_recall_loss_le_0p5pp": all(
            row["vehicle_recall_delta"] >= -0.005 for row in fold_tests.values()
        ),
        "group_bootstrap_p10_positive": percentile(bootstrap_deltas, 0.10) > 0.0,
        "incremental_fp_6x_stress_positive": stress_delta > 0.0,
        "ship_aircraft_prediction_parity": all(composed_parity.values()),
        "background_100mp_not_worse": None,
    }
    passed_pre_background = all(
        value is True for key, value in gates.items() if key != "background_100mp_not_worse"
    )
    payload = {
        "status": "complete",
        "protocol": "vehicle_class_task_vector_outer_policy_cv_v1",
        "threshold": args.threshold,
        "alphas": alphas,
        "training_fold_scores_by_heldout": {
            str(fold): {str(alpha): score for alpha, score in scores.items()}
            for fold, scores in train_scores.items()
        },
        "selected_alpha_by_heldout": {str(key): value for key, value in selected.items()},
        "outer_fold_tests": {str(key): value for key, value in fold_tests.items()},
        "aggregate": {
            "baseline": base_platform,
            "candidate": candidate_platform,
            "score_delta": score_delta,
        },
        "bootstrap": {
            "unit": "source_group",
            "group_count": len(groups),
            "iterations": args.bootstrap_iterations,
            "seed": args.seed,
            "score_delta_p10": percentile(bootstrap_deltas, 0.10),
            "score_delta_p50": percentile(bootstrap_deltas, 0.50),
            "score_delta_p90": percentile(bootstrap_deltas, 0.90),
        },
        "vehicle_incremental_fp_stress": {
            "multiplier": 6.0,
            "score_delta": stress_delta,
            "baseline_vehicle": base_counts.get(24, {}),
            "candidate_vehicle": candidate_counts.get(24, {}),
            "stressed_vehicle": stressed_counts.get(24, {}),
        },
        "raw_single_checkpoint_nonvehicle_prediction_parity": raw_parity,
        "raw_single_checkpoint_deployment_admissible": all(raw_parity.values()),
        "class_disjoint_composed_nonvehicle_prediction_parity": composed_parity,
        "deployment_requirement_if_selected": (
            "implement_and_validate_one-pass coarse-preserving delta head; "
            "offline class-disjoint composition is evaluation-only"
        ),
        "gates": gates,
        "pre_background_admission": passed_pre_background,
        "decision": (
            "pending_background_100mp" if passed_pre_background else "reject_task_vector"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
