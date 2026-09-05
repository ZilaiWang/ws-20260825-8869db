#!/usr/bin/env python3
"""Replay the pre-registered APRR policy on frozen P40/RFS/hierarchy OOF ledgers."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.contracts import Prediction
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
from rsdet.experiments.fixed_proxy import quality_contribution
from rsdet.experiments.task_vector_policy import (
    percentile,
    score_from_fine_counts,
    stress_incremental_vehicle_fp,
)
from rsdet.submission.aprr import AprrConfig, apply_aprr
from rsdet.utils.config import load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prediction(image_id: int, rows: list[dict[str, Any]]) -> Prediction:
    return Prediction(
        image_id=image_id,
        boxes_xyxy=[[float(value) for value in row["bbox_xyxy"]] for row in rows],
        scores=[float(row["score"]) for row in rows],
        labels=[int(row["category_id"]) for row in rows],
    )


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


def _evaluate(gt: Any, predictions: Any, protocol: Any) -> tuple[dict[str, Any], dict[int, dict[str, int]]]:
    ranking = evaluate_ranking_metrics(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=True,
    )
    platform = platform_metrics_payload(build_platform_observed_metrics(ranking))
    counts = {
        int(fine_id): {"tp": row.tp, "fp": row.fp, "fn": row.fn}
        for fine_id, row in ranking.per_fine.items()
    }
    return platform, counts


def _subset(mapping: dict[int, Any], image_ids: set[int]) -> dict[int, Any]:
    return {image_id: mapping.get(image_id, []) for image_id in sorted(image_ids)}


def _compose(
    primary: dict[int, list[dict[str, Any]]],
    ship: dict[int, list[dict[str, Any]]],
    vehicle: dict[int, list[dict[str, Any]]],
    *,
    image_ids: set[int],
    config: AprrConfig,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, int]]:
    output: dict[int, list[dict[str, Any]]] = {}
    totals: collections.Counter[str] = collections.Counter()
    for image_id in sorted(image_ids):
        prediction, stats = apply_aprr(
            _prediction(image_id, primary.get(image_id, [])),
            _prediction(image_id, ship.get(image_id, [])),
            _prediction(image_id, vehicle.get(image_id, [])),
            config=config,
        )
        output[image_id] = _mapping(prediction)
        totals.update(stats)
    return output, dict(totals)


def _events_by_group(gt: Any, predictions: Any, groups: dict[int, str], protocol: Any) -> Any:
    _, trace = evaluate_predictions_with_trace(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ledger: dict[str, dict[int, dict[str, int]]] = collections.defaultdict(
        lambda: collections.defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    )
    for event in trace.matches:
        ledger[groups[event.image_id]][event.category_id]["tp"] += 1
    for event in trace.unmatched_predictions:
        ledger[groups[event.image_id]][event.category_id]["fp"] += 1
    for event in trace.unmatched_ground_truths:
        ledger[groups[event.image_id]][event.category_id]["fn"] += 1
    return ledger


def _sample_counts(ledger: Any, sampled: list[str]) -> dict[int, dict[str, int]]:
    output: dict[int, dict[str, int]] = collections.defaultdict(
        lambda: {"tp": 0, "fp": 0, "fn": 0}
    )
    for group in sampled:
        for fine_id, row in ledger[group].items():
            for key in ("tp", "fp", "fn"):
                output[fine_id][key] += int(row[key])
    return dict(output)


def _coarse_delta(candidate: dict[str, Any], baseline: dict[str, Any], coarse: str, field: str) -> float:
    return float(candidate["per_coarse"][coarse][field]) - float(
        baseline["per_coarse"][coarse][field]
    )


def _fine_rates(counts: dict[int, dict[str, int]], fine_id: int) -> dict[str, float]:
    row = counts[fine_id]
    recall_denominator = int(row["tp"]) + int(row["fn"])
    fdr_denominator = int(row["tp"]) + int(row["fp"])
    return {
        "recall": int(row["tp"]) / recall_denominator if recall_denominator else 1.0,
        "fdr": int(row["fp"]) / fdr_denominator if fdr_denominator else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--rfs", type=Path, required=True)
    parser.add_argument("--hierarchy", type=Path, required=True)
    parser.add_argument("--image-groups", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260905)
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    gt_document = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    fold_ids = {
        fold: {int(row["id"]) for row in gt_document["images"] if int(row["fold"]) == fold}
        for fold in range(3)
    }
    if any(not image_ids for image_ids in fold_ids.values()):
        raise ValueError("ground truth must contain three non-empty folds")
    all_ids = set().union(*fold_ids.values())
    gt = load_coco_ground_truth(args.ground_truth)
    primary = load_coco_predictions(args.primary)
    rfs = load_coco_predictions(args.rfs)
    hierarchy = load_coco_predictions(args.hierarchy)
    if set(gt) != all_ids or any(set(rows) - all_ids for rows in (primary, rfs, hierarchy)):
        raise ValueError("prediction/ground-truth image universe mismatch")
    protocol = parse_evaluation_protocol(load_config(args.project_config))

    primary_thresholds = {0: 0.546, 1: 0.516, 2: 0.501}
    rfs_thresholds = {0: 0.471, 1: 0.451, 2: 0.446}
    variants = {
        "B0": (frozenset({0, 1, 2}), None),
        "S012": (frozenset({0, 1, 2}), None),
        "S0123": (frozenset({0, 1, 2, 3}), None),
        "V060": (frozenset({0, 1, 2}), 0.60),
        "V065": (frozenset({0, 1, 2}), 0.65),
        "S012_V060": (frozenset({0, 1, 2}), 0.60),
        "S012_V065": (frozenset({0, 1, 2}), 0.65),
        "S0123_V060": (frozenset({0, 1, 2, 3}), 0.60),
        "S0123_V065": (frozenset({0, 1, 2, 3}), 0.65),
    }
    outputs: dict[str, dict[int, list[dict[str, Any]]]] = {name: {} for name in variants}
    summaries: dict[str, Any] = {name: {"folds": {}} for name in variants}
    empty: dict[int, list[dict[str, Any]]] = {}
    for fold, image_ids in fold_ids.items():
        for name, (ship_labels, vehicle_protect) in variants.items():
            enable_ship = name.startswith("S")
            enable_vehicle = "_V" in name or name.startswith("V")
            config = AprrConfig(
                primary_threshold=primary_thresholds[fold],
                ship_support_threshold=rfs_thresholds[fold],
                vehicle_protect_threshold=(
                    float(vehicle_protect) if enable_vehicle else primary_thresholds[fold]
                ),
                ship_rescue_labels=ship_labels,
            )
            composed, stats = _compose(
                primary,
                rfs if enable_ship else empty,
                hierarchy if enable_vehicle else empty,
                image_ids=image_ids,
                config=config,
            )
            outputs[name].update(composed)
            platform, counts = _evaluate(_subset(gt, image_ids), composed, protocol)
            summaries[name]["folds"][str(fold)] = {
                "platform": platform,
                "fine_counts": counts,
                "policy_stats": stats,
            }

    for name in variants:
        platform, counts = _evaluate(gt, outputs[name], protocol)
        summaries[name]["aggregate"] = {"platform": platform, "fine_counts": counts}
    baseline = summaries["B0"]
    for name, summary in summaries.items():
        summary["quality_delta"] = quality_contribution(summary["aggregate"]["platform"]) - quality_contribution(
            baseline["aggregate"]["platform"]
        )
        summary["fold_quality_deltas"] = {
            str(fold): quality_contribution(summary["folds"][str(fold)]["platform"])
            - quality_contribution(baseline["folds"][str(fold)]["platform"])
            for fold in range(3)
        }
        summary["coarse_deltas"] = {
            coarse: {
                field: _coarse_delta(
                    summary["aggregate"]["platform"],
                    baseline["aggregate"]["platform"],
                    coarse,
                    field,
                )
                for field in ("macro_recall", "macro_fdr")
            }
            for coarse in ("ship", "aircraft", "vehicle")
        }
        summary["fine_deltas"] = {
            str(fine_id): {
                field: _fine_rates(summary["aggregate"]["fine_counts"], fine_id)[field]
                - _fine_rates(baseline["aggregate"]["fine_counts"], fine_id)[field]
                for field in ("recall", "fdr")
            }
            for fine_id in range(25)
        }
        stressed = stress_incremental_vehicle_fp(
            baseline["aggregate"]["fine_counts"],
            summary["aggregate"]["fine_counts"],
            multiplier=6.0,
        )
        summary["vehicle_incremental_fp_6x_score_delta"] = float(
            score_from_fine_counts(stressed, protocol.category_mapping)["total_score"]
        ) - float(
            score_from_fine_counts(
                baseline["aggregate"]["fine_counts"], protocol.category_mapping
            )["total_score"]
        )

    combined_names = [name for name in variants if "_V" in name]
    selected = max(
        combined_names,
        key=lambda name: (
            min(summaries[name]["fold_quality_deltas"].values()),
            summaries[name]["quality_delta"],
            name,
        ),
    )
    groups_document = json.loads(args.image_groups.read_text(encoding="utf-8"))
    groups = {int(row["image_id"]): str(row["group_id"]) for row in groups_document["samples"]}
    if set(gt) - set(groups):
        raise ValueError("image-groups file does not cover the complete OOF universe")
    base_ledger = _events_by_group(gt, outputs["B0"], groups, protocol)
    selected_ledger = _events_by_group(gt, outputs[selected], groups, protocol)
    group_names = sorted({groups[image_id] for image_id in gt})
    rng = random.Random(args.seed)
    bootstrap: list[float] = []
    for _ in range(args.bootstrap_iterations):
        sampled = [rng.choice(group_names) for _ in group_names]
        base_score = score_from_fine_counts(
            _sample_counts(base_ledger, sampled), protocol.category_mapping
        )["total_score"]
        selected_score = score_from_fine_counts(
            _sample_counts(selected_ledger, sampled), protocol.category_mapping
        )["total_score"]
        bootstrap.append(float(selected_score) - float(base_score))
    summaries[selected]["source_group_bootstrap"] = {
        "groups": len(group_names),
        "iterations": args.bootstrap_iterations,
        "seed": args.seed,
        "p10": percentile(bootstrap, 0.10),
        "p50": percentile(bootstrap, 0.50),
        "p90": percentile(bootstrap, 0.90),
    }

    ship_gates: dict[str, Any] = {}
    for name in ("S012", "S0123"):
        fold_ship_fine_deltas = {
            str(fold): {
                str(fine_id): {
                    field: _fine_rates(
                        summaries[name]["folds"][str(fold)]["fine_counts"], fine_id
                    )[field]
                    - _fine_rates(
                        baseline["folds"][str(fold)]["fine_counts"], fine_id
                    )[field]
                    for field in ("recall", "fdr")
                }
                for fine_id in (0, 1, 2)
            }
            for fold in range(3)
        }
        ship_gates[name] = {
            "all_fold_quality_positive": all(
                delta > 0 for delta in summaries[name]["fold_quality_deltas"].values()
            ),
            "aggregate_recall_gain_ge_3pp": summaries[name]["coarse_deltas"]["ship"]["macro_recall"] >= 0.03,
            "aggregate_fdr_increase_le_2pp": summaries[name]["coarse_deltas"]["ship"]["macro_fdr"] <= 0.02,
            "hm_lqs_qhs_no_recall_regression": all(
                row["recall"] >= 0.0
                for fold in fold_ship_fine_deltas.values()
                for row in fold.values()
            ),
            "hm_lqs_qhs_no_catastrophic_fdr_increase": all(
                row["fdr"] <= 0.25
                for fold in fold_ship_fine_deltas.values()
                for row in fold.values()
            ),
            "hm_lqs_qhs_fold_deltas": fold_ship_fine_deltas,
        }
    vehicle_gates: dict[str, Any] = {}
    for name in ("V060", "V065"):
        vehicle_gates[name] = {
            "all_fold_quality_positive": all(
                delta > 0 for delta in summaries[name]["fold_quality_deltas"].values()
            ),
            "all_fold_recall_loss_le_0p5pp": all(
                _coarse_delta(
                    summaries[name]["folds"][str(fold)]["platform"],
                    baseline["folds"][str(fold)]["platform"],
                    "vehicle",
                    "macro_recall",
                ) >= -0.005
                for fold in range(3)
            ),
            "aggregate_fdr_drop_ge_3pp": summaries[name]["coarse_deltas"]["vehicle"]["macro_fdr"] <= -0.03,
            "incremental_fp_6x_positive": summaries[name]["vehicle_incremental_fp_6x_score_delta"] > 0,
        }
    selected_summary = summaries[selected]
    combined_gate = {
        "selected": selected,
        "all_fold_quality_positive": all(
            delta > 0 for delta in selected_summary["fold_quality_deltas"].values()
        ),
        "source_group_bootstrap_p10_positive": selected_summary["source_group_bootstrap"]["p10"] > 0,
        "aircraft_exactly_unchanged": selected_summary["coarse_deltas"]["aircraft"] == {
            "macro_recall": 0.0,
            "macro_fdr": 0.0,
        },
    }
    combined_gate["passed_evidence_gates"] = all(
        value is True for key, value in combined_gate.items() if key != "selected"
    )
    result = {
        "status": "complete",
        "protocol": "aprr_pre_registered_cv3_replay_v1",
        "role": "frozen_oof_counterfactual_not_hidden_score_prediction",
        "input_sha256": {
            "ground_truth": _sha256(args.ground_truth),
            "primary": _sha256(args.primary),
            "rfs": _sha256(args.rfs),
            "hierarchy": _sha256(args.hierarchy),
            "image_groups": _sha256(args.image_groups),
        },
        "frozen_thresholds": {
            "primary_by_fold": primary_thresholds,
            "rfs_support_by_fold": rfs_thresholds,
            "hierarchy_support": 0.546,
            "ship_support_iou": 0.50,
            "vehicle_support_iou": 0.35,
            "vehicle_protect": [0.60, 0.65],
        },
        "variants": summaries,
        "gates": {"ship": ship_gates, "vehicle": vehicle_gates, "combined": combined_gate},
        "decision": (
            "eligible_for_runtime_validation"
            if combined_gate["passed_evidence_gates"]
            else "stop_aprr_before_runtime"
        ),
        "limitations": [
            "The full P40 and the short OOF checkpoints have different maturity/calibration regimes.",
            "Hard/Sentinel and pseudo-10K are not independent hidden-distribution replicas.",
            "No OOF delta is mechanically added to the official 76.601 anchor.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "selected": selected, "gates": combined_gate}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
