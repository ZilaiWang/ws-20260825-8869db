#!/usr/bin/env python3
"""Score BATIS ledger replays and decompose recovered/lost GT objects."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.absolute_score import platform_confirmed_score
from rsdet.evaluation.coco import load_coco_ground_truth, load_coco_predictions
from rsdet.evaluation.official_metric import (
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)
from rsdet.evaluation.platform_protocol import build_platform_observed_metrics
from rsdet.evaluation.protocol import parse_evaluation_protocol
from rsdet.utils.config import load_config

VARIANTS = ("h0_legacy", "h1a_prefilter", "h1b_threshold_safe", "h2_owner")


def _trace_keys(trace: Any) -> tuple[set[tuple[int, str, int, int]], Counter[str]]:
    matched = {
        (row.image_id, row.class_name, row.category_id, row.ground_truth_index)
        for row in trace.matches
    }
    false_positives = Counter(row.class_name for row in trace.unmatched_predictions)
    return matched, false_positives


def _evaluate(
    gt: Any,
    predictions: Any,
    protocol: Any,
    *,
    complete_taxonomy: bool,
    latency_seconds: float,
) -> tuple[dict[str, Any], Any]:
    pooled, trace = evaluate_predictions_with_trace(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
    )
    ranking = evaluate_ranking_metrics(
        gt,
        predictions,
        class_names=protocol.class_names,
        category_mapping=protocol.category_mapping,
        iou_thresholds=protocol.iou_thresholds,
        require_complete_taxonomy=complete_taxonomy,
    )
    platform = None
    score = None
    if complete_taxonomy:
        platform = build_platform_observed_metrics(
            ranking,
            recall_min=protocol.recall_min,
            fdr_max=protocol.fdr_max,
            latency_seconds=latency_seconds,
            latency_max_seconds=protocol.latency_max_seconds,
        )
        score = platform_confirmed_score(
            {
                name: {
                    "recall": platform.coarse_recall[name],
                    "fdr": platform.coarse_fdr[name],
                }
                for name in ("ship", "aircraft", "vehicle")
            },
            latency_seconds,
            recall_gate=protocol.recall_min,
            fdr_gate=protocol.fdr_max,
            latency_gate_seconds=protocol.latency_max_seconds or 20.0,
        )
    return (
        {
            "prediction_count": int(pooled.details["total_pred"]),
            "pooled": {
                "recall": pooled.recall,
                "fdr": pooled.fdr,
                "tp": int(pooled.details["tp"]),
                "fp": int(pooled.details["fp"]),
                "fn": int(pooled.details["fn"]),
            },
            "gate_recall": None if platform is None else platform.gate_recall,
            "gate_fdr": None if platform is None else platform.gate_fdr,
            "absolute_score_same_latency": (None if score is None else float(score["total_score"])),
            "per_coarse": {
                name: {
                    "macro_recall": (
                        None
                        if name not in ranking.per_coarse
                        else ranking.per_coarse[name].macro_recall
                    ),
                    "macro_fdr": (
                        None
                        if name not in ranking.per_coarse
                        else ranking.per_coarse[name].macro_fdr
                    ),
                    "pooled_recall": pooled.per_class[name].recall,
                    "pooled_fdr": pooled.per_class[name].fdr,
                    "tp": pooled.per_class[name].tp,
                    "fp": pooled.per_class[name].fp,
                    "fn": pooled.per_class[name].fn,
                }
                for name in ("ship", "aircraft", "vehicle")
            },
        },
        trace,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latency-seconds", type=float, default=0.0)
    parser.add_argument("--allow-incomplete-taxonomy", action="store_true")
    parser.add_argument("--project-config", type=Path, default=Path("configs/project.yaml"))
    args = parser.parse_args()

    gt = load_coco_ground_truth(args.ground_truth)
    protocol = parse_evaluation_protocol(load_config(args.project_config))
    metrics: dict[str, dict[str, Any]] = {}
    traces = {}
    for variant in VARIANTS:
        predictions = load_coco_predictions(args.replay_dir / f"{variant}_predictions.json")
        metrics[variant], traces[variant] = _evaluate(
            gt,
            predictions,
            protocol,
            complete_taxonomy=not args.allow_incomplete_taxonomy,
            latency_seconds=args.latency_seconds,
        )

    baseline = metrics["h0_legacy"]
    baseline_matches, baseline_fp = _trace_keys(traces["h0_legacy"])
    comparisons = {}
    for variant in VARIANTS[1:]:
        candidate = metrics[variant]
        candidate_matches, candidate_fp = _trace_keys(traces[variant])
        recovered = Counter(key[1] for key in candidate_matches - baseline_matches)
        lost = Counter(key[1] for key in baseline_matches - candidate_matches)
        per_coarse_delta = {
            name: {
                "macro_recall": (
                    None
                    if candidate["per_coarse"][name]["macro_recall"] is None
                    or baseline["per_coarse"][name]["macro_recall"] is None
                    else candidate["per_coarse"][name]["macro_recall"]
                    - baseline["per_coarse"][name]["macro_recall"]
                ),
                "macro_fdr": (
                    None
                    if candidate["per_coarse"][name]["macro_fdr"] is None
                    or baseline["per_coarse"][name]["macro_fdr"] is None
                    else candidate["per_coarse"][name]["macro_fdr"]
                    - baseline["per_coarse"][name]["macro_fdr"]
                ),
                "tp": candidate["per_coarse"][name]["tp"] - baseline["per_coarse"][name]["tp"],
                "fp": candidate["per_coarse"][name]["fp"] - baseline["per_coarse"][name]["fp"],
                "fn": candidate["per_coarse"][name]["fn"] - baseline["per_coarse"][name]["fn"],
                "recovered_gt": recovered[name],
                "lost_gt": lost[name],
                "unmatched_prediction_delta": candidate_fp[name] - baseline_fp[name],
            }
            for name in ("ship", "aircraft", "vehicle")
        }
        comparisons[variant] = {
            "delta_gate_recall": (
                None
                if candidate["gate_recall"] is None or baseline["gate_recall"] is None
                else candidate["gate_recall"] - baseline["gate_recall"]
            ),
            "delta_gate_fdr": (
                None
                if candidate["gate_fdr"] is None or baseline["gate_fdr"] is None
                else candidate["gate_fdr"] - baseline["gate_fdr"]
            ),
            "delta_absolute_score_same_latency": (
                None
                if candidate["absolute_score_same_latency"] is None
                or baseline["absolute_score_same_latency"] is None
                else candidate["absolute_score_same_latency"]
                - baseline["absolute_score_same_latency"]
            ),
            "per_coarse": per_coarse_delta,
            "aircraft_exact_metric_parity": all(
                per_coarse_delta["aircraft"][key] is not None
                and abs(per_coarse_delta["aircraft"][key]) <= 1e-12
                for key in ("macro_recall", "macro_fdr", "tp", "fp", "fn")
            ),
            "max_coarse_recall_drop_pp": 100.0
            * max(
                0.0,
                max(
                    -row["macro_recall"]
                    for row in per_coarse_delta.values()
                    if row["macro_recall"] is not None
                ),
            ),
        }

    summary = json.loads((args.replay_dir / "summary.json").read_text(encoding="utf-8"))
    payload = {
        "status": "complete",
        "role": "batis_paired_replay_diagnostic_not_formal_admission",
        "metric_protocol": "platform_observed_20260831",
        "complete_taxonomy": not args.allow_incomplete_taxonomy,
        "same_latency_seconds": args.latency_seconds,
        "mechanism_audits": summary["variant_audits"],
        "metrics": metrics,
        "comparisons_to_h0": comparisons,
        "h1a_h1b_exact_prediction_parity": summary["h1a_h1b_exact_prediction_parity"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
