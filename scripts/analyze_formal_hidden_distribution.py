#!/usr/bin/env python3
"""Infer the identifiable part of the formal hidden-set distribution.

This tool deliberately separates facts from priors.  It recovers exact
coarse-class GT/prediction counts from the platform response, compares them
with the official training set, and emits a train-proportional fine-class
allocation only as a scenario for local stress testing.  It never presents
that allocation as recovered hidden ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.absolute_score import platform_confirmed_score


def _largest_remainder(weights: dict[str, int], total: int) -> dict[str, int]:
    weight_sum = sum(weights.values())
    if weight_sum <= 0 or total < 0:
        raise ValueError("weights must be positive and total non-negative")
    quotas = {key: total * value / weight_sum for key, value in weights.items()}
    result = {key: math.floor(value) for key, value in quotas.items()}
    remaining = total - sum(result.values())
    order = sorted(weights, key=lambda key: (-(quotas[key] - result[key]), key))
    for key in order[:remaining]:
        result[key] += 1
    return result


def _read_train_stats(path: Path) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    coarse: dict[str, int] = {}
    fine: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            count = int(row["instance_count"])
            if row["level"] == "macro":
                coarse[row["macro_category"]] = count
            elif row["level"] == "fine":
                fine[row["macro_category"]][row["category_name"]] = count
    required = {"ship", "aircraft", "vehicle"}
    if set(coarse) != required or set(fine) != required:
        raise ValueError(f"training statistics must contain exactly {sorted(required)}")
    return coarse, dict(fine)


def analyze(anchor: dict[str, Any], train_stats_path: Path) -> dict[str, Any]:
    train_coarse, train_fine = _read_train_stats(train_stats_path)
    per_coarse = anchor["per_coarse"]
    official_gt = {name: int(row["tp"]) + int(row["fn"]) for name, row in per_coarse.items()}
    train_total = sum(train_coarse.values())
    official_total = sum(official_gt.values())
    coarse_comparison: dict[str, Any] = {}
    fine_prior_scenarios: dict[str, Any] = {}
    for coarse in ("ship", "aircraft", "vehicle"):
        row = per_coarse[coarse]
        tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])
        pooled_recall = tp / (tp + fn) if tp + fn else 0.0
        pooled_fdr = fp / (tp + fp) if tp + fp else 0.0
        train_share = train_coarse[coarse] / train_total
        official_share = official_gt[coarse] / official_total
        coarse_comparison[coarse] = {
            "train_gt": train_coarse[coarse],
            "formal_gt_exact": official_gt[coarse],
            "train_share": train_share,
            "formal_share": official_share,
            "formal_to_train_share_ratio": official_share / train_share,
            "formal_to_train_count_ratio": official_gt[coarse] / train_coarse[coarse],
            "displayed_macro_recall": float(row["recall"]),
            "pooled_recall_from_counts": pooled_recall,
            "macro_minus_pooled_recall": float(row["recall"]) - pooled_recall,
            "displayed_macro_fdr": float(row["fdr"]),
            "pooled_fdr_from_counts": pooled_fdr,
            "macro_minus_pooled_fdr": float(row["fdr"]) - pooled_fdr,
        }
        fine_prior_scenarios[coarse] = {
            "status": "prior_not_hidden_truth",
            "method": "largest_remainder_train_proportional",
            "formal_gt_total": official_gt[coarse],
            "allocation": _largest_remainder(train_fine[coarse], official_gt[coarse]),
        }

    score = platform_confirmed_score(per_coarse, float(anchor["latency_seconds"]))
    return {
        "status": "complete",
        "schema_version": "formal_hidden_distribution_inference_v1",
        "anchor_schema_version": anchor["schema_version"],
        "formal_total_gt_exact": official_total,
        "formal_total_predictions_exact": sum(
            int(row["tp"]) + int(row["fp"]) for row in per_coarse.values()
        ),
        "coarse_comparison": coarse_comparison,
        "fine_prior_scenarios": fine_prior_scenarios,
        "platform_score_reproduction": score,
        "identifiability": {
            "exactly_identified": [
                "coarse GT totals and proportions",
                "coarse pooled TP/FP/FN",
                "displayed coarse fine-macro Recall/FDR",
                "three-coarse hard-gate aggregation",
                "seven-subscore total-score aggregation",
            ],
            "not_uniquely_identified": [
                "25 fine-class GT counts",
                "per-fine TP/FP/FN for ship and aircraft",
                "number of test images and objects per image",
                "source/domain/airport/scene distribution",
                "object-size, resolution, and background-negative distributions",
            ],
            "reason": (
                "For ship and aircraft, pooled totals plus two macro rates leave many more "
                "unknown per-fine counts than equations. The train-proportional allocation "
                "is therefore a benchmark prior, not a reconstruction."
            ),
        },
        "recommended_proxy_contract": {
            "primary": "official_anchor_calibrated_group_bootstrap_v1",
            "regression_guard": "source_disjoint_sentinel_frozen_v1",
            "exact_coarse_gt_budget": official_gt,
            "aggregation": "platform_confirmed_absolute_score_v2026_08_31",
            "fine_mix_scenarios": [
                "train_proportional",
                "fine_balanced",
                "rare_class_upweighted",
            ],
            "sampling_unit": "source/group then image; never independent boxes",
            "calibration_policy": (
                "Use the incumbent formal result once to freeze proxy difficulty; choose "
                "future thresholds on nested development folds and evaluate on untouched "
                "confirmation groups."
            ),
            "uncertainty": "report median, 10th percentile, and gate-pass rate over >=200 group bootstraps",
            "admission_margin": {
                "macro_coarse_recall_min": 0.87,
                "macro_coarse_fdr_max": 0.18,
                "latency_seconds_max": 15.0,
                "minimum_bootstrap_gate_pass_rate": 0.90,
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--train-stats", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    anchor = json.loads(args.anchor.read_text(encoding="utf-8"))
    payload = analyze(anchor, args.train_stats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
