#!/usr/bin/env python3
"""Triage a paired detector candidate without turning label oracles into policy.

The fixed-threshold comparison is the primary result.  Single-split oracle
frontiers are used only to distinguish calibration drift from a genuine loss
of ranking capacity; their thresholds must never be copied into deployment.
"""

# ruff: noqa: E402, I001

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsdet.evaluation.absolute_score import fdr_points, recall_points


COARSE_CLASSES = ("ship", "aircraft", "vehicle")


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _quality(fixed: dict[str, Any]) -> float:
    platform = fixed["platform"]
    return sum(
        recall_points(float(platform["per_coarse"][coarse]["macro_recall"]))
        + fdr_points(float(platform["per_coarse"][coarse]["macro_fdr"]))
        for coarse in COARSE_CLASSES
    ) / 7.0


def _oracle_quality(frontier: dict[str, Any]) -> float:
    return float(frontier["quality_oracle"]["selection_quality_score"])


def triage(
    baseline_fixed: dict[str, Any],
    candidate_fixed: dict[str, Any],
    baseline_frontier: dict[str, Any],
    candidate_frontier: dict[str, Any],
    *,
    positive_deadband: float,
) -> dict[str, Any]:
    baseline_threshold = float(baseline_fixed["threshold"])
    candidate_threshold = float(candidate_fixed["threshold"])
    if baseline_threshold != candidate_threshold:
        raise ValueError("paired fixed comparisons must use one identical threshold")
    baseline_gt = baseline_fixed["input_sha256"]["gt"]
    candidate_gt = candidate_fixed["input_sha256"]["gt"]
    if baseline_gt != candidate_gt:
        raise ValueError("baseline and candidate GT SHA differ")

    fixed_delta = _quality(candidate_fixed) - _quality(baseline_fixed)
    oracle_delta = _oracle_quality(candidate_frontier) - _oracle_quality(baseline_frontier)
    per_coarse: dict[str, Any] = {}
    for coarse in COARSE_CLASSES:
        base = baseline_fixed["platform"]["per_coarse"][coarse]
        cand = candidate_fixed["platform"]["per_coarse"][coarse]
        per_coarse[coarse] = {
            "delta_macro_recall_pp": 100.0
            * (float(cand["macro_recall"]) - float(base["macro_recall"])),
            "delta_macro_fdr_pp": 100.0
            * (float(cand["macro_fdr"]) - float(base["macro_fdr"])),
        }

    if fixed_delta <= 0.0 and oracle_delta <= 0.0:
        next_action = "stop_ranking_and_fixed_workpoint_both_worse"
    elif fixed_delta <= 0.0 < oracle_delta:
        next_action = "calibration_diagnostic_only_require_crossfit_before_any_policy"
    elif fixed_delta >= positive_deadband and oracle_delta > 0.0:
        next_action = "candidate_for_frozen_hard_and_sentinel_confirmation"
    else:
        next_action = "mixed_or_deadband_stop_unless_independent_evidence_exists"

    return {
        "status": "complete",
        "schema_version": "paired_detector_candidate_triage_v1",
        "fixed_threshold": baseline_threshold,
        "gt_sha256": baseline_gt,
        "fixed_quality": {
            "baseline": _quality(baseline_fixed),
            "candidate": _quality(candidate_fixed),
            "delta": fixed_delta,
        },
        "single_split_oracle_diagnostic": {
            "warning": "label oracle is diagnostic only and cannot select deployment thresholds",
            "baseline": _oracle_quality(baseline_frontier),
            "candidate": _oracle_quality(candidate_frontier),
            "delta": oracle_delta,
        },
        "per_coarse_fixed_delta": per_coarse,
        "positive_deadband": positive_deadband,
        "next_action": next_action,
        "automatic_full_or_submission_admission": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-fixed", type=Path, required=True)
    parser.add_argument("--candidate-fixed", type=Path, required=True)
    parser.add_argument("--baseline-frontier", type=Path, required=True)
    parser.add_argument("--candidate-frontier", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--positive-deadband", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = triage(
        _read(args.baseline_fixed),
        _read(args.candidate_fixed),
        _read(args.baseline_frontier),
        _read(args.candidate_frontier),
        positive_deadband=args.positive_deadband,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
