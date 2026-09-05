#!/usr/bin/env python3
"""One paired score report; no model training, threshold fitting or admission."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.absolute_score import COARSE_CLASSES, platform_confirmed_score
from rsdet.evaluation.platform_protocol import PLATFORM_OBSERVED_PROTOCOL


def _scores(payload: dict[str, Any]) -> dict[str, Any]:
    platform = payload["platform"]
    if platform.get("metric_protocol") != PLATFORM_OBSERVED_PROTOCOL:
        raise ValueError("paired trend requires platform_observed_20260831")
    rates = {
        name: {"recall": row["macro_recall"], "fdr": row["macro_fdr"]}
        for name, row in platform["per_coarse"].items()
    }
    latency = payload.get("latency_seconds")
    if latency is None and platform.get("score_payload") is not None:
        latency = platform["score_payload"].get("latency_seconds")
    # A zero-time calculation only recovers the six quality contributions.
    # The total is deliberately omitted if latency was not measured.
    score = platform_confirmed_score(rates, 0.0 if latency is None else latency)
    quality = sum(score["seven_subscores"][:6]) / 7.0
    return {
        "quality_contribution_out_of_600_over_7": quality,
        "total_score": None if latency is None else score["total_score"],
        "latency_seconds": latency,
        "per_coarse": score["per_coarse"],
        "gate_recall": score["hard_gates"]["macro_coarse_recall"],
        "gate_fdr": score["hard_gates"]["macro_coarse_fdr"],
    }


def compare(
    baseline: dict[str, Any], candidate: dict[str, Any], *, deadband: float = 0.5
) -> dict[str, Any]:
    """Compare identical GT; trade-offs are shown, not vetoed one metric at a time.

    This is a report, NOT a provenance verifier or statistical test.  A positive
    sign does not establish equal training budgets, an untouched holdout, a
    compatible timing environment, or permission to train/submit a full model.
    """
    if not math.isfinite(deadband) or deadband < 0:
        raise ValueError("deadband must be finite and non-negative")
    gt_sha = baseline.get("input_sha256", {}).get("gt")
    if not gt_sha or gt_sha != candidate.get("input_sha256", {}).get("gt"):
        raise ValueError("baseline and candidate must have the same GT SHA")
    b, c = _scores(baseline), _scores(candidate)
    quality_delta = (
        c["quality_contribution_out_of_600_over_7"]
        - b["quality_contribution_out_of_600_over_7"]
    )
    total_delta = (
        None if b["total_score"] is None or c["total_score"] is None
        else c["total_score"] - b["total_score"]
    )
    observed = quality_delta if total_delta is None else total_delta
    direction = "positive" if observed > deadband else (
        "negative" if observed < -deadband else "small_or_uncertain"
    )
    warnings = []
    if total_delta is None:
        warnings.append("Timing missing: quality-only trend, not a total-score comparison.")
    if any(x.get("diagnostic_only") or x.get("data_leakage") for x in (baseline, candidate)):
        warnings.append("Seen-source/exploratory evidence cannot authorize deployment.")
    if any(not x.get("image_coverage", {}).get("negative_coverage_known")
           for x in (baseline, candidate)):
        warnings.append("Negative-image coverage was not verified by the input report.")
    return {
        "protocol": "paired_trend_review_v1",
        "metric_protocol": PLATFORM_OBSERVED_PROTOCOL,
        "gt_sha256": gt_sha,
        "baseline": b,
        "candidate": c,
        "delta_quality_contribution": quality_delta,
        "delta_total_score": total_delta,
        "direction": direction,
        "deadband_score_points": deadband,
        "deadband_is_not_a_confidence_interval": True,
        "per_coarse_delta": {
            name: {
                "recall_pp": 100 * (c["per_coarse"][name]["recall"]
                                    - b["per_coarse"][name]["recall"]),
                "fdr_pp": 100 * (c["per_coarse"][name]["fdr"]
                                 - b["per_coarse"][name]["fdr"]),
                "score_contribution": sum(
                    c["per_coarse"][name][key] - b["per_coarse"][name][key]
                    for key in ("recall_points", "fdr_points")
                ) / 7.0,
            }
            for name in COARSE_CLASSES
        },
        "warnings": warnings,
        "training_and_timing_comparability_verified": False,
        "automatic_full_training_or_submission_admission": False,
        "official_score_forecast": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    result = compare(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
