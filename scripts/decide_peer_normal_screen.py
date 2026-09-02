#!/usr/bin/env python3
"""Compare paired detector frontiers under the observed platform aggregation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

COARSE = ("ship", "aircraft", "vehicle")
PLATFORM_OBSERVED_PROTOCOL = "platform_observed_20260831"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_platform(point: dict[str, Any]) -> dict[str, Any]:
    platform = point.get("platform")
    if not isinstance(platform, dict) or platform.get(
        "metric_protocol"
    ) != PLATFORM_OBSERVED_PROTOCOL:
        raise ValueError(
            f"{PLATFORM_OBSERVED_PROTOCOL} metrics are required; a legacy "
            "pooled/25-fine frontier cannot authorize expansion"
        )
    return platform


def _row(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_platform = _require_platform(baseline)
    cand_platform = _require_platform(candidate)
    return {
        "baseline_platform_gate_recall": base_platform["gate_recall"],
        "candidate_platform_gate_recall": cand_platform["gate_recall"],
        "delta_platform_gate_recall": (
            float(cand_platform["gate_recall"])
            - float(base_platform["gate_recall"])
        ),
        "baseline_platform_gate_fdr": base_platform["gate_fdr"],
        "candidate_platform_gate_fdr": cand_platform["gate_fdr"],
        "delta_platform_gate_fdr": (
            float(cand_platform["gate_fdr"])
            - float(base_platform["gate_fdr"])
        ),
        "per_coarse_macro_recall_delta": {
            name: (
                float(cand_platform["per_coarse"][name]["macro_recall"])
                - float(base_platform["per_coarse"][name]["macro_recall"])
            )
            for name in COARSE
        },
        "diagnostic_pooled_recall_delta": (
            float(candidate["pooled_recall"]) - float(baseline["pooled_recall"])
        ),
        "diagnostic_fine25_macro_recall_delta": (
            float(candidate["fine25_macro_recall"])
            - float(baseline["fine25_macro_recall"])
        ),
    }


def decide(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    minimum_recall_gain: float = 0.005,
    maximum_coarse_drop: float = 0.005,
) -> dict[str, Any]:
    if baseline["input_sha256"]["gt"] != candidate["input_sha256"]["gt"]:
        raise ValueError("baseline and candidate use different ground truth")
    levels = sorted(set(baseline["frontiers"]) & set(candidate["frontiers"]))
    if "0.150" not in levels:
        raise ValueError("paired frontiers must contain the frozen 0.150 level")
    rows = {
        level: _row(baseline["frontiers"][level], candidate["frontiers"][level])
        for level in levels
    }
    target = rows["0.150"]
    gates = {
        "platform_recall_gain_ge_0p5pp": (
            target["delta_platform_gate_recall"] >= minimum_recall_gain
        ),
        "platform_fdr_nondegrade": target["delta_platform_gate_fdr"] <= 0.0,
        "coarse_macro_recall_floor_ge_minus_0p5pp": min(
            target["per_coarse_macro_recall_delta"].values()
        )
        >= -maximum_coarse_drop,
    }
    admitted = all(gates.values())
    return {
        "status": (
            "complete_platform_screen_admitted"
            if admitted
            else "complete_platform_screen_rejected"
        ),
        "protocol": "paired_single_fold_platform_observed_screen_v2",
        "scope": "diagnostic_only_not_deployment_admission",
        "frontiers": rows,
        "frozen_gate": {
            "target_fdr": 0.15,
            "minimum_platform_recall_gain": minimum_recall_gain,
            "maximum_platform_fdr_delta": 0.0,
            "maximum_any_coarse_macro_recall_drop": maximum_coarse_drop,
        },
        "gates": gates,
        "next_action": (
            "run_fixed_hard_sentinel_tiled_screen"
            if admitted
            else "stop_without_scale_or_cv3_expansion"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--research-only-unlicensed-reference", action="store_true")
    args = parser.parse_args()
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    result = decide(baseline, candidate)
    result["research_only_unlicensed_reference"] = bool(
        args.research_only_unlicensed_reference
    )
    result["sha256"] = {
        "baseline_frontier": _sha256(args.baseline),
        "candidate_frontier": _sha256(args.candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
