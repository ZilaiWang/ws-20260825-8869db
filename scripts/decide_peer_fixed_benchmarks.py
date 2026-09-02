#!/usr/bin/env python3
"""Gate a peer detector on fixed Hard and Sentinel-B one-fold screens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COARSE = ("ship", "aircraft", "vehicle")
PLATFORM_OBSERVED_PROTOCOL = "platform_observed_20260831"


def _require_platform(payload: dict[str, Any]) -> dict[str, Any]:
    platform = payload.get("platform")
    if not isinstance(platform, dict) or platform.get(
        "metric_protocol"
    ) != PLATFORM_OBSERVED_PROTOCOL:
        raise ValueError(
            f"{PLATFORM_OBSERVED_PROTOCOL} metrics are required; legacy "
            "pooled/25-fine payloads cannot authorize this gate"
        )
    return platform


def _delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    candidate_platform = _require_platform(candidate)
    baseline_platform = _require_platform(baseline)
    return {
        "platform_gate_recall": float(candidate_platform["gate_recall"])
        - float(baseline_platform["gate_recall"]),
        "platform_gate_fdr": float(candidate_platform["gate_fdr"])
        - float(baseline_platform["gate_fdr"]),
        "per_coarse_macro_recall": {
            name: float(candidate_platform["per_coarse"][name]["macro_recall"])
            - float(baseline_platform["per_coarse"][name]["macro_recall"])
            for name in COARSE
        },
        "per_coarse_macro_fdr": {
            name: float(candidate_platform["per_coarse"][name]["macro_fdr"])
            - float(baseline_platform["per_coarse"][name]["macro_fdr"])
            for name in COARSE
        },
        "diagnostic_pooled_recall": float(candidate["pooled_recall"])
        - float(baseline["pooled_recall"]),
        "diagnostic_pooled_fdr": float(candidate["pooled_fdr"])
        - float(baseline["pooled_fdr"]),
        "diagnostic_fine25_macro_recall": float(candidate["fine25_macro_recall"])
        - float(baseline["fine25_macro_recall"]),
    }


def decide(
    hard_baseline: dict[str, Any],
    hard_candidate: dict[str, Any],
    sentinel_baseline: dict[str, Any],
    sentinel_candidate: dict[str, Any],
    *,
    minimum_hard_recall_gain: float = 0.005,
    maximum_coarse_drop: float = 0.005,
) -> dict[str, Any]:
    if hard_baseline["input_sha256"]["gt"] != hard_candidate["input_sha256"]["gt"]:
        raise ValueError("Hard baseline and candidate use different ground truth")
    if sentinel_baseline["input_sha256"]["gt"] != sentinel_candidate["input_sha256"]["gt"]:
        raise ValueError("Sentinel baseline and candidate use different ground truth")
    hard = _delta(
        hard_candidate["frontiers"]["0.150"],
        hard_baseline["frontiers"]["0.150"],
    )
    sentinel = _delta(sentinel_candidate, sentinel_baseline)
    gates = {
        "hard_platform_recall_gain": hard["platform_gate_recall"]
        >= minimum_hard_recall_gain,
        "hard_platform_fdr_nondegrade": hard["platform_gate_fdr"] <= 0.0,
        "hard_coarse_macro_recall_protection": min(
            hard["per_coarse_macro_recall"].values()
        )
        >= -maximum_coarse_drop,
        "sentinel_platform_recall_nondegrade": sentinel["platform_gate_recall"]
        >= 0.0,
        "sentinel_platform_fdr_nondegrade": sentinel["platform_gate_fdr"] <= 0.0,
        "sentinel_coarse_macro_recall_protection": min(
            sentinel["per_coarse_macro_recall"].values()
        )
        >= -maximum_coarse_drop,
    }
    admitted = all(gates.values())
    return {
        "status": "screen_admitted" if admitted else "screen_rejected",
        "scope": "fold0_fixed_hard_and_sentinel_b_platform_observed_diagnostic_only",
        "warning": (
            "This one-fold screen can authorize a source-grouped CV3 replay only; "
            "it cannot authorize deployment or threshold selection."
        ),
        "research_only_unlicensed_reference": True,
        "frozen_gate": {
            "minimum_hard_platform_recall_gain": minimum_hard_recall_gain,
            "maximum_any_coarse_macro_recall_drop": maximum_coarse_drop,
            "maximum_platform_fdr_delta": 0.0,
            "sentinel_threshold_source": (
                "each model's own Hard platform-observed FDR15 frontier"
            ),
        },
        "observed": {"hard_fdr15_delta": hard, "sentinel_fixed_threshold_delta": sentinel},
        "gates": gates,
        "next_action": (
            "expand_exact_module_to_source_grouped_cv3_then_model_l"
            if admitted
            else "stop_peer_hcl_route_without_scale_or_parameter_scan"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-baseline", type=Path, required=True)
    parser.add_argument("--hard-candidate", type=Path, required=True)
    parser.add_argument("--sentinel-baseline", type=Path, required=True)
    parser.add_argument("--sentinel-candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = decide(
        json.loads(args.hard_baseline.read_text(encoding="utf-8")),
        json.loads(args.hard_candidate.read_text(encoding="utf-8")),
        json.loads(args.sentinel_baseline.read_text(encoding="utf-8")),
        json.loads(args.sentinel_candidate.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
