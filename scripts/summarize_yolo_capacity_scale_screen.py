#!/usr/bin/env python3
"""Summarize the frozen 2x2 YOLO capacity/scale fold0 screen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.decide_detector_fold_screen import decide

KEYS = ("s1024", "s1280", "m1024", "m1280")
PAIRS = {
    "scale_at_s": ("s1024", "s1280"),
    "capacity_at_1024": ("s1024", "m1024"),
    "capacity_at_1280": ("s1280", "m1280"),
    "scale_at_m": ("m1024", "m1280"),
}


def _point(frontier: dict[str, Any]) -> dict[str, Any]:
    point = frontier["frontiers"]["0.150"]
    platform = point.get("platform")
    if not isinstance(platform, dict):
        raise ValueError("platform_observed frontier is required")
    return {
        "threshold": float(point["threshold"]),
        "gate_recall": float(platform["gate_recall"]),
        "gate_fdr": float(platform["gate_fdr"]),
        "per_coarse": platform["per_coarse"],
        "diagnostic_pooled_recall": float(point["pooled_recall"]),
        "diagnostic_pooled_fdr": float(point["pooled_fdr"]),
    }


def summarize(frontiers: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if set(frontiers) != set(KEYS):
        raise ValueError(f"exactly these conditions are required: {KEYS}")
    gt_hashes = {payload["input_sha256"]["gt"] for payload in frontiers.values()}
    if len(gt_hashes) != 1:
        raise ValueError("all four conditions must use identical ground truth")

    pairwise = {
        name: decide(frontiers[baseline], frontiers[candidate])
        for name, (baseline, candidate) in PAIRS.items()
    }
    primary = {
        key: decide(frontiers["s1024"], frontiers[key])
        for key in ("s1280", "m1024", "m1280")
    }
    admitted = [
        key
        for key, decision in primary.items()
        if decision["status"] == "screen_admitted_for_cv3"
    ]
    selected = None
    if admitted:
        selected = max(
            admitted,
            key=lambda key: (
                _point(frontiers[key])["gate_recall"],
                -_point(frontiers[key])["gate_fdr"],
            ),
        )
    return {
        "status": "complete_same_length_fold0_screen",
        "metric_protocol": "platform_observed_20260831",
        "scope": "same_fold_same_seed_same_epoch_exploration_only",
        "warning": (
            "This 2x2 screen isolates capacity and input scale but uses held-out "
            "labels for its frontier. It can authorize CV3 only, never deployment."
        ),
        "frozen_factors": {
            "fold": 0,
            "epochs": 40,
            "seed": 42,
            "total_batch": 8,
            "innovation": "Y5 RandomRotate90 p=1.0",
            "checkpoint": "last",
        },
        "conditions": {key: _point(frontiers[key]) for key in KEYS},
        "pairwise_factor_decisions": pairwise,
        "primary_vs_s1024": primary,
        "selected_for_cv3": selected,
        "next_action": (
            "run_selected_condition_source_grouped_cv3_against_matched_control"
            if selected is not None
            else "stop_capacity_scale_route"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in KEYS:
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frontiers = {
        key: json.loads(getattr(args, key).read_text(encoding="utf-8"))
        for key in KEYS
    }
    result = summarize(frontiers)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
