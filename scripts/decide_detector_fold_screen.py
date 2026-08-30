#!/usr/bin/env python3
"""Apply the frozen HERA-Guard V4 gate to held-out detector frontiers.

This is a *screening* decision only.  Both frontiers use the same held-out
labels and therefore cannot select a deployment threshold or replace CV3.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

COARSE = ("ship", "aircraft", "vehicle")
TARGET_COARSE = ("ship", "vehicle")


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete_diagnostic_only":
        raise ValueError(f"incomplete frontier: {path}")
    return payload


def decide(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    floor_gain: float = 0.01,
    fdr15_gain: float = 0.005,
    max_coarse_drop: float = 0.005,
) -> dict[str, Any]:
    baseline_gt = baseline["input_sha256"]["gt"]
    candidate_gt = candidate["input_sha256"]["gt"]
    if baseline_gt != candidate_gt:
        raise ValueError("baseline and candidate GT SHA256 differ")

    b_floor = baseline["score_floor_metrics"]
    c_floor = candidate["score_floor_metrics"]
    b_fdr15 = baseline["frontiers"]["0.150"]
    c_fdr15 = candidate["frontiers"]["0.150"]

    floor_delta = {
        coarse: float(c_floor["per_coarse"][coarse]["recall"])
        - float(b_floor["per_coarse"][coarse]["recall"])
        for coarse in COARSE
    }
    fdr15_delta = {
        coarse: float(c_fdr15["per_coarse"][coarse]["recall"])
        - float(b_fdr15["per_coarse"][coarse]["recall"])
        for coarse in COARSE
    }
    pooled_delta = float(c_fdr15["recall"]) - float(b_fdr15["recall"])

    floor_gate = max(floor_delta[name] for name in TARGET_COARSE) >= floor_gain
    frontier_gate = pooled_delta >= fdr15_gain
    protection_gate = min(fdr15_delta.values()) >= -max_coarse_drop
    admitted = floor_gate and frontier_gate and protection_gate

    return {
        "status": "screen_admitted_for_cv3" if admitted else "screen_rejected",
        "scope": "single_heldout_fold_diagnostic_only",
        "warning": (
            "Held-out labels were used to compute both frontiers. Admission only "
            "authorizes CV3; it does not authorize deployment or a threshold."
        ),
        "gt_sha256": baseline_gt,
        "frozen_gate": {
            "ship_or_vehicle_score_floor_recall_gain_min": floor_gain,
            "pooled_fdr15_recall_gain_min": fdr15_gain,
            "max_any_coarse_fdr15_recall_drop": max_coarse_drop,
        },
        "observed": {
            "score_floor_recall_delta": floor_delta,
            "pooled_fdr15_recall_delta": pooled_delta,
            "fdr15_recall_delta": fdr15_delta,
            "candidate_fdr15": float(c_fdr15["fdr"]),
        },
        "gates": {
            "ship_or_vehicle_floor_gain": floor_gate,
            "pooled_fdr15_gain": frontier_gate,
            "coarse_recall_protection": protection_gate,
        },
        "next_action": (
            "run_source_grouped_cv3_without_reusing_this_fold_threshold"
            if admitted
            else "stop_detector_route"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--floor-gain", type=float, default=0.01)
    parser.add_argument("--fdr15-gain", type=float, default=0.005)
    parser.add_argument("--max-coarse-drop", type=float, default=0.005)
    args = parser.parse_args()

    result = decide(
        _load(args.baseline),
        _load(args.candidate),
        floor_gain=args.floor_gain,
        fdr15_gain=args.fdr15_gain,
        max_coarse_drop=args.max_coarse_drop,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
