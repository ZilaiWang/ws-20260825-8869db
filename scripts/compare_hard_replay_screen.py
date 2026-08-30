#!/usr/bin/env python3
"""Compare one-fold hard-replay replacement against the frozen CV3 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_frontiers(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    base_floor = baseline["candidate_floor"]
    cand_floor = candidate["candidate_floor"]
    base = baseline["frontiers"]["0.150"]["crossfit"]
    cand = candidate["frontiers"]["0.150"]["crossfit"]
    coarse_drops = {
        name: float(base["per_coarse"][name]["recall"])
        - float(cand["per_coarse"][name]["recall"])
        for name in ("ship", "aircraft", "vehicle")
    }
    delta_recall = float(cand["recall"]) - float(base["recall"])
    delta_fdr = float(cand["fdr"]) - float(base["fdr"])
    floor_delta = float(cand_floor["recall"]) - float(base_floor["recall"])
    recall_path = delta_recall >= 0.002 and delta_fdr <= 0.003
    precision_path = delta_fdr <= -0.005 and delta_recall >= -0.002
    passed = (
        floor_delta >= -0.003
        and max(coarse_drops.values()) <= 0.005
        and (recall_path or precision_path)
    )
    return {
        "status": "complete",
        "protocol": "one_fold_replacement_hard_replay_screen_v1",
        "candidate_floor_recall_delta": floor_delta,
        "recall_delta_at_fdr_0p15": delta_recall,
        "fdr_delta_at_fdr_0p15": delta_fdr,
        "coarse_recall_drops": coarse_drops,
        "recall_path_passed": recall_path,
        "precision_path_passed": precision_path,
        "screen_passed": passed,
        "next_action": (
            "expand_to_three_folds_then_run_both_fixed_benchmarks"
            if passed
            else "stop_hard_replay_no_parameter_scan"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare_frontiers(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.candidate.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
