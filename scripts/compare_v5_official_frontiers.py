#!/usr/bin/env python3
"""Compare a V5 candidate with its frozen official-frontier baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _crossfit(payload: dict[str, Any], level: str) -> dict[str, Any]:
    try:
        return payload["frontiers"][level]["crossfit"]
    except KeyError as error:
        raise ValueError(f"frontier does not contain FDR level {level}") from error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--level", default="0.150")
    parser.add_argument("--minimum-recall-gain", type=float, default=0.005)
    parser.add_argument("--maximum-fdr-regression", type=float, default=0.005)
    parser.add_argument("--maximum-coarse-recall-drop", type=float, default=0.005)
    args = parser.parse_args()

    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate_payload = json.loads(args.candidate.read_text(encoding="utf-8"))
    baseline = _crossfit(baseline_payload, args.level)
    candidate = _crossfit(candidate_payload, args.level)
    coarse_names = ("ship", "aircraft", "vehicle")
    coarse_delta = {
        name: float(candidate["per_coarse"][name]["recall"])
        - float(baseline["per_coarse"][name]["recall"])
        for name in coarse_names
    }
    recall_delta = float(candidate["recall"]) - float(baseline["recall"])
    fdr_delta = float(candidate["fdr"]) - float(baseline["fdr"])
    gates = {
        "recall_gain": recall_delta >= args.minimum_recall_gain,
        "fdr_noninferiority": fdr_delta <= args.maximum_fdr_regression,
        "every_coarse_recall_noninferior": min(coarse_delta.values())
        >= -args.maximum_coarse_recall_drop,
    }
    result = {
        "status": "complete",
        "protocol": "v5_paired_official_frontier_admission_v1",
        "level": args.level,
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "recall": recall_delta,
            "fdr": fdr_delta,
            "macro_recall": float(candidate["macro_recall"])
            - float(baseline["macro_recall"]),
            "per_coarse_recall": coarse_delta,
        },
        "thresholds": {
            "minimum_recall_gain": args.minimum_recall_gain,
            "maximum_fdr_regression": args.maximum_fdr_regression,
            "maximum_coarse_recall_drop": args.maximum_coarse_recall_drop,
        },
        "gates": gates,
        "admitted": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
