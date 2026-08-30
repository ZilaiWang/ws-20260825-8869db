#!/usr/bin/env python3
"""Compile frozen HERA-Guard V3 stage decisions from official frontiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _point(payload: dict[str, Any], level: float) -> dict[str, Any]:
    return payload["frontiers"][f"{level:.3f}"]["crossfit"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--stage", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fdr-level", type=float, default=0.15)
    parser.add_argument("--minimum-recall-gain", type=float, default=0.005)
    parser.add_argument("--maximum-coarse-recall-drop", type=float, default=0.005)
    args = parser.parse_args()
    baseline_payload = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = _point(baseline_payload, args.fdr_level)
    rows = []
    for spec in args.stage:
        if "=" not in spec:
            raise ValueError("--stage must use NAME=PATH")
        name, raw_path = spec.split("=", 1)
        path = Path(raw_path)
        point = _point(json.loads(path.read_text(encoding="utf-8")), args.fdr_level)
        coarse_deltas = {
            coarse: float(point["per_coarse"][coarse]["recall"])
            - float(baseline["per_coarse"][coarse]["recall"])
            for coarse in baseline["per_coarse"]
        }
        recall_gain = float(point["recall"]) - float(baseline["recall"])
        passed = bool(
            point["fdr"] <= args.fdr_level
            and recall_gain >= args.minimum_recall_gain
            and min(coarse_deltas.values()) >= -args.maximum_coarse_recall_drop
        )
        rows.append(
            {
                "stage": name,
                "recall": float(point["recall"]),
                "fdr": float(point["fdr"]),
                "tp": int(point["tp"]),
                "fp": int(point["fp"]),
                "recall_gain": recall_gain,
                "tp_gain": int(point["tp"]) - int(baseline["tp"]),
                "fp_delta": int(point["fp"]) - int(baseline["fp"]),
                "coarse_recall_delta": coarse_deltas,
                "formal_stage_gate": passed,
                "frontier": str(path),
            }
        )
    admitted = [row for row in rows if row["formal_stage_gate"]]
    best = max(admitted, key=lambda row: (row["recall"], -row["fdr"])) if admitted else None
    result = {
        "status": "metric_risk_stage_comparison_complete",
        "fdr_level": args.fdr_level,
        "baseline": {
            "recall": float(baseline["recall"]),
            "fdr": float(baseline["fdr"]),
            "tp": int(baseline["tp"]),
            "fp": int(baseline["fp"]),
        },
        "gate": {
            "minimum_recall_gain": args.minimum_recall_gain,
            "maximum_coarse_recall_drop": args.maximum_coarse_recall_drop,
        },
        "stages": rows,
        "admitted_stage": None if best is None else best["stage"],
        "next_action": (
            "admit_metric_risk_and_start_dual_view"
            if best is not None
            else "stop_tabular_metric_risk_and_require_new_pixel_evidence"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
