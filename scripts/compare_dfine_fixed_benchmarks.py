#!/usr/bin/env python3
"""Compare frozen Y5 and Y5+D-FINE results on Hard10K and Sentinel.

This is deliberately a decision compiler rather than another tuner: all four
input metric files must already have been produced by the exact official
matching evaluator at deployment-frozen thresholds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"ship", "aircraft", "vehicle"}
    per_coarse = payload.get("official_ranking", {}).get("per_coarse", {})
    if set(per_coarse) != required:
        raise ValueError(f"{path}: official per_coarse taxonomy is not complete")
    return payload


def _condition(
    base_path: Path,
    dual_path: Path,
    *,
    rescored_coarse: set[str],
) -> dict[str, Any]:
    base = _load(base_path)
    dual = _load(dual_path)
    rows: dict[str, Any] = {}
    for coarse in ("ship", "aircraft", "vehicle"):
        before = base["official_ranking"]["per_coarse"][coarse]
        after = dual["official_ranking"]["per_coarse"][coarse]
        rows[coarse] = {
            "base_recall": float(before["macro_recall"]),
            "dual_recall": float(after["macro_recall"]),
            "delta_recall": float(after["macro_recall"] - before["macro_recall"]),
            "base_fdr": float(before["macro_fdr"]),
            "dual_fdr": float(after["macro_fdr"]),
            "delta_fdr": float(after["macro_fdr"] - before["macro_fdr"]),
        }
    protected_coarse = {"ship", "aircraft", "vehicle"} - rescored_coarse
    protected = all(
        abs(rows[name][field]) <= 1e-12
        for name in protected_coarse
        for field in ("delta_recall", "delta_fdr")
    )
    rescored_nondegrade = all(
        rows[name]["delta_recall"] >= -1e-12
        and rows[name]["delta_fdr"] <= 1e-12
        for name in rescored_coarse
    )
    return {
        "base": str(base_path),
        "dual": str(dual_path),
        "base_sha256": _sha256(base_path),
        "dual_sha256": _sha256(dual_path),
        "metrics": rows,
        "protected_coarse_exactly_unchanged": protected,
        "rescored_coarse_nondegrade": rescored_nondegrade,
        "base_latency_seconds": base.get("latency_seconds"),
        "dual_latency_seconds": dual.get("latency_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hard-base", type=Path, required=True)
    parser.add_argument("--hard-dual", type=Path, required=True)
    parser.add_argument("--sentinel-base", type=Path, required=True)
    parser.add_argument("--sentinel-dual", type=Path, required=True)
    parser.add_argument("--hard-airvehicle", type=Path)
    parser.add_argument("--sentinel-airvehicle", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if (args.hard_airvehicle is None) != (args.sentinel_airvehicle is None):
        raise ValueError("aircraft+vehicle metrics must be provided for both conditions")
    candidates: dict[str, dict[str, Any]] = {
        "vehicle_only": {
            "rescored_coarse": ["vehicle"],
            "conditions": {
                "hard10k": _condition(
                    args.hard_base, args.hard_dual, rescored_coarse={"vehicle"}
                ),
                "sentinel": _condition(
                    args.sentinel_base, args.sentinel_dual, rescored_coarse={"vehicle"}
                ),
            },
        }
    }
    if args.hard_airvehicle is not None and args.sentinel_airvehicle is not None:
        candidates["aircraft_vehicle"] = {
            "rescored_coarse": ["aircraft", "vehicle"],
            "conditions": {
                "hard10k": _condition(
                    args.hard_base,
                    args.hard_airvehicle,
                    rescored_coarse={"aircraft", "vehicle"},
                ),
                "sentinel": _condition(
                    args.sentinel_base,
                    args.sentinel_airvehicle,
                    rescored_coarse={"aircraft", "vehicle"},
                ),
            },
        }
    admitted_routes: list[str] = []
    for name, candidate in candidates.items():
        conditions = candidate["conditions"]
        both_safe = all(
            item["protected_coarse_exactly_unchanged"]
            and item["rescored_coarse_nondegrade"]
            for item in conditions.values()
        )
        meaningful_gain = any(
            item["metrics"][coarse]["delta_recall"] >= 0.005
            or item["metrics"][coarse]["delta_fdr"] <= -0.005
            for item in conditions.values()
            for coarse in candidate["rescored_coarse"]
        )
        admitted = both_safe and meaningful_gain
        candidate["decision"] = {
            "both_conditions_nondegrading": both_safe,
            "meaningful_gain": meaningful_gain,
            "formal_submission_admission": admitted,
        }
        if admitted:
            admitted_routes.append(name)
    selected = (
        "aircraft_vehicle"
        if "aircraft_vehicle" in admitted_routes
        else ("vehicle_only" if "vehicle_only" in admitted_routes else None)
    )
    payload = {
        "status": "complete",
        "protocol": "dfine_full_fixed_hard10k_sentinel_decision_v1",
        "threshold_tuning_on_benchmarks": False,
        "candidates": candidates,
        "decision": {
            "formal_submission_admission": selected is not None,
            "selected_route": selected,
            "reason": (
                "selected_dominant_safe_route_on_two_frozen_benchmarks"
                if selected is not None
                else "not_admitted_without_two_benchmark_safety_and_gain"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
