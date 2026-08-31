#!/usr/bin/env python3
"""Compile one frozen Normal/Hard/Sentinel HERA-Guard candidate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rsdet.evaluation.absolute_score import competition_score


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or "frontiers" not in payload:
        raise ValueError(f"frontier is not complete: {path}")
    return payload


def _row(payload: dict, level: str) -> dict:
    try:
        return payload["frontiers"][level]["crossfit"]
    except KeyError as error:
        raise ValueError(f"frontier lacks FDR level {level}") from error


def _compare_rows(base: dict, candidate: dict, base_path: Path, candidate_path: Path) -> dict:
    per_coarse = {}
    for coarse in ("ship", "aircraft", "vehicle"):
        before = base["per_coarse"][coarse]
        after = candidate["per_coarse"][coarse]
        per_coarse[coarse] = {
            "base_recall": float(before["recall"]),
            "candidate_recall": float(after["recall"]),
            "delta_recall": float(after["recall"] - before["recall"]),
            "base_fdr": float(before["fdr"]),
            "candidate_fdr": float(after["fdr"]),
            "delta_fdr": float(after["fdr"] - before["fdr"]),
        }
    base_score = competition_score(float(base["recall"]), float(base["fdr"]), 0.0)
    candidate_score = competition_score(
        float(candidate["recall"]), float(candidate["fdr"]), 0.0
    )
    return {
        "base_frontier": str(base_path.resolve()),
        "candidate_frontier": str(candidate_path.resolve()),
        "base_sha256": _sha256(base_path),
        "candidate_sha256": _sha256(candidate_path),
        "base_recall": float(base["recall"]),
        "candidate_recall": float(candidate["recall"]),
        "delta_recall": float(candidate["recall"] - base["recall"]),
        "base_fdr": float(base["fdr"]),
        "candidate_fdr": float(candidate["fdr"]),
        "delta_fdr": float(candidate["fdr"] - base["fdr"]),
        "base_macro_recall": float(base["macro_recall"]),
        "candidate_macro_recall": float(candidate["macro_recall"]),
        "delta_macro_recall": float(candidate["macro_recall"] - base["macro_recall"]),
        "absolute_score_equal_latency": {
            "base": base_score["total_score"],
            "candidate": candidate_score["total_score"],
            "delta": candidate_score["total_score"] - base_score["total_score"],
            "note": "pooled Recall/FDR interpretation; latency fixed and cancels",
        },
        "per_coarse": per_coarse,
    }


def _compare(base_path: Path, candidate_path: Path, level: str) -> dict:
    base_payload = _load(base_path)
    candidate_payload = _load(candidate_path)
    return _compare_rows(
        _row(base_payload, level), _row(candidate_payload, level), base_path, candidate_path
    )


def _load_frozen(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete" or "metrics" not in payload:
        raise ValueError(f"frozen-threshold evaluation is not complete: {path}")
    metrics = payload["metrics"]
    return {
        "recall": float(metrics["recall"]),
        "fdr": float(metrics["fdr"]),
        "macro_recall": float(metrics["macro_recall"]),
        "per_coarse": metrics["per_coarse"],
    }


def _compare_frozen(base_path: Path, candidate_path: Path) -> dict:
    return _compare_rows(
        _load_frozen(base_path), _load_frozen(candidate_path), base_path, candidate_path
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for condition in ("normal", "hard", "sentinel"):
        parser.add_argument(f"--{condition}-base", type=Path, required=True)
        parser.add_argument(f"--{condition}-candidate", type=Path, required=True)
    parser.add_argument("--sentinel-base-frozen", type=Path)
    parser.add_argument("--sentinel-candidate-frozen", type=Path)
    parser.add_argument("--fdr-level", default="0.150")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.sentinel_base_frozen is None) != (args.sentinel_candidate_frozen is None):
        parser.error("provide both frozen Sentinel evaluations, or neither")
    comparisons = {
        condition: _compare(
            getattr(args, f"{condition}_base"),
            getattr(args, f"{condition}_candidate"),
            args.fdr_level,
        )
        for condition in ("normal", "hard", "sentinel")
    }
    sentinel_thresholds_frozen_from_hard = args.sentinel_base_frozen is not None
    if sentinel_thresholds_frozen_from_hard:
        comparisons["sentinel"] = _compare_frozen(
            args.sentinel_base_frozen, args.sentinel_candidate_frozen
        )
    normal = comparisons["normal"]
    hard = comparisons["hard"]
    sentinel = comparisons["sentinel"]
    normal_floor = (
        normal["delta_recall"] >= -0.003
        and normal["delta_macro_recall"] >= -0.003
    )
    all_coarse_guard = all(
        condition["per_coarse"][coarse]["delta_recall"] >= -0.005
        for condition in comparisons.values()
        for coarse in ("ship", "aircraft", "vehicle")
    )
    risk_guard = all(condition["delta_fdr"] <= 0.01 for condition in comparisons.values())
    primary_gain = any(
        hard["per_coarse"][coarse]["delta_recall"] >= 0.005
        and sentinel["per_coarse"][coarse]["delta_recall"] >= 0.0
        for coarse in ("ship", "vehicle")
    )
    absolute_score_guard = (
        hard["absolute_score_equal_latency"]["delta"] >= 0.0
        and sentinel["absolute_score_equal_latency"]["delta"] >= 0.0
    )
    admitted = (
        normal_floor
        and all_coarse_guard
        and risk_guard
        and primary_gain
        and absolute_score_guard
    )
    payload = {
        "status": "complete",
        "protocol": "hera_guard_final_three_benchmark_frozen_candidate_gate_v2",
        "fdr_level": args.fdr_level,
        "sentinel_thresholds_frozen_from_hard": sentinel_thresholds_frozen_from_hard,
        "threshold_tuning_on_sentinel": not sentinel_thresholds_frozen_from_hard,
        "comparisons": comparisons,
        "gates": {
            "normal_recall_and_macro_floor_minus_0p3pp": normal_floor,
            "all_coarse_recall_floor_minus_0p5pp": all_coarse_guard,
            "fdr_worsening_at_most_1pp": risk_guard,
            "hard_primary_gain_0p5pp_and_sentinel_same_direction": primary_gain,
            "hard_and_frozen_sentinel_absolute_score_non_decrease": absolute_score_guard,
        },
        "decision": {
            "formal_cv3_expansion_admission": admitted,
            "next_action": "expand_cv3" if admitted else "stop_without_parameter_scan",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
