#!/usr/bin/env python3
"""Decide whether the formal P2 structure is admitted over M1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from rsdet.experiments.cv3_oof import OOF_CONTRACT_VERSION, sha256_file

DECISION_CONTRACT_VERSION = "y2_p2_formal_decision_v1"


def _load(path: Path, field: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{field} 顶层必须是对象")
    return dict(payload)


def _fold_vehicle_recall(result: Mapping[str, Any]) -> dict[int, float]:
    output: dict[int, float] = {}
    for fold in result["per_fold"]:
        fold_id = int(fold["held_out_fold"])
        vehicle = fold["methods"]["C0_global"]["held_out"]["official_ranking"]["per_coarse"][
            "vehicle"
        ]
        output[fold_id] = float(vehicle["pooled_recall"])
    if set(output) != {0, 1, 2}:
        raise ValueError("校准结果缺少三折 vehicle held-out 指标")
    return output


def build_p2_decision(
    *,
    m1_result: Mapping[str, Any],
    p2_result: Mapping[str, Any],
    p2_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen single-factor admission contract."""

    formal_integrity = bool(
        p2_metadata.get("contract_version") == OOF_CONTRACT_VERSION
        and p2_metadata.get("status") == "complete_downstream_ready"
        and p2_metadata.get("model_key") == "P2"
        and int(p2_metadata.get("image_count", -1)) == 4481
        and int(p2_metadata.get("fold_count", -1)) == 3
        and float(p2_metadata.get("low_score_threshold", -1.0)) == 0.001
    )
    m1 = m1_result["merged_held_out"]["C0_global"]
    p2 = p2_result["merged_held_out"]["C0_global"]
    m1_vehicle = m1["official_ranking"]["per_coarse"]["vehicle"]
    p2_vehicle = p2["official_ranking"]["per_coarse"]["vehicle"]
    deltas = {
        "pooled_recall": float(p2["recall"]) - float(m1["recall"]),
        "pooled_fdr": float(p2["fdr"]) - float(m1["fdr"]),
        "macro_recall": float(p2["official_ranking"]["overall_macro_recall"])
        - float(m1["official_ranking"]["overall_macro_recall"]),
        "macro_fdr": float(p2["official_ranking"]["overall_macro_fdr"])
        - float(m1["official_ranking"]["overall_macro_fdr"]),
        "vehicle_pooled_recall": float(p2_vehicle["pooled_recall"])
        - float(m1_vehicle["pooled_recall"]),
        "vehicle_pooled_fdr": float(p2_vehicle["pooled_fdr"]) - float(m1_vehicle["pooled_fdr"]),
    }
    m1_fold = _fold_vehicle_recall(m1_result)
    p2_fold = _fold_vehicle_recall(p2_result)
    fold_directions = [
        {
            "fold": fold,
            "m1_vehicle_recall": m1_fold[fold],
            "p2_vehicle_recall": p2_fold[fold],
            "delta": p2_fold[fold] - m1_fold[fold],
        }
        for fold in range(3)
    ]
    positive_vehicle_folds = sum(item["delta"] > 0.0 for item in fold_directions)
    checks = {
        "formal_integrity": formal_integrity,
        "official_gate": bool(p2["official_gate_passed"]),
        "pooled_recall_noninferiority": deltas["pooled_recall"] >= -0.005,
        "pooled_fdr_safety": deltas["pooled_fdr"] <= 0.01,
        "macro_recall_noninferiority": deltas["macro_recall"] >= -0.005,
        "vehicle_recall_minimum_gain": deltas["vehicle_pooled_recall"] >= 0.02,
        "vehicle_fold_direction": positive_vehicle_folds >= 2,
    }
    admitted = all(checks.values())
    return {
        "contract_version": DECISION_CONTRACT_VERSION,
        "status": "complete",
        "p2_structure_admission": admitted,
        "quality_stage_admission": admitted,
        "checks": checks,
        "rules": {
            "maximum_pooled_recall_drop": 0.005,
            "maximum_pooled_fdr_increase": 0.01,
            "maximum_macro_recall_drop": 0.005,
            "minimum_vehicle_recall_gain": 0.02,
            "minimum_positive_vehicle_folds": 2,
        },
        "deltas_vs_m1_C0": deltas,
        "vehicle_fold_directions": fold_directions,
        "interpretation": (
            "P2 admitted; Y3 may test one quality module."
            if admitted
            else "P2 not admitted; stop Y3 and retain the historical mechanism evidence only."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Y2 formal P2 scientific decision")
    parser.add_argument("--m1-calibration", type=Path, required=True)
    parser.add_argument("--p2-calibration", type=Path, required=True)
    parser.add_argument("--p2-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        m1_path = args.m1_calibration.expanduser().resolve()
        p2_path = args.p2_calibration.expanduser().resolve()
        metadata_path = args.p2_aggregate.expanduser().resolve() / "oof_metadata.json"
        decision = build_p2_decision(
            m1_result=_load(m1_path, "M1 calibration"),
            p2_result=_load(p2_path, "P2 calibration"),
            p2_metadata=_load(metadata_path, "P2 aggregate metadata"),
        )
        decision["artifacts"] = {
            "m1_calibration": {"path": str(m1_path), "sha256": sha256_file(m1_path)},
            "p2_calibration": {"path": str(p2_path), "sha256": sha256_file(p2_path)},
            "p2_metadata": {
                "path": str(metadata_path),
                "sha256": sha256_file(metadata_path),
            },
        }
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"决策文件已存在，禁止覆盖: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(decision, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        print(f"Y2_P2_DECISION_FAIL: {error}", file=sys.stderr)
        return 1
    print(f"Y2_P2_DECISION_PASS admitted={decision['p2_structure_admission']} output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
