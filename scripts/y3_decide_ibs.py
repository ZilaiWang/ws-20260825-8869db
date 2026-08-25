#!/usr/bin/env python3
"""Decide whether the single IBS P2-neck pair improves formal P2 quality."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from rsdet.experiments.cv3_oof import OOF_CONTRACT_VERSION, sha256_file

DECISION_CONTRACT_VERSION = "y3_ibs_quality_decision_v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON 顶层必须是对象: {path}")
    return dict(payload)


def _coarse(payload: Mapping[str, Any], coarse: str) -> Mapping[str, Any]:
    return payload["official_ranking"]["per_coarse"][coarse]


def _fold_vehicle_fdr(result: Mapping[str, Any]) -> dict[int, float]:
    return {
        int(item["held_out_fold"]): float(
            _coarse(item["methods"]["C0_global"]["held_out"], "vehicle")["pooled_fdr"]
        )
        for item in result["per_fold"]
    }


def build_ibs_decision(
    *,
    p2_result: Mapping[str, Any],
    y3_result: Mapping[str, Any],
    y3_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    p2 = p2_result["merged_held_out"]["C0_global"]
    y3 = y3_result["merged_held_out"]["C0_global"]
    p2_vehicle = _coarse(p2, "vehicle")
    y3_vehicle = _coarse(y3, "vehicle")
    deltas = {
        "pooled_recall": float(y3["recall"]) - float(p2["recall"]),
        "pooled_fdr": float(y3["fdr"]) - float(p2["fdr"]),
        "macro_recall": float(y3["official_ranking"]["overall_macro_recall"])
        - float(p2["official_ranking"]["overall_macro_recall"]),
        "macro_fdr": float(y3["official_ranking"]["overall_macro_fdr"])
        - float(p2["official_ranking"]["overall_macro_fdr"]),
        "vehicle_recall": float(y3_vehicle["pooled_recall"]) - float(p2_vehicle["pooled_recall"]),
        "vehicle_fdr": float(y3_vehicle["pooled_fdr"]) - float(p2_vehicle["pooled_fdr"]),
    }
    p2_folds = _fold_vehicle_fdr(p2_result)
    y3_folds = _fold_vehicle_fdr(y3_result)
    if set(p2_folds) != {0, 1, 2} or set(y3_folds) != {0, 1, 2}:
        raise ValueError("P2/Y3 缺少三折 vehicle FDR")
    fold_directions = [
        {
            "fold": fold,
            "p2_vehicle_fdr": p2_folds[fold],
            "y3_vehicle_fdr": y3_folds[fold],
            "delta": y3_folds[fold] - p2_folds[fold],
        }
        for fold in range(3)
    ]
    positive_quality_folds = sum(item["delta"] < 0.0 for item in fold_directions)
    formal_integrity = bool(
        y3_metadata.get("contract_version") == OOF_CONTRACT_VERSION
        and y3_metadata.get("status") == "complete_downstream_ready"
        and y3_metadata.get("model_key") == "Y3"
        and int(y3_metadata.get("image_count", -1)) == 4481
        and int(y3_metadata.get("fold_count", -1)) == 3
        and float(y3_metadata.get("low_score_threshold", -1.0)) == 0.001
    )
    checks = {
        "formal_integrity": formal_integrity,
        "official_gate": bool(y3["official_gate_passed"]),
        "pooled_recall_noninferiority": deltas["pooled_recall"] >= -0.003,
        "pooled_fdr_safety": deltas["pooled_fdr"] <= 0.005,
        "macro_recall_noninferiority": deltas["macro_recall"] >= -0.003,
        "vehicle_recall_noninferiority": deltas["vehicle_recall"] >= -0.005,
        "quality_gain": deltas["vehicle_fdr"] <= -0.02 or deltas["macro_fdr"] <= -0.01,
        "vehicle_fdr_fold_direction": positive_quality_folds >= 2,
    }
    admitted = all(checks.values())
    return {
        "contract_version": DECISION_CONTRACT_VERSION,
        "status": "complete",
        "ibs_p2_pair_admission": admitted,
        "checks": checks,
        "rules": {
            "maximum_pooled_recall_drop": 0.003,
            "maximum_pooled_fdr_increase": 0.005,
            "maximum_macro_recall_drop": 0.003,
            "maximum_vehicle_recall_drop": 0.005,
            "minimum_vehicle_fdr_gain": 0.02,
            "alternative_minimum_macro_fdr_gain": 0.01,
            "minimum_positive_vehicle_fdr_folds": 2,
        },
        "deltas_vs_formal_P2_C0": deltas,
        "vehicle_fdr_fold_directions": fold_directions,
        "next_action": (
            "retain IBS pair and evaluate calibrated operating point"
            if admitted
            else "remove IBS pair; do not add SFRCF in the same branch"
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Y3 IBS P2 quality decision")
    parser.add_argument("--p2-calibration", type=Path, required=True)
    parser.add_argument("--y3-calibration", type=Path, required=True)
    parser.add_argument("--y3-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        p2 = args.p2_calibration.expanduser().resolve()
        y3 = args.y3_calibration.expanduser().resolve()
        metadata = args.y3_aggregate.expanduser().resolve() / "oof_metadata.json"
        decision = build_ibs_decision(
            p2_result=_load(p2),
            y3_result=_load(y3),
            y3_metadata=_load(metadata),
        )
        decision["artifacts"] = {
            "p2_calibration": {"path": str(p2), "sha256": sha256_file(p2)},
            "y3_calibration": {"path": str(y3), "sha256": sha256_file(y3)},
            "y3_metadata": {"path": str(metadata), "sha256": sha256_file(metadata)},
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
        print(f"Y3_IBS_DECISION_FAIL: {error}", file=sys.stderr)
        return 1
    print(f"Y3_IBS_DECISION_PASS admitted={decision['ibs_p2_pair_admission']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
