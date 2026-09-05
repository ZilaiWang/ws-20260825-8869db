#!/usr/bin/env python3
"""Decompose training- and inference-resolution effects from four frontiers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

KEYS = ("t1024_i1024", "t1024_i1280", "t1280_i1024", "t1280_i1280")
COARSE = ("ship", "aircraft", "vehicle")


def _point(payload: dict[str, Any], level: str) -> dict[str, Any]:
    row = payload["frontiers"][level]
    if "crossfit" in row:
        metrics = row["crossfit"]
        threshold: float | dict[str, float] = {
            str(key): float(value) for key, value in row["crossfit_thresholds"].items()
        }
        selection = "outer_crossfit"
    else:
        metrics = row
        threshold = float(row["threshold"])
        selection = "same_split_oracle"
    platform = metrics["platform"]
    return {
        "selection": selection,
        "threshold": threshold,
        "gate_recall": float(platform["gate_recall"]),
        "gate_fdr": float(platform["gate_fdr"]),
        "per_coarse": {
            name: {
                "recall": float(platform["per_coarse"][name]["macro_recall"]),
                "fdr": float(platform["per_coarse"][name]["macro_fdr"]),
            }
            for name in COARSE
        },
    }


def _subtract(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    return {
        "gate_recall_pp": 100.0 * (first["gate_recall"] - second["gate_recall"]),
        "gate_fdr_pp": 100.0 * (first["gate_fdr"] - second["gate_fdr"]),
        "per_coarse": {
            name: {
                "recall_pp": 100.0
                * (first["per_coarse"][name]["recall"] - second["per_coarse"][name]["recall"]),
                "fdr_pp": 100.0
                * (first["per_coarse"][name]["fdr"] - second["per_coarse"][name]["fdr"]),
            }
            for name in COARSE
        },
    }


def _interaction(
    x00: dict[str, Any], x01: dict[str, Any], x10: dict[str, Any], x11: dict[str, Any]
) -> dict[str, Any]:
    return {
        "gate_recall_pp": 100.0
        * (x11["gate_recall"] - x10["gate_recall"] - x01["gate_recall"] + x00["gate_recall"]),
        "gate_fdr_pp": 100.0
        * (x11["gate_fdr"] - x10["gate_fdr"] - x01["gate_fdr"] + x00["gate_fdr"]),
        "per_coarse": {
            name: {
                field: 100.0
                * (
                    x11["per_coarse"][name][source]
                    - x10["per_coarse"][name][source]
                    - x01["per_coarse"][name][source]
                    + x00["per_coarse"][name][source]
                )
                for field, source in (("recall_pp", "recall"), ("fdr_pp", "fdr"))
            }
            for name in COARSE
        },
    }


def _data_identity(payload: dict[str, Any]) -> tuple[str, str]:
    if "fold_image_ids" in payload:
        return "fold_image_ids", json.dumps(payload["fold_image_ids"], sort_keys=True)
    return "gt_sha256", str(payload["input_sha256"]["gt"])


def analyze(frontiers: dict[str, dict[str, Any]], level: str = "0.150") -> dict[str, Any]:
    if set(frontiers) != set(KEYS):
        raise ValueError(f"exactly these inputs are required: {KEYS}")
    identities = {_data_identity(payload) for payload in frontiers.values()}
    if len(identities) != 1:
        raise ValueError("all four frontiers must use identical evaluation images")
    points = {key: _point(value, level) for key, value in frontiers.items()}
    selections = {point["selection"] for point in points.values()}
    if len(selections) != 1:
        raise ValueError("cannot mix same-split oracle and outer-crossfit frontiers")
    x00, x01, x10, x11 = (points[key] for key in KEYS)
    return {
        "schema_version": "resolution_cross_matrix_v2",
        "diagnostic_only": True,
        "selection": next(iter(selections)),
        "fdr_level_is_constraint_not_heldout_guarantee": True,
        "fdr_level": level,
        "points": points,
        "effects": {
            "inference_resolution_at_train1024": _subtract(x01, x00),
            "inference_resolution_at_train1280": _subtract(x11, x10),
            "training_resolution_at_infer1024": _subtract(x10, x00),
            "training_resolution_at_infer1280": _subtract(x11, x01),
            "total_x11_vs_x00": _subtract(x11, x00),
            "train_infer_interaction": _interaction(x00, x01, x10, x11),
        },
        "warning": (
            "The decomposition compares operating points under one selection contract. "
            "It is diagnostic and does not authorize copying thresholds to deployment."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in KEYS:
        parser.add_argument("--" + argument.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--fdr-level", default="0.150")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frontiers = {
        key: json.loads(getattr(args, key).read_text(encoding="utf-8")) for key in KEYS
    }
    result = analyze(frontiers, args.fdr_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["effects"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
