#!/usr/bin/env python3
"""MG01：编译背景视图人工复核并执行冻结质量门禁。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from rsdet.grouping.contracts import PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.view_review import REVIEW_METHODS, build_view_review_rows


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="编译 MAR20 背景视图人工复核")
    parser.add_argument("--view-audit", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primary-dilation", type=float, default=0.15)
    parser.add_argument("--primary-fill", choices=REVIEW_METHODS, default="telea")
    parser.add_argument("--minimum-valid-rate", type=float, default=0.90)
    parser.add_argument("--maximum-aircraft-remnant-rate", type=float, default=0.05)
    parser.add_argument("--maximum-inpaint-artifact-rate", type=float, default=0.10)
    parser.add_argument("--maximum-background-tile-aircraft", type=int, default=0)
    return parser.parse_args(argv)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _binary(row: dict[str, str], column: str) -> int:
    value = row[column].strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{row['node_uid']}: {column} 必须完整填写为 0/1，actual={value!r}")
    return int(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    for name in (
        "minimum_valid_rate",
        "maximum_aircraft_remnant_rate",
        "maximum_inpaint_artifact_rate",
    ):
        value = float(getattr(args, name))
        if not 0 <= value <= 1:
            raise ValueError(f"{name} 必须在 [0,1]")
    if args.maximum_background_tile_aircraft < 0:
        raise ValueError("maximum-background-tile-aircraft 不得为负")
    audit_path = Path(args.view_audit).expanduser().resolve()
    review_path = Path(args.review).expanduser().resolve()
    audit = _read_csv(audit_path)
    review = _read_csv(review_path)
    if not audit or not review:
        raise ValueError("view audit/review 不能为空")
    expected_rows = build_view_review_rows(audit, primary_dilation=args.primary_dilation)
    expected_by_node = {row["node_uid"]: row for row in expected_rows}
    audit_nodes = set(expected_by_node)
    review_nodes = [row["node_uid"] for row in review]
    if len(review_nodes) != len(set(review_nodes)):
        raise ValueError("manual view review node_uid 重复")
    if set(review_nodes) != audit_nodes:
        raise ValueError(
            "manual view review 节点集合与 view audit 不一致: "
            f"missing={sorted(audit_nodes - set(review_nodes))[:10]}, "
            f"extra={sorted(set(review_nodes) - audit_nodes)[:10]}"
        )
    values = {"valid": [_binary(row, "valid") for row in review]}
    for method in REVIEW_METHODS:
        for suffix in ("aircraft_remnant", "inpaint_artifact"):
            column = f"{method}_{suffix}"
            values[column] = [_binary(row, column) for row in review]
    availability = []
    background_aircraft = []
    for row in review:
        node_uid = row["node_uid"]
        expected = int(expected_by_node[node_uid]["background_tile_available"])
        actual = _binary(row, "background_tile_available")
        if actual != expected:
            raise ValueError(
                f"{node_uid}: background_tile_available 是机器预填字段，"
                f"expected={expected}, actual={actual}"
            )
        availability.append(actual)
        text = row["background_tile_aircraft"].strip()
        if actual:
            background_aircraft.append(_binary(row, "background_tile_aircraft"))
        elif text not in {"", "0"}:
            raise ValueError(
                f"{node_uid}: 无 background tile 时 background_tile_aircraft 只能留空或填 0"
            )
    count = len(review)
    valid_indices = [index for index, value in enumerate(values["valid"]) if value]
    valid_count = len(valid_indices)
    method_metrics = {}
    admissible_methods = []
    for method in REVIEW_METHODS:
        remnant_values = values[f"{method}_aircraft_remnant"]
        artifact_values = values[f"{method}_inpaint_artifact"]
        remnant_rate = (
            sum(remnant_values[index] for index in valid_indices) / valid_count
            if valid_count
            else None
        )
        artifact_rate = (
            sum(artifact_values[index] for index in valid_indices) / valid_count
            if valid_count
            else None
        )
        admitted = bool(
            valid_count
            and remnant_rate is not None
            and artifact_rate is not None
            and remnant_rate <= args.maximum_aircraft_remnant_rate
            and artifact_rate <= args.maximum_inpaint_artifact_rate
        )
        method_metrics[method] = {
            "aircraft_remnant_rate": remnant_rate,
            "inpaint_artifact_rate": artifact_rate,
            "admitted": admitted,
        }
        if admitted:
            admissible_methods.append(method)
    background_aircraft_count = sum(background_aircraft)
    background_tile_admission = (
        bool(background_aircraft)
        and background_aircraft_count <= args.maximum_background_tile_aircraft
    )
    metrics = {
        "reviewed_nodes": count,
        "valid_rate": sum(values["valid"]) / count,
        "method_metrics": method_metrics,
        "background_tile_available_count": sum(availability),
        "background_tile_unavailable_count": count - sum(availability),
        "background_tile_aircraft_count": background_aircraft_count,
        "background_tile_admission": background_tile_admission,
    }
    failures = []
    if metrics["valid_rate"] < args.minimum_valid_rate:
        failures.append("valid_rate_below_threshold")
    if not admissible_methods:
        failures.append("no_mask_fill_method_passed_manual_gate")
    formal_view_admission = not failures
    report = {
        "status": "pass" if formal_view_admission else "fail",
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": "mar20-view-review-wide-v2",
        "formal_view_admission": formal_view_admission,
        "admissible_fill_methods": admissible_methods,
        "primary_fill": args.primary_fill,
        "primary_fill_admission": args.primary_fill in admissible_methods,
        "background_tile_admission": background_tile_admission,
        "metrics": metrics,
        "thresholds": {
            "primary_dilation": args.primary_dilation,
            "minimum_valid_rate": args.minimum_valid_rate,
            "maximum_aircraft_remnant_rate": args.maximum_aircraft_remnant_rate,
            "maximum_inpaint_artifact_rate": args.maximum_inpaint_artifact_rate,
            "maximum_background_tile_aircraft": args.maximum_background_tile_aircraft,
        },
        "failures": failures,
        "inputs": {
            "view_audit_sha256": sha256_file(audit_path),
            "manual_review_sha256": sha256_file(review_path),
        },
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if formal_view_admission else 2


if __name__ == "__main__":
    raise SystemExit(main())
