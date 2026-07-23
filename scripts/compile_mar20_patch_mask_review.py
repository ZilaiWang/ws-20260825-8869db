#!/usr/bin/env python3
"""Compile the manual patch-mask overlay audit without changing frozen thresholds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from rsdet.grouping.contracts import MASKED_PATCH_PROTOCOL_VERSION, atomic_write_json, sha256_file


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile MAR20 patch-mask review")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-valid-rate", type=float, default=0.95)
    parser.add_argument("--minimum-aircraft-coverage-rate", type=float, default=0.95)
    parser.add_argument("--maximum-excessive-loss-rate", type=float, default=0.10)
    return parser.parse_args(argv)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _bit(row: dict[str, str], field: str) -> int:
    value = row[field].strip()
    if value not in {"0", "1"}:
        raise ValueError(f"{row.get('node_uid')}:{field} must be 0 or 1")
    return int(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_path = Path(args.audit).expanduser().resolve()
    summary_path = Path(args.audit_summary).expanduser().resolve()
    review_path = Path(args.review).expanduser().resolve()
    audit = _read(audit_path)
    review = _read(review_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("automatic_geometry_gate") != "pass":
        raise ValueError("automatic patch-mask gate did not pass")
    by_uid = {row["node_uid"]: row for row in review}
    if len(by_uid) != len(review) or {row["node_uid"] for row in audit} != set(by_uid):
        raise ValueError("audit/review node set mismatch or duplicate")
    valid = [_bit(row, "valid") for row in review]
    cover_fields = [
        "dilation_0p10_aircraft_covered",
        "dilation_0p15_aircraft_covered",
        "dilation_0p20_aircraft_covered",
    ]
    coverage = {
        field: sum(_bit(row, field) for row in review) / len(review) for field in cover_fields
    }
    excessive = sum(_bit(row, "dilation_0p15_excessive_background_loss") for row in review) / len(
        review
    )
    metrics = {
        "reviewed_nodes": len(review),
        "valid_rate": sum(valid) / len(valid),
        "coverage_rate": coverage,
        "primary_excessive_background_loss_rate": excessive,
    }
    admitted = bool(
        metrics["valid_rate"] >= args.minimum_valid_rate
        and min(coverage.values()) >= args.minimum_aircraft_coverage_rate
        and excessive <= args.maximum_excessive_loss_rate
    )
    result = {
        "status": "pass" if admitted else "fail",
        "protocol_version": MASKED_PATCH_PROTOCOL_VERSION,
        "formal_patch_mask_admission": admitted,
        "metrics": metrics,
        "thresholds": {
            "minimum_valid_rate": args.minimum_valid_rate,
            "minimum_aircraft_coverage_rate": args.minimum_aircraft_coverage_rate,
            "maximum_excessive_background_loss_rate": args.maximum_excessive_loss_rate,
        },
        "inputs": {
            "audit_sha256": sha256_file(audit_path),
            "audit_summary_sha256": sha256_file(summary_path),
            "review_sha256": sha256_file(review_path),
        },
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
