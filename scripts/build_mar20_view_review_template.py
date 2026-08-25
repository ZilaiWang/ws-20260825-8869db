#!/usr/bin/env python3
"""从既有 view_audit.csv 生成方法分列的人工复核模板。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

from rsdet.grouping.contracts import PROTOCOL_VERSION, atomic_write_json, sha256_file
from rsdet.grouping.view_review import build_view_review_rows, write_view_review_template


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 MAR20 view review v2 模板")
    parser.add_argument("--view-audit", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--primary-dilation", type=float, default=0.15)
    parser.add_argument("--summary-output")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_path = Path(args.view_audit).expanduser().resolve()
    with audit_path.open("r", encoding="utf-8", newline="") as file:
        audit_rows = list(csv.DictReader(file))
    rows = build_view_review_rows(audit_rows, primary_dilation=args.primary_dilation)
    output = Path(args.output).expanduser().resolve()
    write_view_review_template(output, rows)
    availability = [int(row["background_tile_available"]) for row in rows]
    summary = {
        "status": "waiting_for_method_specific_manual_review",
        "protocol_version": PROTOCOL_VERSION,
        "schema_version": "mar20-view-review-wide-v2",
        "primary_dilation": args.primary_dilation,
        "node_count": len(rows),
        "background_tile_available_count": sum(availability),
        "background_tile_unavailable_count": len(rows) - sum(availability),
        "view_audit_sha256": sha256_file(audit_path),
        "template_sha256": sha256_file(output),
    }
    summary_path = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output
        else output.with_suffix(".summary.json")
    )
    atomic_write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
