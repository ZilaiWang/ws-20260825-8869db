#!/usr/bin/env python3
"""Audit and aggregate three held-out fold prediction files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.experiments.cv3_oof import audit_and_aggregate_oof


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计并汇总 M1/M3/P2/Y3 三折 OOF")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    formal = parser.add_mutually_exclusive_group(required=True)
    formal.add_argument("--formal-crop-manifest", type=Path)
    formal.add_argument(
        "--diagnostic-without-formal-crop",
        action="store_true",
        help="仅生成不可下游消费的诊断 aggregate；必须使用独立新目录",
    )
    parser.add_argument("--expected-image-count", type=int, default=4481)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        metadata = audit_and_aggregate_oof(
            manifest_path=args.manifest,
            plan_path=args.plan,
            run_root=args.run_root,
            output_dir=args.output_dir,
            expected_manifest_sha256=args.manifest_sha256,
            expected_image_count=args.expected_image_count,
            formal_crop_manifest_path=args.formal_crop_manifest,
            diagnostic_without_formal_crop=args.diagnostic_without_formal_crop,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"CV3_OOF_AUDIT_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "CV3_OOF_AUDIT_PASS "
        f"model={metadata['model_key']} images={metadata['image_count']} "
        f"proposals={metadata['proposal_count']} "
        f"downstream={metadata['downstream_admission']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
