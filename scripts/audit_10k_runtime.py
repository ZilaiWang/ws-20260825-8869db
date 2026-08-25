#!/usr/bin/env python3
"""Audit segmented 10K runtime measurements and enforce the 20 s gate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.experiments.runtime_10k import audit_runtime_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 E 的正式 10K 分段测速")
    parser.add_argument("--input", type=Path, required=True, help="逐次运行 JSONL")
    parser.add_argument("--hardware", type=Path, required=True, help="硬件环境 JSON")
    parser.add_argument(
        "--benchmark-contract",
        type=Path,
        required=True,
        help="图像来源、模型、切片和计时方法冻结合同",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-width", type=int, default=10000)
    parser.add_argument("--expected-height", type=int, default=10000)
    parser.add_argument("--minimum-measured-runs", type=int, default=10)
    parser.add_argument("--maximum-after-read-seconds", type=float, default=20.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = audit_runtime_file(
            input_path=args.input,
            output_path=args.output,
            hardware_path=args.hardware,
            benchmark_contract_path=args.benchmark_contract,
            expected_width=args.expected_width,
            expected_height=args.expected_height,
            minimum_measured_runs=args.minimum_measured_runs,
            maximum_after_read_seconds=args.maximum_after_read_seconds,
        )
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"RUNTIME_10K_AUDIT_FAIL: {error}", file=sys.stderr)
        return 1
    gate = summary["time_gate"]
    print(
        "RUNTIME_10K_AUDIT_"
        f"{'PASS' if gate['passed'] else 'SCIENTIFIC_FAIL'} "
        f"p50={summary['total_after_read']['p50_seconds']:.4f}s "
        f"p95={summary['total_after_read']['p95_seconds']:.4f}s "
        f"max={summary['total_after_read']['max_seconds']:.4f}s"
    )
    return 0 if gate["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
