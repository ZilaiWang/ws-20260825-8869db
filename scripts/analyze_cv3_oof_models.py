#!/usr/bin/env python3
"""Analyze completed M1/M3 formal CV3 OOF aggregates on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from rsdet.analysis.oof_detection import run_formal_oof_analysis


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M1/M3 正式 OOF 官方指标、错误分解与对象级互补性分析",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/experiments/m1_m3_cv3_oof_analysis_v1.yaml"),
    )
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.yaml"),
    )
    parser.add_argument("--formal-crop-manifest", type=Path, required=True)
    parser.add_argument("--m1-aggregate", type=Path, required=True)
    parser.add_argument("--m3-aggregate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_formal_oof_analysis(
            config_path=args.config,
            project_config_path=args.project_config,
            formal_crop_manifest_path=args.formal_crop_manifest,
            m1_aggregate_dir=args.m1_aggregate,
            m3_aggregate_dir=args.m3_aggregate,
            output_dir=args.output_dir,
        )
    except (
        csv.Error,
        json.JSONDecodeError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        print(f"M1_M3_OOF_ANALYSIS_FAIL: {error}", file=sys.stderr)
        return 1
    print(
        "M1_M3_OOF_ANALYSIS_PASS "
        f"images={result['counts']['images']} "
        f"objects={result['counts']['ground_truth_objects']} "
        f"status={result['status']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
