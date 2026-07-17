#!/usr/bin/env python3
"""P04 特征缓存完整性审计。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rsdet.analysis.p04_features import compare_cache_overlap, compare_repeat_caches
from rsdet.features.p04_cache import FeatureCache


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 P04 feature cache")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--skip-sha256", action="store_true")
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument("--expected-objects", type=int)
    parser.add_argument(
        "--expected-feature",
        action="append",
        default=[],
        metavar="NAME=DIM",
        help="可重复；将缓存特征名和维度变成硬门禁",
    )
    parser.add_argument(
        "--expected-common-rows",
        type=int,
        help="compare-cache/overlap 的每个共同特征必须有该数量公共行",
    )
    parser.add_argument(
        "--compare-cache",
        help="可选的同配置重复提取 cache，用于确定性门禁",
    )
    parser.add_argument(
        "--compare-overlap-cache",
        help="只比较公共 annotation/view，用于 calibration 与全量 cache 一致性",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.compare_cache and args.compare_overlap_cache:
        raise ValueError("compare-cache 与 compare-overlap-cache 不得同时使用")
    if args.expected_rows is not None and args.expected_rows <= 0:
        raise ValueError("--expected-rows 必须大于 0")
    if args.expected_objects is not None and args.expected_objects <= 0:
        raise ValueError("--expected-objects 必须大于 0")
    if args.expected_common_rows is not None and args.expected_common_rows <= 0:
        raise ValueError("--expected-common-rows 必须大于 0")
    cache = FeatureCache(args.cache_dir, verify_sha256=not args.skip_sha256)
    report = cache.audit()
    failures: list[str] = []
    if args.expected_rows is not None and report["row_count"] != args.expected_rows:
        failures.append(
            f"row_count expected={args.expected_rows}, actual={report['row_count']}"
        )
    if args.expected_objects is not None and report["object_count"] != args.expected_objects:
        failures.append(
            f"object_count expected={args.expected_objects}, actual={report['object_count']}"
        )
    expected_features: dict[str, int] = {}
    for value in args.expected_feature:
        try:
            name, dimension_text = value.split("=", maxsplit=1)
            dimension = int(dimension_text)
        except ValueError as error:
            raise ValueError("--expected-feature 必须是 NAME=DIM") from error
        if not name or dimension <= 0 or name in expected_features:
            raise ValueError("--expected-feature 名称须唯一且 DIM>0")
        expected_features[name] = dimension
    if expected_features and report["feature_dimensions"] != expected_features:
        failures.append(
            "feature_dimensions expected="
            f"{expected_features}, actual={report['feature_dimensions']}"
        )
    if args.compare_cache:
        report["repeat_comparison"] = compare_repeat_caches(
            args.cache_dir, args.compare_cache
        )
        if report["repeat_comparison"]["status"] != "pass":
            report["status"] = "fail"
    if args.compare_overlap_cache:
        report["overlap_comparison"] = compare_cache_overlap(
            args.cache_dir, args.compare_overlap_cache
        )
        if report["overlap_comparison"]["status"] != "pass":
            report["status"] = "fail"
    if args.expected_common_rows is not None:
        comparison = report.get("repeat_comparison") or report.get("overlap_comparison")
        if comparison is None:
            raise ValueError("--expected-common-rows 必须与 compare 参数一起使用")
        observed = {
            name: values["common_row_count"]
            for name, values in comparison["features"].items()
        }
        wrong = {
            name: count
            for name, count in observed.items()
            if count != args.expected_common_rows
        }
        if wrong:
            failures.append(
                f"common_row_count expected={args.expected_common_rows}, actual={wrong}"
            )
    if failures:
        report["status"] = "fail"
    report["expectation_failures"] = failures
    if args.output:
        target = Path(args.output).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
