#!/usr/bin/env python3
"""P04-3 raw/CleanDIFT 稳定性报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from rsdet.analysis.p04_features import (
    analyze_clean_d4,
    analyze_raw_ensemble,
    compare_repeat_caches,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 P04 扩散特征稳定性")
    parser.add_argument("--raw-cache", required=True)
    parser.add_argument("--clean-cache", required=True)
    parser.add_argument("--clean-repeat-cache")
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = {
        "status": "complete",
        "raw_ensemble": analyze_raw_ensemble(args.raw_cache),
        "clean_d4": analyze_clean_d4(args.clean_cache),
    }
    if args.clean_repeat_cache:
        report["clean_repeat"] = compare_repeat_caches(
            args.clean_cache, args.clean_repeat_cache
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))
    # ensemble4 是科学决策信号，不是代码/资产失败；即使不通过也要
    # 正常交付报告，只停止扩展 raw DIFT。repeat 确定性失败才返回 2。
    repeat_status = report.get("clean_repeat", {}).get("status", "pass")
    return 0 if repeat_status == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
