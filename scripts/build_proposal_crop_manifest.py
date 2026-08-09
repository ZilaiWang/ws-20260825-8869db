#!/usr/bin/env python3
"""N2-1：从对象证据 manifest 生成 Pred-OOF proposal crop manifest。

用法（CPU 即可）：
    PYTHONPATH=src python scripts/build_proposal_crop_manifest.py \
        --evidence outputs/N0-EVIDENCE-M1/pred_oof_evidence.json \
        --formal-manifest outputs/P0-2-FORMAL-CROP-LOCALCHECK/formal_crop_manifest.csv \
        --output outputs/N2-PROPO-CROP/proposal_crop_manifest.csv \
        --include-views deployable_positive oracle_positive hard_negative \
        --background-candidates fp_bg_no_oracle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.analysis.proposal_crops import (
    build_proposal_crop_manifest,
    split_by_fold,
    stats_by_view,
    write_proposal_crop_manifest,
)
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="n2_proposal_crop")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="N2-1 proposal crop manifest",
    )
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--include-views",
        nargs="+",
        default=["deployable_positive", "oracle_positive", "hard_negative"],
    )
    parser.add_argument(
        "--background-candidates",
        default="fp_bg_no_oracle",
        choices=("fp_bg_no_oracle",),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """运行 N2-1 manifest 生成。"""
    args = parse_args(argv)
    try:
        evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
        rows, summary = build_proposal_crop_manifest(
            evidence_manifest=evidence,
            formal_manifest_path=args.formal_manifest,
            include_views=args.include_views,
            background_candidates=args.background_candidates,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("manifest 生成失败: %s", error)
        return 1

    output = Path(args.output).expanduser().resolve()
    write_proposal_crop_manifest(rows, output)
    folded = split_by_fold(rows)
    by_view = stats_by_view(rows)

    logger.info("=== N2-1 proposal crop manifest ===")
    logger.info(
        "候选总数 %d → 写入 %d（缺图像路径 %d）",
        summary["total_candidates"],
        summary["rows_written"],
        summary["missing_image_paths"],
    )
    logger.info("视图计数: %s", summary["view_counts"])
    logger.info("背景候选: %d", summary["background_candidates"])
    for fold in sorted(folded):
        logger.info("  fold %d: %d 条", fold, len(folded[fold]))
    for view, stats in sorted(by_view.items()):
        logger.info(
            "  %s: n=%d background=%d",
            view,
            stats["n"],
            stats["background"],
        )
    logger.info("已保存: %s", output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
