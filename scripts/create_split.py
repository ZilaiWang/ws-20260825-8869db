#!/usr/bin/env python3
"""生成 XH-202625 的冻结 group-aware dev split。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rsdet.data.splits import create_dev_split
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="create_split")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成分组分层 dev_v1 划分")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="只读数据集根目录"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/splits/dev_v1.json"),
        help="manifest 输出路径",
    )
    parser.add_argument("--version", default="dev_v1")
    parser.add_argument("--data-version", default="official_raw_v1")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-exact-duplicate-merge",
        action="store_true",
        help="不合并字节级完全重复图（不推荐）",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        statistics = create_dev_split(
            data_root=args.data_root,
            output_path=args.output,
            version=args.version,
            data_version=args.data_version,
            val_ratio=args.val_ratio,
            seed=args.seed,
            merge_exact_duplicates=not args.no_exact_duplicate_merge,
        )
    except (
        OSError,
        TypeError,
        ValueError,
        RuntimeError,
        json.JSONDecodeError,
    ) as error:
        logger.error("划分失败: %s", error)
        return 1
    logger.info(
        "划分完成: train=%d, val=%d, val_ratio=%.4f, groups=%d",
        statistics["train"]["images"],
        statistics["val"]["images"],
        statistics["val_image_ratio"],
        statistics["grouping"]["final_groups"],
    )
    logger.info("manifest: %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
