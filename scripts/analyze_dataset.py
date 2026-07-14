#!/usr/bin/env python3
"""数据审计入口脚本。

用法:
    python scripts/analyze_dataset.py --data-root /path/to/data --output-dir outputs/audit
"""

import argparse
import logging
import sys
from pathlib import Path

from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="analyze_dataset")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="数据分析与审计工具",
    )
    parser.add_argument("--data-root", type=Path, required=True, help="数据集根目录")
    parser.add_argument("--config", type=Path, default=None, help="数据配置文件（可选）")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/audit"), help="报告输出目录")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    if args.verbose:
        logging.getLogger("rsdet").setLevel(logging.DEBUG)

    # 检查数据根目录
    data_root = Path(args.data_root)
    if not data_root.exists():
        logger.error(f"数据根目录不存在: {data_root}")
        return 1

    # 基础文件统计
    logger.info(f"数据根目录: {data_root}")
    files = list(data_root.rglob("*"))
    logger.info(f"文件总数: {len(files)}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_files = [f for f in files if f.suffix.lower() in image_extensions]
    logger.info(f"图像文件数: {len(image_files)}")

    # 输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"报告输出目录: {args.output_dir}")

    # TODO: 接入完整数据审计模块（类别分布、bbox 统计、稀疏标注检查等）

    logger.info("数据审计完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
