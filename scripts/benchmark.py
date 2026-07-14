#!/usr/bin/env python3
"""推理速度测试脚本。

统计完整推理流程各阶段耗时，不只用 model forward 时间。
"""

import argparse
import sys
from pathlib import Path

from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="benchmark")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="推理速度测试",
    )
    parser.add_argument("--config", type=Path, required=True, help="推理配置文件")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--runs", type=int, default=10, help="测试次数")
    parser.add_argument("--image-size", type=int, default=10000, help="模拟图像大小（无数据时）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.exists():
        logger.error("配置文件不存在: %s", args.config)
        return 1
    load_config(args.config)
    logger.info(
        "测速计划: warmup=%d, runs=%d, image=%dx%d",
        args.warmup,
        args.runs,
        args.image_size,
        args.image_size,
    )
    logger.error("完整测速尚未接入模型和图像流水线，未生成任何性能结果")
    return 2


if __name__ == "__main__":
    sys.exit(main())
