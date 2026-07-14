#!/usr/bin/env python3
"""推理速度测试脚本。

统计完整推理流程各阶段耗时，不只用 model forward 时间。
"""

import argparse
import logging
import sys
from pathlib import Path

from rsdet.evaluation.runtime import RuntimeBreakdown, timed_block
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="benchmark")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="推理速度测试",
    )
    parser.add_argument("--config", type=Path, required=True, help="推理配置文件")
    parser.add_argument("--warmup", type=int, default=3, help="预热次数")
    parser.add_argument("--runs", type=int, default=10, help="测试次数")
    parser.add_argument("--image-size", type=int, default=10000, help="模拟图像大小（无数据时）")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    logger.info(f"推理速度测试（预热 {args.warmup} 次，测试 {args.runs} 次）")
    logger.info(f"模拟图像大小: {args.image_size}x{args.image_size}")

    # 演示计时框架
    rt = RuntimeBreakdown()

    # 模拟各阶段耗时
    with timed_block(rt, "tiling"):
        pass  # TODO: 实际切片

    with timed_block(rt, "model"):
        pass  # TODO: 实际推理

    logger.info("各阶段耗时:")
    for k, v in rt.to_dict().items():
        logger.info(f"  {k}: {v:.4f}s")

    logger.warning("benchmark 功能待模型确定后实现完整流程")
    return 0


if __name__ == "__main__":
    sys.exit(main())
