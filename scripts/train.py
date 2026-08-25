#!/usr/bin/env python3
"""Train the paper-aligned BHC-DETR detector."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rsdet.engine.trainer import train
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="训练 BHC-DETR，可选 UHR 小目标扩展（arXiv:2512.24074/2604.21435）"
    )
    parser.add_argument("--config", type=Path, required=True, help="BHC-DETR YAML 配置")
    parser.add_argument("--resume", type=Path, default=None, help="从 last.pt 恢复")
    parser.add_argument("--device", type=str, default=None, help="覆盖训练设备，如 cuda:0")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅审计配置、数据划分和样本，不构建模型/启动训练",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="仅用于冒烟测试：达到优化器步数后停止",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.config.is_file():
        logger.error("配置文件不存在: %s", args.config)
        return 1
    if args.max_steps is not None and args.max_steps <= 0:
        logger.error("--max-steps 必须 > 0")
        return 1
    try:
        result = train(
            load_config(args.config),
            resume=args.resume,
            device_override=args.device,
            dry_run=args.dry_run,
            max_steps=args.max_steps,
        )
    except (ImportError, OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.error("BHC-DETR 训练失败: %s", error)
        return 1
    logger.info("BHC-DETR: %s", result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
