#!/usr/bin/env python3
"""训练入口脚本。

用法:
    python scripts/train.py --config configs/train.example.yaml
"""

import argparse
import logging
import sys
from pathlib import Path

from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="train")


def parse_args(argv: list | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="模型训练",
    )
    parser.add_argument("--config", type=Path, required=True, help="训练配置文件")
    parser.add_argument("--resume", type=Path, default=None, help="从 checkpoint 恢复")
    parser.add_argument("--device", type=str, default=None, help="设备覆盖（如 cpu, cuda:0）")
    return parser.parse_args(argv)


def main(argv: list | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return 1

    config = load_config(config_path)
    model_name = config.get("model", "")

    if not model_name:
        logger.warning(
            "尚未选择基线模型。请在配置文件中设置 model 字段，"
            "并在 registry 中注册对应检测器。"
        )
        logger.info("当前已注册的模型: 待执行 python -c 'from rsdet.models.registry import list_models; print(list_models())' 查看")
        return 1

    logger.info(f"训练配置已加载，模型: {model_name}")
    logger.warning("训练功能待基线模型确定后实现")
    return 0


if __name__ == "__main__":
    sys.exit(main())
