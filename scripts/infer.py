#!/usr/bin/env python3
"""推理入口脚本。

支持:
    1. 单小图推理
    2. 10K 大图切片推理
    3. COCO JSON 输出

用法:
    python scripts/infer.py --config configs/infer.example.yaml
"""

import argparse
import sys
from pathlib import Path

from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="infer")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="模型推理",
    )
    parser.add_argument("--config", type=Path, required=True, help="推理配置文件")
    parser.add_argument("--checkpoint", type=Path, default=None, help="模型权重路径（覆盖配置）")
    parser.add_argument("--device", type=str, default=None, help="设备覆盖")
    parser.add_argument("--output", type=Path, default=None, help="输出 JSON 路径（覆盖配置）")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        return 1

    config = load_config(config_path)

    checkpoint = args.checkpoint or config.get("checkpoint", "")
    if not checkpoint:
        logger.error("未指定模型权重路径（--checkpoint 或配置文件 checkpoint 字段）")
        return 1

    tile_size = config.get("tile_size", 1024)
    tile_overlap = config.get("tile_overlap", 200)

    logger.info("checkpoint: %s", checkpoint)
    logger.info("device: %s", args.device or config.get("device", "cuda"))
    logger.info("tile_size/overlap: %s/%s", tile_size, tile_overlap)
    logger.info("输出: %s", args.output or config.get("output_json", "未指定"))
    logger.error("推理流水线尚未接入基线模型，未生成预测结果")
    return 2


if __name__ == "__main__":
    sys.exit(main())
