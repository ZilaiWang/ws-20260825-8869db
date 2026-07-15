#!/usr/bin/env python3
"""校验模型交付的 COCO detection JSON，不运行模型或评测。"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rsdet.predictions import load_coco_prediction_records, validate_coco_prediction_records
from rsdet.utils.config import load_config
from rsdet.utils.logging import setup_logging

logger = setup_logging(name="validate_predictions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="校验统一 COCO prediction JSON")
    parser.add_argument("--pred", type=Path, required=True, help="待校验预测 JSON")
    parser.add_argument(
        "--project-config",
        type=Path,
        default=Path("configs/project.yaml"),
        help="类别映射配置",
    )
    parser.add_argument(
        "--gt",
        type=Path,
        default=None,
        help="可选 COCO GT；提供后额外校验 image_id 和图像边界",
    )
    return parser.parse_args(argv)


def _load_image_sizes(path: Path) -> dict[int, tuple[int, int]]:
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError("GT 必须是含 images 列表的 COCO JSON 对象")

    image_sizes: dict[int, tuple[int, int]] = {}
    for index, image in enumerate(data["images"]):
        if not isinstance(image, dict):
            raise ValueError(f"GT images[{index}] 必须是对象")
        missing = {"id", "width", "height"} - set(image)
        if missing:
            raise ValueError(f"GT images[{index}] 缺少字段: {sorted(missing)}")
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"GT image_id={image_id} 尺寸非法: {width}x{height}")
        if image_id in image_sizes:
            raise ValueError(f"GT image_id 重复: {image_id}")
        image_sizes[image_id] = (width, height)
    return image_sizes


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for path, label in ((args.pred, "预测"), (args.project_config, "项目配置")):
        if not path.exists():
            logger.error("%s文件不存在: %s", label, path)
            return 1
    if args.gt is not None and not args.gt.exists():
        logger.error("GT 文件不存在: %s", args.gt)
        return 1

    try:
        config = load_config(args.project_config)
        category_mapping = config["task"]["dataset_category_mapping"]
        allowed_category_ids = {int(category_id) for category_id in category_mapping}
        records = load_coco_prediction_records(args.pred)
        image_sizes = _load_image_sizes(args.gt) if args.gt is not None else None
        summary = validate_coco_prediction_records(
            records,
            allowed_category_ids=allowed_category_ids,
            image_sizes=image_sizes,
        )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        logger.error("预测文件不符合公共契约: %s", error)
        return 1

    logger.info(
        "PASS: detections=%d, images=%d, categories=%d",
        summary["detections"],
        summary["images_with_predictions"],
        summary["categories_with_predictions"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
