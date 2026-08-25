"""单图与大图滑窗推理核心。"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from rsdet.contracts import InferenceSample, Prediction
from rsdet.engine.predictor import predict_batches
from rsdet.evaluation.runtime import RuntimeBreakdown, timed_block
from rsdet.postprocess.tile_fusion import fuse_tile_predictions
from rsdet.tiling.slicer import generate_tiles


def predict_image(
    detector: Any,
    *,
    image_id: int,
    image: np.ndarray,
    batch_size: int,
    tiling_config: Mapping[str, Any],
    runtime: RuntimeBreakdown,
) -> Prediction:
    """对一张已读入内存的 RGB 图执行直接或滑窗推理。"""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image 必须为 HxWx3 RGB 数组")
    if batch_size <= 0:
        raise ValueError("batch_size 必须 > 0")
    height, width = image.shape[:2]
    tile_size = int(tiling_config.get("tile_size", 1280))
    overlap = int(tiling_config.get("overlap", 256))
    use_tiling = bool(tiling_config.get("enabled", True)) and (
        bool(tiling_config.get("force", False)) or width > tile_size or height > tile_size
    )
    if not use_tiling:
        sample = InferenceSample(image_id, image, width, height)
        with timed_block(runtime, "model"):
            return predict_batches(
                detector,
                [sample],
                batch_size=1,
                allowed_category_ids=range(25),
            )[0]

    with timed_block(runtime, "tiling"):
        raw_tiles = generate_tiles(width, height, tile_size, overlap)
        tiles = [replace(tile, parent_image_id=image_id) for tile in raw_tiles]
        samples = [
            InferenceSample(
                tile.tile_id,
                np.ascontiguousarray(
                    image[
                        tile.y_offset : tile.y_offset + tile.height,
                        tile.x_offset : tile.x_offset + tile.width,
                    ]
                ),
                tile.width,
                tile.height,
                {
                    "parent_image_id": image_id,
                    "x_offset": tile.x_offset,
                    "y_offset": tile.y_offset,
                },
            )
            for tile in tiles
        ]
    with timed_block(runtime, "model"):
        tile_predictions = predict_batches(
            detector,
            samples,
            batch_size=batch_size,
            allowed_category_ids=range(25),
        )
    with timed_block(runtime, "postprocess"):
        return fuse_tile_predictions(
            tile_predictions,
            tiles,
            parent_image_id=image_id,
            image_width=width,
            image_height=height,
            fine_nms_iou=float(tiling_config.get("fine_nms_iou", 0.55)),
            coarse_nms_iou=(
                None
                if tiling_config.get("coarse_nms_iou") is None
                else float(tiling_config.get("coarse_nms_iou", 0.85))
            ),
            max_detections=int(tiling_config.get("max_detections", 2000)),
        )


__all__ = ["predict_image"]
