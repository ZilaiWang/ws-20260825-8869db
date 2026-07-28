"""端到端大图推理 pipeline：切片 → 推理 → 坐标恢复 → 融合 → 导出。

不绑定任何具体检测框架 (无 import ultralytics)。
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from rsdet.contracts import InferenceSample, Prediction, TileRecord
from rsdet.engine.predictor import predict_batches
from rsdet.models.base import BaseDetector
from rsdet.postprocess.tile_fusion import fuse_tile_predictions
from rsdet.tiling.slicer import generate_tiles


@dataclass
class PipelineConfig:
    """大图推理配置。"""

    tile_size: int = 1024
    overlap: int = 128
    batch_size: int = 16
    score_threshold: float = 0.0
    iou_thresholds: Dict[int, float] | None = None  # None = 使用默认三大类阈值


@dataclass
class PipelineTiming:
    """端到端推理耗时拆分。"""

    pipeline_s: float = 0.0       # 图在内存中 → 融合完成（不含读图）
    model_only_s: float = 0.0     # 纯 predict_batches 耗时
    tiling_s: float = 0.0         # 切片 + 裁图 + 构造 InferenceSample 耗时
    fusion_s: float = 0.0          # 融合耗时
    n_tiles: int = 0
    n_detections: int = 0
    peak_vram_gb: float = 0.0     # 峰值显存 (GiB)，仅在 torch 可用时回填

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pipeline_s": round(self.pipeline_s, 4),
            "model_only_s": round(self.model_only_s, 4),
            "tiling_s": round(self.tiling_s, 4),
            "fusion_s": round(self.fusion_s, 4),
            "n_tiles": self.n_tiles,
            "n_detections": self.n_detections,
            "peak_vram_gb": round(self.peak_vram_gb, 2),
        }


def _peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / (1024**3))
    except ImportError:
        pass
    return 0.0


def _reset_vram_peak() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def _extract_tile_image(
    full_image: np.ndarray,
    tile: TileRecord,
) -> np.ndarray:
    """从大图 numpy 数组中裁出 tile 对应的子图。

    full_image 形状: (H, W, 3) 或 (H, W)，uint8。
    返回保证有 3 个通道。
    """
    y1 = tile.y_offset
    y2 = y1 + tile.height
    x1 = tile.x_offset
    x2 = x1 + tile.width
    patch = full_image[y1:y2, x1:x2]
    if patch.ndim == 2:
        patch = np.stack([patch] * 3, axis=-1)
    elif patch.shape[2] == 1:
        patch = np.broadcast_to(patch, (patch.shape[0], patch.shape[1], 3))
    elif patch.shape[2] >= 4:
        # RGBA 或其他多通道 → 取前 3 通道 (RGB)
        patch = patch[:, :, :3]
    return patch.copy()


def run_pipeline(
    image: np.ndarray,
    detector: BaseDetector,
    *,
    config: PipelineConfig | None = None,
    parent_image_id: int = 0,
    tile_metadata_fn: Any | None = None,
) -> tuple[Prediction, PipelineTiming]:
    """对大图运行完整推理流水线。

    流程:
        1. generate_tiles() → 切片列表
        2. 逐 tile 裁图 → 构造 InferenceSample
        3. predict_batches() → 各 tile 的 Prediction
        4. fuse_tile_predictions() → 全局 Prediction
        5. 计时各阶段

    Args:
        image: 大图 numpy 数组 (H, W, 3) uint8 RGB。
        detector: 已加载的检测器实例。
        config: pipeline 配置；默认使用 PipelineConfig()。
        parent_image_id: 原图 image_id，填入输出 Prediction。
        tile_metadata_fn: 可选回调 (tile: TileRecord) → dict，
            为每个 tile 生成额外的 metadata（如 mock 真值注入）。

    Returns:
        (融合后的全局 Prediction, PipelineTiming)
    """
    if config is None:
        config = PipelineConfig()

    h, w = image.shape[:2]

    timing = PipelineTiming()

    # -------- 切片 --------
    t0 = time.perf_counter()
    tiles = generate_tiles(
        image_width=w,
        image_height=h,
        tile_size=config.tile_size,
        overlap=config.overlap,
    )
    timing.n_tiles = len(tiles)

    # 构造 InferenceSample 列表
    samples: List[InferenceSample] = []
    for tile in tiles:
        tile.image_id = tile.tile_id
        tile.parent_image_id = parent_image_id
        patch = _extract_tile_image(image, tile)
        meta: Dict[str, Any] = {
            "tile_x_offset": tile.x_offset,
            "tile_y_offset": tile.y_offset,
            "tile_width": tile.width,
            "tile_height": tile.height,
        }
        if tile_metadata_fn is not None:
            extra = tile_metadata_fn(tile)
            if extra:
                meta.update(extra)
        samples.append(
            InferenceSample(
                image_id=tile.tile_id,
                image=patch,
                width=tile.width,
                height=tile.height,
                metadata=meta,
            )
        )
    timing.tiling_s = time.perf_counter() - t0

    # -------- 推理 --------
    _reset_vram_peak()
    t0 = time.perf_counter()
    tile_predictions = predict_batches(
        detector,
        samples,
        batch_size=config.batch_size,
    )
    timing.model_only_s = time.perf_counter() - t0
    timing.peak_vram_gb = _peak_vram_gb()

    # -------- 融合 --------
    t0 = time.perf_counter()
    fused = fuse_tile_predictions(
        tile_predictions,
        tiles,
        image_width=w,
        image_height=h,
        parent_image_id=parent_image_id,
        score_threshold=config.score_threshold,
        iou_thresholds=config.iou_thresholds,
    )
    timing.fusion_s = time.perf_counter() - t0

    timing.pipeline_s = timing.tiling_s + timing.model_only_s + timing.fusion_s
    timing.n_detections = len(fused.boxes_xyxy)

    return fused, timing
