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
from rsdet.postprocess.global_aggregation import (
    GlobalObject,
    fuse_global_predictions,
    global_object_manifest,
)
from rsdet.postprocess.tile_fusion import fuse_tile_predictions
from rsdet.tiling.slicer import generate_tiles


@dataclass
class PipelineConfig:
    """大图推理配置。"""

    tile_size: int = 1024
    overlap: int = 128
    batch_size: int = 16
    score_threshold: float = 0.0      # tile 路径：融合前过滤；global 路径：聚合后过滤
    fine_nms_iou: float = 0.55        # tile_fusion 细类内 NMS 阈值
    coarse_nms_iou: float | None = 0.85  # tile_fusion 官方粗类 NMS 阈值（None 关闭）
    max_detections: int | None = None    # 最终保留检测数上限
    fusion: str = "tile"              # "tile" = 基线 tile_fusion；"global" = E 的全局聚合
    cluster_eps: float = 50.0         # 全局聚合 Spatial Gate 中心距离阈值
    merge_iou: float = 0.3            # 全局聚合语义门 IoU 阈值
    nms_iou: float = 0.5              # 全局聚合同类 NMS 阈值


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


def _filter_low_score(prediction: Prediction, threshold: float) -> Prediction:
    """融合前按分数阈值过滤检测（保留 score >= threshold 的框）。"""
    if threshold <= 0.0:
        return prediction
    keep = [i for i, s in enumerate(prediction.scores) if s >= threshold]
    return Prediction(
        prediction.image_id,
        [prediction.boxes_xyxy[i] for i in keep],
        [prediction.scores[i] for i in keep],
        [prediction.labels[i] for i in keep],
    )


def run_pipeline(
    image: np.ndarray,
    detector: BaseDetector,
    *,
    config: PipelineConfig | None = None,
    parent_image_id: int = 0,
    tile_metadata_fn: Any | None = None,
    collect_objects: bool = False,
) -> tuple[Prediction, PipelineTiming] | tuple[Prediction, PipelineTiming, List[GlobalObject]]:
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

    Args:
        image: 大图 numpy 数组 (H, W, 3) uint8 RGB。
        detector: 已加载的检测器实例。
        config: pipeline 配置；默认使用 PipelineConfig()。
        parent_image_id: 原图 image_id，填入输出 Prediction。
        tile_metadata_fn: 可选回调 (tile: TileRecord) → dict，
            为每个 tile 生成额外的 metadata（如 mock 真值注入）。
        collect_objects: 为 True 且 fusion="global" 时返回三元组，
            第三项为对象级清单（主线 2 输出契约）。

    Returns:
        collect_objects=False：(融合后的全局 Prediction, PipelineTiming)
        collect_objects=True：上述二元组外加 List[GlobalObject]。
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
    if config.fusion == "global":
        # global 路径：低分候选保留作证据，过滤放到聚合后（见 fuse_global_predictions）
        fused = fuse_global_predictions(
            tile_predictions,
            tiles,
            image_width=w,
            image_height=h,
            parent_image_id=parent_image_id,
            cluster_eps=config.cluster_eps,
            merge_iou=config.merge_iou,
            nms_iou=config.nms_iou,
            score_threshold=config.score_threshold,
            max_detections=config.max_detections,
        )
    else:
        if config.score_threshold > 0.0:
            tile_predictions = [
                _filter_low_score(p, config.score_threshold)
                for p in tile_predictions
            ]
        fused = fuse_tile_predictions(
            tile_predictions,
            tiles,
            image_width=w,
            image_height=h,
            parent_image_id=parent_image_id,
            fine_nms_iou=config.fine_nms_iou,
            coarse_nms_iou=config.coarse_nms_iou,
            max_detections=config.max_detections,
        )
    timing.fusion_s = time.perf_counter() - t0

    timing.pipeline_s = timing.tiling_s + timing.model_only_s + timing.fusion_s
    timing.n_detections = len(fused.boxes_xyxy)

    if collect_objects:
        if config.fusion != "global":
            raise ValueError("collect_objects=True requires config.fusion == 'global'")
        objects = global_object_manifest(
            tile_predictions,
            tiles,
            image_width=w,
            image_height=h,
            parent_image_id=parent_image_id,
            cluster_eps=config.cluster_eps,
            merge_iou=config.merge_iou,
            nms_iou=config.nms_iou,
            score_threshold=config.score_threshold,
            max_detections=config.max_detections,
        )
        return fused, timing, objects
    return fused, timing
