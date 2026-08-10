"""E 的端到端 pipeline 集成测试。"""

import pytest

from rsdet.models.registry import build_model
from rsdet.pipeline.large_image import (
    PipelineConfig,
    _extract_tile_image,
    run_pipeline,
)
from rsdet.pipeline.mock_model import MockDetector
from rsdet.tiling.slicer import generate_tiles
from rsdet.tiling.synthetic import generate_synthetic_scene

import numpy as np


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _tile_metadata_for_mock(scene):
    """为每个 tile 生成包含 gt_boxes 的 metadata（mock detector 使用）。"""
    from rsdet.contracts import TileRecord

    def _fn(tile: TileRecord) -> dict:
        gt_boxes = []
        for obj in scene.objects:
            if tile.tile_id not in obj.tile_ids:
                continue
            gx1, gy1, gx2, gy2 = obj.bbox
            lx1 = max(0.0, gx1 - tile.x_offset)
            ly1 = max(0.0, gy1 - tile.y_offset)
            lx2 = min(float(tile.width), gx2 - tile.x_offset)
            ly2 = min(float(tile.height), gy2 - tile.y_offset)
            if lx2 <= lx1 or ly2 <= ly1:
                continue
            gt_boxes.append(
                {
                    "bbox": [lx1, ly1, lx2, ly2],
                    "category_id": obj.category_id,
                    "score": 1.0,
                }
            )
        return {"gt_boxes": gt_boxes}

    return _fn


# ---------------------------------------------------------------------------
# 单元：tile 图像裁取
# ---------------------------------------------------------------------------

class TestExtractTileImage:
    def test_extract_preserves_content(self):
        """裁取后像素值与原图对应区域一致。"""
        full = np.random.RandomState(0).randint(0, 255, (100, 100, 3), dtype=np.uint8)
        tiles = generate_tiles(100, 100, 50, 10)
        tile = tiles[0]
        patch = _extract_tile_image(full, tile)
        assert patch.shape == (tile.height, tile.width, 3)
        np.testing.assert_array_equal(patch, full[tile.y_offset:tile.y_offset + tile.height,
                                                    tile.x_offset:tile.x_offset + tile.width])


# ---------------------------------------------------------------------------
# 集成：端到端 pipeline
# ---------------------------------------------------------------------------

class TestEndToEndPipeline:
    def test_mock_detector_registered(self):
        """mock 模型已在注册表中。"""
        detector = build_model("mock", {"init_args": {}})
        assert isinstance(detector, MockDetector)

    def test_small_synthetic_pipeline(self):
        """小图 pipeline 端到端跑通：无崩溃，格式正确。"""
        scene = generate_synthetic_scene(
            image_size=2048,
            tile_size=1024,
            overlap=128,
            num_ships=3,
            num_aircraft=5,
            num_vehicles=2,
            seed=42,
        )

        detector = build_model("mock", {"init_args": {}})
        detector.eval()

        config = PipelineConfig(
            tile_size=1024,
            overlap=128,
            batch_size=4,
            score_threshold=0.0,
        )

        prediction, timing = run_pipeline(
            scene.image,
            detector,
            config=config,
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )

        # 基本断言
        assert prediction.image_id == 0
        assert len(prediction.boxes_xyxy) == len(prediction.scores) == len(prediction.labels)
        assert timing.n_tiles > 0
        assert timing.pipeline_s > 0.0
        assert timing.model_only_s >= 0.0

    def test_mock_recovers_all_objects(self):
        """mock 检测器 + fusion 应恢复所有目标（无噪声时）。"""
        scene = generate_synthetic_scene(
            image_size=2048,
            tile_size=1024,
            overlap=256,
            num_ships=5,
            num_aircraft=10,
            num_vehicles=3,
            seed=42,
        )

        detector = build_model("mock", {"init_args": {}})
        detector.eval()

        config = PipelineConfig(tile_size=1024, overlap=256, batch_size=4)

        prediction, timing = run_pipeline(
            scene.image,
            detector,
            config=config,
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )

        # 无噪声 mock 应检测到所有 object
        # （但跨 tile 的同一 object 会被 NMS 去重）
        # 至少应 ≥ 每个目标至少 1 个检测
        n_objects = len(scene.objects)
        assert len(prediction.boxes_xyxy) >= n_objects, (
            f"检测数 {len(prediction.boxes_xyxy)} < 目标数 {n_objects}"
        )

    def test_empty_image_returns_empty(self):
        """完全无目标的图像应返回空 Prediction。"""
        # 生成只有 0 个目标的场景
        scene = generate_synthetic_scene(
            image_size=1024,
            tile_size=1024,
            overlap=128,
            num_ships=0,
            num_aircraft=0,
            num_vehicles=0,
            seed=42,
        )

        detector = build_model("mock", {"init_args": {}})
        detector.eval()

        config = PipelineConfig(tile_size=1024, overlap=128, batch_size=1)
        prediction, timing = run_pipeline(
            scene.image,
            detector,
            config=config,
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )

        assert len(prediction.boxes_xyxy) == 0
        assert len(prediction.scores) == 0
        assert len(prediction.labels) == 0

    def test_timing_fields_populated(self):
        """PipelineTiming 各字段均有值。"""
        scene = generate_synthetic_scene(
            image_size=2048, tile_size=1024, overlap=128, seed=99,
            num_ships=1, num_aircraft=2, num_vehicles=1,
        )

        detector = build_model("mock", {"init_args": {}})
        detector.eval()

        config = PipelineConfig(tile_size=1024, overlap=128, batch_size=4)
        _, timing = run_pipeline(
            scene.image,
            detector,
            config=config,
            tile_metadata_fn=_tile_metadata_for_mock(scene),
        )

        assert timing.pipeline_s > 0.0
        assert timing.model_only_s >= 0.0
        assert timing.tiling_s >= 0.0
        assert timing.fusion_s >= 0.0
        assert timing.n_tiles > 0
        assert timing.n_detections > 0
        d = timing.to_dict()
        assert "pipeline_s" in d
        assert "n_tiles" in d

    def test_model_replaceable(self):
        """pipeline 接受任何 BaseDetector 实例（用 dummy 测试）。"""
        detector = build_model("dummy", {"init_args": {}})
        scene = generate_synthetic_scene(
            image_size=1024, tile_size=1024, overlap=0, seed=0,
            num_ships=0, num_aircraft=0, num_vehicles=0,
        )
        config = PipelineConfig(tile_size=1024, overlap=0, batch_size=1)
        prediction, timing = run_pipeline(scene.image, detector, config=config)
        # dummy 返回空预测，不应崩溃
        assert len(prediction.boxes_xyxy) == 0

    def test_config_defaults(self):
        """PipelineConfig 默认值可用。"""
        config = PipelineConfig()
        assert config.tile_size == 1024
        assert config.overlap == 128
        assert config.batch_size == 16
        assert config.score_threshold == 0.0
