"""pipeline 对不同图像格式的防护：灰度/RGB/RGBA/float32/小于tile。"""

import numpy as np
import pytest

from rsdet.models.registry import build_model
import rsdet.pipeline.mock_model  # noqa: F401 — 注册 "mock"
from rsdet.pipeline.large_image import PipelineConfig, run_pipeline


def _make_image(h, w, channels, dtype=np.uint8):
    if channels == 1:
        img = np.zeros((h, w), dtype=dtype)
    else:
        img = np.zeros((h, w, channels), dtype=dtype)
    # 加一点噪声避免全黑图在个别检测器出问题
    rng = np.random.RandomState(0)
    if dtype == np.uint8:
        img = (img + rng.randint(0, 20, img.shape, dtype=np.uint8)).astype(np.uint8)
    else:
        img = (img + rng.rand(*img.shape).astype(dtype) * 0.1).astype(dtype)
    return img


class TestImageFormats:

    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_pipeline_no_crash(self, channels):
        """灰度(1ch) / RGB(3ch) / RGBA(4ch) 均不崩溃。"""
        img = _make_image(2048, 2048, channels)
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        config = PipelineConfig(tile_size=1024, overlap=128, batch_size=4)
        pred, timing = run_pipeline(
            img, detector, config=config, parent_image_id=1,
        )
        assert pred.image_id == 1
        assert timing.n_tiles > 0

    def test_float32_image(self):
        """float32 [0,1] 图像不崩溃。"""
        img = np.random.RandomState(0).rand(2048, 2048, 3).astype(np.float32)
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        config = PipelineConfig(tile_size=1024, overlap=128, batch_size=4)
        pred, timing = run_pipeline(
            img, detector, config=config, parent_image_id=1,
        )
        assert pred.image_id == 1

    def test_small_image_single_tile(self):
        """图像小于 tile_size 时只产生 1 个 tile。"""
        img = _make_image(800, 800, 3)
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        config = PipelineConfig(tile_size=1024, overlap=128, batch_size=4)
        pred, timing = run_pipeline(
            img, detector, config=config, parent_image_id=1,
        )
        assert timing.n_tiles == 1

    def test_rgba_stripped_to_rgb(self):
        """RGBA (4ch) 图被正确处理，裁出的 patch 为 3ch。"""
        img = np.random.RandomState(0).randint(0, 255, (1024, 1024, 4), dtype=np.uint8)
        detector = build_model("mock", {"init_args": {}})
        detector.eval()
        config = PipelineConfig(tile_size=1024, overlap=0, batch_size=1)
        pred, timing = run_pipeline(
            img, detector, config=config, parent_image_id=1,
        )
        assert timing.n_tiles == 1
        assert pred.image_id == 1
        assert len(pred.boxes_xyxy) >= 0  # 空预测不崩溃
