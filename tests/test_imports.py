"""核心模块导入和最小接口测试。"""

import importlib

import pytest


@pytest.mark.parametrize(
    "module_name",
    [
        "rsdet",
        "rsdet.contracts",
        "rsdet.data.datasets",
        "rsdet.data.manifests",
        "rsdet.data.splits",
        "rsdet.engine.predictor",
        "rsdet.engine.trainer",
        "rsdet.evaluation.official_metric",
        "rsdet.evaluation.runtime",
        "rsdet.models.base",
        "rsdet.models.registry",
        "rsdet.predictions",
        "rsdet.postprocess.calibration",
        "rsdet.postprocess.nms",
        "rsdet.postprocess.tile_fusion",
        "rsdet.tiling.coordinates",
        "rsdet.tiling.slicer",
        "rsdet.utils.config",
        "rsdet.utils.logging",
        "rsdet.utils.seed",
        "rsdet.visualization.detection_visualizer",
    ],
)
def test_core_module_import(module_name: str) -> None:
    """所有核心模块都能在无 GPU、无数据条件下导入。"""
    assert importlib.import_module(module_name)


def test_contracts_and_dummy_detector() -> None:
    """公共结构和测试检测器可构建。"""
    from rsdet.contracts import (
        AnnotationRecord,
        ImageRecord,
        InferenceSample,
        Prediction,
        TileRecord,
    )
    from rsdet.models.base import BaseDetector
    from rsdet.models.registry import DummyDetector, build_model, list_models

    assert ImageRecord(1, "test.jpg", 1000, 1000).image_id == 1
    assert AnnotationRecord(1, 1, 0, [10, 10, 100, 100]).category_id == 0
    assert InferenceSample(1, None, 1000, 1000).width == 1000
    assert Prediction(1, [], [], []).image_id == 1
    assert TileRecord(0, 1, 0, 0, 512, 512).tile_id == 0
    assert list_models()["dummy"] is DummyDetector
    assert isinstance(build_model("dummy", {}), BaseDetector)


def test_runtime_and_seed_helpers() -> None:
    """计时和随机种子工具可运行。"""
    from rsdet.evaluation.runtime import RuntimeBreakdown, timed_block
    from rsdet.utils.seed import set_seed

    runtime = RuntimeBreakdown()
    with timed_block(runtime, "model"):
        pass
    assert runtime.model >= 0
    set_seed(42)
