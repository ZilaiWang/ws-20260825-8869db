"""核心模块导入测试。"""

import pytest


def test_import_rsdet():
    """rsdet 包可导入。"""
    import rsdet
    assert rsdet.__version__ == "0.1.0"


def test_import_contracts():
    """contracts 模块可导入。"""
    from rsdet.contracts import ImageRecord, AnnotationRecord, Prediction, TileRecord

    img = ImageRecord(image_id=1, file_path="test.jpg", width=1000, height=1000)
    assert img.image_id == 1
    assert img.width == 1000

    ann = AnnotationRecord(
        annotation_id=1, image_id=1, category_id=0, bbox_xyxy=[10, 10, 100, 100]
    )
    assert ann.bbox_xyxy == [10, 10, 100, 100]

    pred = Prediction(image_id=1, boxes_xyxy=[], scores=[], labels=[])
    assert pred.image_id == 1

    tile = TileRecord(
        tile_id=0, parent_image_id=1, x_offset=0, y_offset=0, width=512, height=512
    )
    assert tile.tile_id == 0


def test_import_models():
    """models 模块可导入。"""
    from rsdet.models.base import BaseDetector
    from rsdet.models.registry import register_model, build_model, list_models, DummyDetector

    models = list_models()
    assert "dummy" in models
    assert models["dummy"] is DummyDetector

    detector = build_model("dummy", {})
    assert isinstance(detector, BaseDetector)


def test_import_tiling():
    """tiling 模块可导入。"""
    from rsdet.tiling.coordinates import xyxy_to_xywh, xywh_to_xyxy, tile_to_full, full_to_tile, clip_bbox
    from rsdet.tiling.slicer import generate_tiles

    result = xyxy_to_xywh([0, 0, 100, 200])
    assert result == [0, 0, 100, 200]

    tiles = generate_tiles(3000, 2000, 1024, 200)
    assert len(tiles) > 0


def test_import_evaluation():
    """evaluation 模块可导入。"""
    from rsdet.evaluation.official_metric import evaluate_predictions, PerClassMetrics, OverallMetrics
    from rsdet.evaluation.runtime import RuntimeBreakdown, timed_block

    m = PerClassMetrics(tp=5, fp=1, fn=2)
    assert m.recall == 5 / 7
    assert m.fdr == 1 / 6

    rt = RuntimeBreakdown()
    with timed_block(rt, "model"):
        pass
    assert rt.model >= 0


def test_import_postprocess():
    """postprocess 模块可导入。"""
    import rsdet.postprocess.nms
    import rsdet.postprocess.tile_fusion
    import rsdet.postprocess.calibration


def test_import_utils():
    """utils 模块可导入。"""
    from rsdet.utils.config import load_config, merge_configs
    from rsdet.utils.logging import setup_logging
    from rsdet.utils.seed import set_seed

    merged = merge_configs({"a": 1}, {"b": 2})
    assert merged["a"] == 1
    assert merged["b"] == 2

    set_seed(42)
