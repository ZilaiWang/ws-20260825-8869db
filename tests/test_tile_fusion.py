"""跨切片坐标恢复与分层去重测试。"""

from rsdet.contracts import Prediction, TileRecord
from rsdet.postprocess.tile_fusion import fuse_tile_predictions


def test_fusion_restores_coordinates_and_removes_duplicate() -> None:
    tiles = [
        TileRecord(0, 9, 0, 0, 600, 600),
        TileRecord(1, 9, 400, 0, 600, 600),
    ]
    predictions = [
        Prediction(0, [[450, 100, 550, 200]], [0.9], [4]),
        Prediction(1, [[50, 100, 150, 200]], [0.8], [4]),
    ]

    fused = fuse_tile_predictions(
        predictions,
        tiles,
        parent_image_id=9,
        image_width=1000,
        image_height=600,
    )

    assert fused.image_id == 9
    assert fused.boxes_xyxy == [[450.0, 100.0, 550.0, 200.0]]
    assert fused.scores == [0.9]


def test_coarse_nms_removes_different_fine_duplicate() -> None:
    tiles = [
        TileRecord(0, 1, 0, 0, 600, 600),
        TileRecord(1, 1, 400, 0, 600, 600),
    ]
    predictions = [
        Prediction(0, [[450, 100, 550, 200]], [0.7], [4]),
        Prediction(1, [[50, 100, 150, 200]], [0.9], [5]),
    ]

    fused = fuse_tile_predictions(
        predictions,
        tiles,
        parent_image_id=1,
        image_width=1000,
        image_height=600,
        coarse_nms_iou=0.85,
    )

    assert fused.labels == [5]
    assert fused.scores == [0.9]


def test_coarse_nms_can_be_disabled() -> None:
    tiles = [TileRecord(0, 1, 0, 0, 600, 600), TileRecord(1, 1, 400, 0, 600, 600)]
    predictions = [
        Prediction(0, [[450, 100, 550, 200]], [0.7], [4]),
        Prediction(1, [[50, 100, 150, 200]], [0.9], [5]),
    ]
    fused = fuse_tile_predictions(
        predictions,
        tiles,
        parent_image_id=1,
        image_width=1000,
        image_height=600,
        coarse_nms_iou=None,
    )
    assert fused.labels == [5, 4]
