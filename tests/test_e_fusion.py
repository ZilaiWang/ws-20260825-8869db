"""E 的 tile 融合测试（适配 master 规范版 tile_fusion API）。

新版 API：fuse_tile_predictions(tile_predictions, tiles, *,
parent_image_id, image_width, image_height,
fine_nms_iou=0.55, coarse_nms_iou=0.85, max_detections=None)

- fine NMS：细类内去重（默认 0.55）
- coarse NMS：官方三大类内、跨细类的近重复框合并（默认 0.85）
"""

import pytest

from rsdet.contracts import Prediction, TileRecord
from rsdet.postprocess.tile_fusion import fuse_tile_predictions


def _make_tile_prediction(
    tile_id: int = 0,
    boxes: list | None = None,
    scores: list | None = None,
    labels: list | None = None,
) -> Prediction:
    return Prediction(
        image_id=tile_id,
        boxes_xyxy=boxes or [],
        scores=scores or [],
        labels=labels or [],
    )


def _make_tile_record(
    tile_id: int = 0,
    x_offset: int = 0,
    y_offset: int = 0,
    width: int = 1024,
    height: int = 1024,
    parent_image_id: int = 0,
) -> TileRecord:
    return TileRecord(
        tile_id=tile_id,
        parent_image_id=parent_image_id,
        x_offset=x_offset,
        y_offset=y_offset,
        width=width,
        height=height,
    )


class TestFusionCoordinateRestore:
    def test_single_tile_passthrough(self):
        """单个 tile 的局部框 + offset = 全局框，误差 0。"""
        tile_pred = _make_tile_prediction(
            tile_id=0, boxes=[[100, 200, 300, 400]], scores=[0.9], labels=[4]
        )
        tile_rec = _make_tile_record(tile_id=0, x_offset=500, y_offset=300)
        fused = fuse_tile_predictions(
            [tile_pred],
            [tile_rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert fused.boxes_xyxy[0] == [600.0, 500.0, 800.0, 700.0]
        assert fused.scores[0] == 0.9
        assert fused.labels[0] == 4
        assert fused.image_id == 0

    def test_out_of_bounds_clipped(self):
        """越界框被裁剪到 [0, image_width] × [0, image_height]。"""
        tile_pred = _make_tile_prediction(
            tile_id=0, boxes=[[9900, 9900, 10100, 10100]], scores=[0.9], labels=[4]
        )
        tile_rec = _make_tile_record(tile_id=0, x_offset=0, y_offset=0, width=10000, height=10000)
        fused = fuse_tile_predictions(
            [tile_pred],
            [tile_rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        box = fused.boxes_xyxy[0]
        assert 0 <= box[0] <= 10000 and 0 <= box[1] <= 10000
        assert 0 <= box[2] <= 10000 and 0 <= box[3] <= 10000
        assert box[0] <= box[2] and box[1] <= box[3]

    def test_fully_out_of_bounds_discarded(self):
        """裁剪后退化为零面积框 → 被丢弃。"""
        tile_pred = _make_tile_prediction(
            tile_id=0, boxes=[[20000, 20000, 21000, 21000]], scores=[0.9], labels=[4]
        )
        tile_rec = _make_tile_record()
        fused = fuse_tile_predictions(
            [tile_pred],
            [tile_rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 0

    def test_empty_predictions(self):
        """所有 tile 无预测 → 返回空 Prediction。"""
        pred = _make_tile_prediction(tile_id=0)
        tile_rec = _make_tile_record()
        fused = fuse_tile_predictions(
            [pred],
            [tile_rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert fused.image_id == 0
        assert len(fused.boxes_xyxy) == 0
        assert len(fused.scores) == 0
        assert len(fused.labels) == 0

    def test_parent_image_id_passthrough(self):
        """融合后 image_id 为指定的 parent_image_id。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[10, 10, 100, 100]], scores=[0.9], labels=[0]
        )
        rec = _make_tile_record(parent_image_id=42)
        fused = fuse_tile_predictions(
            [pred],
            [rec],
            parent_image_id=42,
            image_width=10000,
            image_height=10000,
        )
        assert fused.image_id == 42


class TestFusionNMS:
    def test_overlap_dedup_same_fine_class(self):
        """同一目标跨 2 tile（同细类）→ 细类 NMS 合并为 1 框。"""
        # 目标全局 [500, 500, 600, 600]；tile1 右移 100 → 局部框相应左移
        pred0 = _make_tile_prediction(
            tile_id=0, boxes=[[500, 500, 600, 600]], scores=[0.9], labels=[4]
        )
        pred1 = _make_tile_prediction(
            tile_id=1, boxes=[[400, 500, 500, 600]], scores=[0.85], labels=[4]
        )
        tile0 = _make_tile_record(tile_id=0, x_offset=0, y_offset=0, width=600, height=600)
        tile1 = _make_tile_record(tile_id=1, x_offset=100, y_offset=0, width=600, height=600)

        fused = fuse_tile_predictions(
            [pred0, pred1],
            [tile0, tile1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.scores[0] == 0.9  # 保留高分

    def test_fine_class_kept_within_coarse(self):
        """同粗类内不同细类、完全重叠 → coarse NMS(0.85) 合并为 1 框。"""
        box = [500, 500, 600, 600]
        pred0 = _make_tile_prediction(tile_id=0, boxes=[box], scores=[0.9], labels=[5])  # C-130
        pred1 = _make_tile_prediction(tile_id=1, boxes=[box], scores=[0.85], labels=[6])  # C-17
        rec0 = _make_tile_record(tile_id=0)
        rec1 = _make_tile_record(tile_id=1)
        fused = fuse_tile_predictions(
            [pred0, pred1],
            [rec0, rec1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.labels[0] == 5  # 高分者获胜

    def test_cross_coarse_class_not_merged(self):
        """跨粗类（舰船 vs 飞机）完全重叠 → 不合并，保留 2 框。"""
        box = [500, 500, 600, 600]
        pred0 = _make_tile_prediction(tile_id=0, boxes=[box], scores=[0.9], labels=[0])  # 舰船
        pred1 = _make_tile_prediction(tile_id=1, boxes=[box], scores=[0.85], labels=[5])  # 飞机
        rec0 = _make_tile_record(tile_id=0)
        rec1 = _make_tile_record(tile_id=1)
        fused = fuse_tile_predictions(
            [pred0, pred1],
            [rec0, rec1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 2

    def test_coarse_nms_can_be_disabled(self):
        """coarse_nms_iou=None → 跨细类近重复框不被合并。"""
        box = [500, 500, 600, 600]
        pred0 = _make_tile_prediction(tile_id=0, boxes=[box], scores=[0.9], labels=[5])
        pred1 = _make_tile_prediction(tile_id=1, boxes=[box], scores=[0.85], labels=[6])
        rec0 = _make_tile_record(tile_id=0)
        rec1 = _make_tile_record(tile_id=1)
        fused = fuse_tile_predictions(
            [pred0, pred1],
            [rec0, rec1],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
            coarse_nms_iou=None,
        )
        assert len(fused.boxes_xyxy) == 2

    def test_max_detections_limits_output(self):
        """max_detections 限制最终输出框数。"""
        pred0 = _make_tile_prediction(
            tile_id=0,
            boxes=[[10, 10, 100, 100], [200, 200, 300, 300], [400, 400, 500, 500]],
            scores=[0.9, 0.7, 0.5],
            labels=[4, 4, 4],
        )
        rec = _make_tile_record(tile_id=0)
        fused = fuse_tile_predictions(
            [pred0],
            [rec],
            parent_image_id=0,
            image_width=10000,
            image_height=10000,
            max_detections=1,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.scores[0] == 0.9  # 保留最高分


class TestFusionValidation:
    def test_raises_on_mismatched_lengths(self):
        """tile_predictions 和 tile_records 长度不一致时报错。"""
        with pytest.raises(ValueError):
            fuse_tile_predictions(
                [_make_tile_prediction(), _make_tile_prediction()],
                [_make_tile_record()],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_duplicate_tile_id(self):
        """重复 tile_id 报错。"""
        with pytest.raises(ValueError):
            fuse_tile_predictions(
                [_make_tile_prediction(tile_id=0), _make_tile_prediction(tile_id=0)],
                [_make_tile_record(tile_id=0), _make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_prediction_image_mismatch(self):
        """prediction.image_id 与 tile.tile_id 不一致报错。"""
        with pytest.raises(ValueError):
            fuse_tile_predictions(
                [_make_tile_prediction(tile_id=99)],
                [_make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_parent_mismatch(self):
        """tile.parent_image_id 与 parent_image_id 不一致报错。"""
        rec = _make_tile_record(parent_image_id=7)
        with pytest.raises(ValueError):
            fuse_tile_predictions(
                [_make_tile_prediction(tile_id=0)],
                [rec],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_invalid_score(self):
        """score 越界 [0,1] 报错。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[10, 10, 100, 100]], scores=[1.5], labels=[4]
        )
        with pytest.raises(ValueError):
            fuse_tile_predictions(
                [pred],
                [_make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )

    def test_raises_on_invalid_label(self):
        """label 非整数报错。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[10, 10, 100, 100]], scores=[0.9], labels=["4"]
        )
        with pytest.raises(ValueError):
            fuse_tile_predictions(
                [pred],
                [_make_tile_record(tile_id=0)],
                parent_image_id=0,
                image_width=10000,
                image_height=10000,
            )
