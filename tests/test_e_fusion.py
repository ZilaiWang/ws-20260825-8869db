"""E 的 tile 融合逻辑测试。"""

import pytest

from rsdet.contracts import Prediction, TileRecord
from rsdet.postprocess.tile_fusion import (
    DEFAULT_IOU_THRESHOLDS,
    _compute_ious,
    _nms_per_class,
    fuse_tile_predictions,
)

import numpy as np


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

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
) -> TileRecord:
    return TileRecord(
        tile_id=tile_id,
        parent_image_id=0,
        x_offset=x_offset,
        y_offset=y_offset,
        width=width,
        height=height,
    )


# ---------------------------------------------------------------------------
# IoU 计算
# ---------------------------------------------------------------------------

class TestIoUComputation:
    def test_perfect_overlap(self):
        """完全重叠 → IoU = 1.0。"""
        boxes = np.array([[10, 20, 110, 220]], dtype=np.float64)
        ious = _compute_ious(np.array([10, 20, 110, 220]), boxes)
        assert ious[0] == pytest.approx(1.0)

    def test_no_overlap(self):
        """不重叠 → IoU = 0.0。"""
        boxes = np.array([[200, 200, 300, 300]], dtype=np.float64)
        ious = _compute_ious(np.array([10, 20, 110, 220]), boxes)
        assert ious[0] == 0.0

    def test_partial_overlap(self):
        """部分重叠 → 0 < IoU < 1。"""
        boxes = np.array([[50, 50, 150, 250]], dtype=np.float64)
        ious = _compute_ious(np.array([10, 20, 110, 220]), boxes)
        assert 0.0 < ious[0] < 1.0

    def test_vectorized_multiple_boxes(self):
        """同时计算多框 IoU。"""
        box = np.array([10, 20, 110, 220])
        boxes = np.array(
            [
                [10, 20, 110, 220],  # 完全重叠
                [300, 300, 400, 400],  # 不重叠
                [50, 50, 150, 250],  # 部分重叠
            ],
            dtype=np.float64,
        )
        ious = _compute_ious(box, boxes)
        assert len(ious) == 3
        assert ious[0] == pytest.approx(1.0)
        assert ious[1] == 0.0
        assert 0.0 < ious[2] < 1.0


# ---------------------------------------------------------------------------
# NMS 单类逻辑
# ---------------------------------------------------------------------------

class TestNMS:
    def test_empty_boxes(self):
        """空输入返回空索引。"""
        idx = _nms_per_class(np.array([]), np.array([]), 0.5)
        assert len(idx) == 0

    def test_single_box(self):
        """单框直接保留。"""
        boxes = np.array([[10, 20, 110, 220]], dtype=np.float64)
        scores = np.array([0.9])
        idx = _nms_per_class(boxes, scores, 0.5)
        assert list(idx) == [0]

    def test_two_non_overlapping(self):
        """两个不重叠的框都保留。"""
        boxes = np.array([[10, 20, 110, 220], [300, 300, 400, 400]], dtype=np.float64)
        scores = np.array([0.9, 0.8])
        idx = _nms_per_class(boxes, scores, 0.5)
        assert sorted(idx.tolist()) == [0, 1]

    def test_two_heavily_overlapping(self):
        """两个高度重叠的框只保留高分者。"""
        boxes = np.array(
            [[10, 20, 110, 220], [15, 25, 105, 215]], dtype=np.float64
        )
        scores = np.array([0.9, 0.8])
        idx = _nms_per_class(boxes, scores, 0.5)
        assert list(idx) == [0]  # 高分保留

    def test_nms_respects_threshold(self):
        """高 IoU 阈值 → 保留更多框。"""
        # 两个框 IoU ≈ 0.92
        boxes = np.array(
            [[10, 20, 110, 220], [15, 25, 105, 215]], dtype=np.float64
        )
        scores = np.array([0.9, 0.8])
        # 阈值 0.95 > 真实 IoU → 都保留
        idx_high = _nms_per_class(boxes, scores, 0.95)
        assert len(idx_high) == 2
        # 阈值 0.5 < 真实 IoU → 只保留高分
        idx_low = _nms_per_class(boxes, scores, 0.5)
        assert len(idx_low) == 1


# ---------------------------------------------------------------------------
# 融合：坐标恢复 + 边界 + 过滤
# ---------------------------------------------------------------------------

class TestFusion:
    def test_single_tile_passthrough(self):
        """单个 tile 的局部框 + offset = 全局框，误差 0。"""
        tile_pred = _make_tile_prediction(
            tile_id=0,
            boxes=[[100, 200, 300, 400]],
            scores=[0.9],
            labels=[4],
        )
        tile_rec = _make_tile_record(
            tile_id=0, x_offset=500, y_offset=300, width=1024, height=1024
        )
        fused = fuse_tile_predictions(
            [tile_pred],
            [tile_rec],
            image_width=10000,
            image_height=10000,
            parent_image_id=0,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.boxes_xyxy[0] == [600.0, 500.0, 800.0, 700.0]
        assert fused.scores[0] == 0.9
        assert fused.labels[0] == 4
        assert fused.image_id == 0

    def test_overlap_dedup(self):
        """同一目标跨 2 tile → 融合后只有 1 框。"""
        # 目标全局坐标: [500, 500, 600, 600]，落在 tile0 和 tile1 重叠区
        # tile0: (0, 0) → 局部 [500, 500, 600, 600]
        # tile1: (100, 0) → 局部 [400, 500, 500, 600]
        # 注：tile_size=600, overlap=500 → stride=100
        pred0 = _make_tile_prediction(
            tile_id=0,
            boxes=[[500, 500, 600, 600]],
            scores=[0.9],
            labels=[4],
        )
        pred1 = _make_tile_prediction(
            tile_id=1,
            boxes=[[400, 500, 500, 600]],
            scores=[0.85],
            labels=[4],
        )
        tile0 = _make_tile_record(tile_id=0, x_offset=0, y_offset=0, width=600, height=600)
        tile1 = _make_tile_record(tile_id=1, x_offset=100, y_offset=0, width=600, height=600)

        fused = fuse_tile_predictions(
            [pred0, pred1],
            [tile0, tile1],
            image_width=10000,
            image_height=10000,
        )
        # 两个预测恢复后是同一个位置，NMS 应去重
        assert len(fused.boxes_xyxy) == 1

    def test_out_of_bounds_clipped(self):
        """越界框被裁剪到 [0, image_width]。"""
        # 目标完全在图像边缘外
        tile_pred = _make_tile_prediction(
            tile_id=0,
            boxes=[[9900, 9900, 10100, 10100]],
            scores=[0.9],
            labels=[4],
        )
        tile_rec = _make_tile_record(
            tile_id=0, x_offset=0, y_offset=0, width=10000, height=10000
        )
        fused = fuse_tile_predictions(
            [tile_pred],
            [tile_rec],
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 1
        box = fused.boxes_xyxy[0]
        # 裁剪后不应越界
        assert 0 <= box[0] <= 10000
        assert 0 <= box[1] <= 10000
        assert 0 <= box[2] <= 10000
        assert 0 <= box[3] <= 10000
        assert box[0] <= box[2]
        assert box[1] <= box[3]

    def test_empty_predictions(self):
        """所有 tile 无预测 → 返回空 Prediction。"""
        pred = _make_tile_prediction(tile_id=0)
        tile_rec = _make_tile_record()
        fused = fuse_tile_predictions(
            [pred],
            [tile_rec],
            image_width=10000,
            image_height=10000,
        )
        assert fused.image_id == 0
        assert len(fused.boxes_xyxy) == 0
        assert len(fused.scores) == 0
        assert len(fused.labels) == 0

    def test_vehicle_uses_lower_iou(self):
        """车辆 (category_id=24) 用 IoU 0.35 而非 0.50。"""
        # 两个车辆框 IoU ≈ 0.40: 用 0.35 应合并，用 0.50 则不合并
        # 全局坐标
        box_a = [500, 500, 600, 600]  # 100x100
        box_b = [520, 520, 620, 620]  # 100x100, shift by 20
        # IoU = (80*80) / (10000 + 10000 - 6400) = 6400 / 13600 ≈ 0.47
        # 实际上 offset=20: inter=(80*80)=6400, union=20000-6400=13600, IoU≈0.4706

        pred0 = _make_tile_prediction(
            tile_id=0, boxes=[box_a], scores=[0.9], labels=[24]
        )
        pred1 = _make_tile_prediction(
            tile_id=1, boxes=[box_b], scores=[0.85], labels=[24]
        )
        rec0 = _make_tile_record(tile_id=0, x_offset=0, y_offset=0, width=10000, height=10000)
        rec1 = _make_tile_record(tile_id=1, x_offset=0, y_offset=0, width=10000, height=10000)

        fused = fuse_tile_predictions(
            [pred0, pred1],
            [rec0, rec1],
            image_width=10000,
            image_height=10000,
        )
        # 车辆 IoU 阈值 0.35 < 0.47 → 应该合并
        assert len(fused.boxes_xyxy) == 1

    def test_cross_class_not_merged(self):
        """完全重叠但不同细类 → 保留 2 框。"""
        box = [500, 500, 600, 600]
        pred0 = _make_tile_prediction(
            tile_id=0, boxes=[box], scores=[0.9], labels=[5]  # C-130
        )
        pred1 = _make_tile_prediction(
            tile_id=1, boxes=[box], scores=[0.85], labels=[6]  # C-17
        )
        rec0 = _make_tile_record(tile_id=0, x_offset=0, y_offset=0, width=10000, height=10000)
        rec1 = _make_tile_record(tile_id=1, x_offset=0, y_offset=0, width=10000, height=10000)

        fused = fuse_tile_predictions(
            [pred0, pred1],
            [rec0, rec1],
            image_width=10000,
            image_height=10000,
        )
        assert len(fused.boxes_xyxy) == 2

    def test_score_threshold_filters_low_scores(self):
        """score < threshold 的框被丢弃。"""
        tile_pred = _make_tile_prediction(
            tile_id=0,
            boxes=[[100, 200, 300, 400], [500, 600, 700, 800]],
            scores=[0.9, 0.05],
            labels=[4, 5],
        )
        tile_rec = _make_tile_record()
        fused = fuse_tile_predictions(
            [tile_pred],
            [tile_rec],
            image_width=10000,
            image_height=10000,
            score_threshold=0.3,
        )
        assert len(fused.boxes_xyxy) == 1
        assert fused.scores[0] == 0.9

    def test_different_parents_uses_parent_image_id(self):
        """融合后 image_id 为指定的 parent_image_id。"""
        pred = _make_tile_prediction(
            tile_id=0, boxes=[[10, 10, 100, 100]], scores=[0.9], labels=[0]
        )
        rec = _make_tile_record()
        fused = fuse_tile_predictions(
            [pred], [rec],
            image_width=10000, image_height=10000,
            parent_image_id=42,
        )
        assert fused.image_id == 42

    def test_empty_after_score_filter(self):
        """所有框被分数过滤后返回空 Prediction。"""
        pred = _make_tile_prediction(
            tile_id=0,
            boxes=[[10, 10, 100, 100]],
            scores=[0.01],
            labels=[0],
        )
        rec = _make_tile_record()
        fused = fuse_tile_predictions(
            [pred], [rec],
            image_width=10000, image_height=10000,
            score_threshold=0.5,
        )
        assert len(fused.boxes_xyxy) == 0

    def test_raises_on_mismatched_lengths(self):
        """tile_predictions 和 tile_records 长度不一致时报错。"""
        with pytest.raises(ValueError):
            fuse_tile_predictions(
                [_make_tile_prediction(), _make_tile_prediction()],
                [_make_tile_record()],
                image_width=10000,
                image_height=10000,
            )

    def test_default_iou_thresholds_coverage(self):
        """DEFAULT_IOU_THRESHOLDS 覆盖全部 25 个细类。"""
        for cid in range(25):
            assert cid in DEFAULT_IOU_THRESHOLDS, f"category_id={cid} 缺失阈值"
        assert DEFAULT_IOU_THRESHOLDS[0] == 0.50
        assert DEFAULT_IOU_THRESHOLDS[4] == 0.50
        assert DEFAULT_IOU_THRESHOLDS[23] == 0.50
        assert DEFAULT_IOU_THRESHOLDS[24] == 0.35
