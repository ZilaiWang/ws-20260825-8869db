"""官方评估指标测试。"""

import pytest
from rsdet.evaluation.official_metric import evaluate_predictions, _compute_iou, IOU_THRESHOLDS


def _make_gt(image_id, bbox_xyxy, category_id=0):
    return {"bbox_xyxy": bbox_xyxy, "category_id": category_id}


def _make_pred(image_id, bbox_xyxy, score, category_id=0):
    return {"bbox_xyxy": bbox_xyxy, "score": score, "category_id": category_id}


class TestOfficialMetric:

    def test_one_pred_matches_one_gt(self):
        """一个预测正确匹配一个 GT: TP=1, FP=0, FN=0"""
        gt = {1: [_make_gt(1, [10, 10, 100, 100], 0)]}
        pred = {1: [_make_pred(1, [10, 10, 100, 100], 0.9, 0)]}
        result = evaluate_predictions(gt, pred, ["ship"])
        assert result.per_class["ship"].tp == 1
        assert result.per_class["ship"].fp == 0
        assert result.per_class["ship"].fn == 0

    def test_two_preds_one_gt_high_score_tp(self):
        """同一 GT 有两个预测框: 高分 TP, 低分 FP"""
        gt = {1: [_make_gt(1, [10, 10, 100, 100], 0)]}
        pred = {
            1: [
                _make_pred(1, [10, 10, 100, 100], 0.9, 0),
                _make_pred(1, [12, 12, 98, 98], 0.5, 0),
            ]
        }
        result = evaluate_predictions(gt, pred, ["ship"])
        assert result.per_class["ship"].tp == 1
        assert result.per_class["ship"].fp == 1
        assert result.per_class["ship"].fn == 0

    def test_pred_completely_mismatched(self):
        """预测框完全不匹配: FP=1, FN=1"""
        gt = {1: [_make_gt(1, [10, 10, 100, 100], 0)]}
        pred = {1: [_make_pred(1, [500, 500, 600, 600], 0.9, 0)]}
        result = evaluate_predictions(gt, pred, ["ship"])
        assert result.per_class["ship"].fp == 1
        assert result.per_class["ship"].fn == 1
        assert result.per_class["ship"].tp == 0

    def test_vehicle_iou_0_40_should_match(self):
        """vehicle IoU=0.40: 应视为匹配（阈值 0.35）"""
        # 构造 IoU ≈ 0.40: box_a=10×10, box_b 偏移使交集=4
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 2)]}  # vehicle
        # 偏移 20, 大小 100: 交集 ≈ 80*80=6400, 并集=10000+10000-6400=13600 → IoU≈0.47
        # 偏移 40: 交集=60*60=3600, 并集=20000-3600=16400, IoU≈0.22
        # 30: 交集=70*70=4900, 并集=20000-4900=15100, IoU≈0.32
        # 25: 交集=75*75=5625, 并集=20000-5625=14375, IoU≈0.391
        pred = {1: [_make_pred(1, [25, 25, 125, 125], 0.9, 2)]}
        result = evaluate_predictions(gt, pred, ["ship", "aircraft", "vehicle"])
        # vehicle cid=2, IoU threshold=0.35, IoU ≈ 0.39, 应匹配
        iou = _compute_iou([0, 0, 100, 100], [25, 25, 125, 125])
        assert iou >= IOU_THRESHOLDS["vehicle"], f"IoU={iou:.4f} >= {IOU_THRESHOLDS['vehicle']}"
        assert result.per_class["vehicle"].tp == 1

    def test_ship_iou_0_40_should_not_match(self):
        """ship IoU=0.40: 不应匹配（阈值 0.50）"""
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 0)]}  # ship
        pred = {1: [_make_pred(1, [25, 25, 125, 125], 0.9, 0)]}
        iou = _compute_iou([0, 0, 100, 100], [25, 25, 125, 125])
        assert iou < IOU_THRESHOLDS["ship"], f"IoU={iou:.4f} < {IOU_THRESHOLDS['ship']}"
        result = evaluate_predictions(gt, pred, ["ship"])
        assert result.per_class["ship"].fp == 1
        assert result.per_class["ship"].fn == 1

    def test_ship_iou_0_55_should_match(self):
        """ship IoU=0.55: 应匹配（阈值 0.50）"""
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 0)]}  # ship
        pred = {1: [_make_pred(1, [12, 12, 112, 112], 0.9, 0)]}
        iou = _compute_iou([0, 0, 100, 100], [12, 12, 112, 112])
        assert iou >= IOU_THRESHOLDS["ship"], f"IoU={iou:.4f} >= {IOU_THRESHOLDS['ship']}"
        result = evaluate_predictions(gt, pred, ["ship"])
        assert result.per_class["ship"].tp == 1

    def test_no_pred_has_gt(self):
        """无预测但有 GT: FN 正确"""
        gt = {1: [_make_gt(1, [10, 10, 100, 100], 0)]}
        pred = {}
        result = evaluate_predictions(gt, pred, ["ship"])
        assert result.per_class["ship"].fn == 1
        assert result.per_class["ship"].tp == 0
        assert result.per_class["ship"].fp == 0

    def test_pred_no_gt(self):
        """有预测但无 GT: FP 正确"""
        gt = {}
        pred = {1: [_make_pred(1, [10, 10, 100, 100], 0.9, 0)]}
        result = evaluate_predictions(gt, pred, ["ship"])
        assert result.per_class["ship"].fp == 1
        assert result.per_class["ship"].tp == 0
        assert result.per_class["ship"].fn == 0

    def test_score_order_affects_matching(self):
        """分数排序改变匹配归属: 验证降序 greedy matching"""
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 0)]}
        # 两个预测都匹配，但高分先匹配
        pred = {
            1: [
                _make_pred(1, [10, 10, 110, 110], 0.9, 0),
                _make_pred(1, [5, 5, 95, 95], 0.99, 0),  # 高分
            ]
        }
        result = evaluate_predictions(gt, pred, ["ship"])
        # 0.99 评分先匹配（得分高者优先），0.9评分成为FP
        assert result.per_class["ship"].tp == 1
        assert result.per_class["ship"].fp == 1


class TestIoU:

    def test_perfect_overlap(self):
        assert _compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_no_overlap(self):
        assert _compute_iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0

    def test_empty_box(self):
        """空框标注。"""
        assert _compute_iou([0, 0, 0, 0], [0, 0, 100, 100]) == 0.0
