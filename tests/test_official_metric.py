"""官方评估指标测试。"""

import pytest

from rsdet.evaluation.official_metric import (
    IOU_THRESHOLDS,
    _compute_iou,
    evaluate_predictions,
    evaluate_predictions_with_trace,
    evaluate_ranking_metrics,
)


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

    def test_fine_categories_are_aggregated_after_matching(self):
        """细类分别匹配后归并，category_id=24 使用 vehicle 的 0.35 阈值。"""
        mapping = {0: "ship", 4: "aircraft", 24: "vehicle"}
        gt = {
            1: [_make_gt(1, [0, 0, 100, 100], 0)],
            2: [_make_gt(2, [0, 0, 100, 100], 24)],
        }
        pred = {
            1: [_make_pred(1, [0, 0, 100, 100], 0.9, 0)],
            2: [_make_pred(2, [25, 25, 125, 125], 0.8, 24)],
        }
        result = evaluate_predictions(gt, pred, category_mapping=mapping)
        assert result.per_class["ship"].tp == 1
        assert result.per_class["vehicle"].tp == 1
        assert result.details["tp"] == 2

    def test_wrong_fine_category_in_same_coarse_class_is_fp_and_fn(self):
        """框完全重合但飞机型号错误：预测为 FP，原 GT 为 FN。"""
        mapping = {4: "aircraft", 5: "aircraft"}
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 4)]}
        pred = {1: [_make_pred(1, [0, 0, 100, 100], 0.99, 5)]}

        result = evaluate_predictions(gt, pred, ["aircraft"], category_mapping=mapping)

        assert result.per_class["aircraft"].tp == 0
        assert result.per_class["aircraft"].fp == 1
        assert result.per_class["aircraft"].fn == 1
        assert result.recall == 0.0
        assert result.fdr == 1.0

    def test_custom_iou_thresholds_are_applied(self):
        """CLI 配置可把官方 IoU 阈值显式传给评估核心。"""
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 0)]}
        pred = {1: [_make_pred(1, [25, 25, 125, 125], 0.9, 0)]}

        result = evaluate_predictions(
            gt,
            pred,
            ["ship"],
            iou_thresholds={"ship": 0.35},
        )

        assert result.per_class["ship"].tp == 1
        assert result.details["iou_thresholds"] == {"ship": 0.35}

    def test_missing_fine_category_mapping_fails(self):
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 24)]}
        with pytest.raises(ValueError, match="缺少三大类映射"):
            evaluate_predictions(gt, {})

    def test_trace_is_same_source_as_official_counts_and_preserves_indices(self):
        mapping = {0: "ship", 1: "ship"}
        gt = {
            1: [
                _make_gt(1, [0, 0, 10, 10], 0),
                _make_gt(1, [20, 0, 30, 10], 1),
            ]
        }
        pred = {
            1: [
                _make_pred(1, [0, 0, 10, 10], 0.9, 0),
                _make_pred(1, [0, 0, 10, 10], 0.8, 0),
            ]
        }

        metrics, trace = evaluate_predictions_with_trace(
            gt,
            pred,
            ["ship"],
            mapping,
        )

        assert len(trace.matches) == metrics.details["tp"] == 1
        assert len(trace.unmatched_predictions) == metrics.details["fp"] == 1
        assert len(trace.unmatched_ground_truths) == metrics.details["fn"] == 1
        assert trace.matches[0].prediction_index == 0
        assert trace.matches[0].ground_truth_index == 0
        assert trace.unmatched_predictions[0].prediction_index == 1
        assert trace.unmatched_ground_truths[0].ground_truth_index == 1


class TestRankingMetrics:
    """官方评分方案 V1.6 排名口径（大类内细类指标简单平均）测试。"""

    def test_macro_is_fine_average_not_pooled(self):
        """大类 macro = 细类 Recall/FDR 的简单平均，与 pooled 不同。"""
        mapping = {0: "ship", 1: "ship"}  # 两个船细类
        # 细类 0：2 GT 全对；细类 1：1 GT 全漏 + 1 错检（TP=0, FP=1 → FDR 1.0）。
        # macro: Recall=(1+0)/2=0.5, FDR=(0+1.0)/2=0.5
        # pooled: Recall=2/3≈0.667, FDR=1/3≈0.333
        gt = {
            1: [
                _make_gt(1, [0, 0, 100, 100], 0),
                _make_gt(1, [200, 200, 300, 300], 0),
            ],
            2: [_make_gt(2, [0, 0, 100, 100], 1)],
        }
        pred = {
            1: [
                _make_pred(1, [0, 0, 100, 100], 0.9, 0),
                _make_pred(1, [200, 200, 300, 300], 0.8, 0),
            ],
            2: [_make_pred(2, [500, 500, 600, 600], 0.7, 1)],
        }
        ranking = evaluate_ranking_metrics(
            gt,
            pred,
            ["ship"],
            category_mapping=mapping,
            require_complete_taxonomy=False,
        )

        fine0 = ranking.per_fine[0]
        fine1 = ranking.per_fine[1]
        assert fine0.recall == 1.0
        assert fine0.fdr == 0.0
        assert fine1.recall == 0.0
        assert fine1.fdr == 1.0

        coarse = ranking.per_coarse["ship"]
        assert coarse.macro_recall == pytest.approx(0.5)
        assert coarse.macro_fdr == pytest.approx(0.5)
        # pooled 对照：Recall 与 FDR 都偏向"大户"细类 0
        assert coarse.pooled_recall == pytest.approx(2 / 3)
        assert coarse.pooled_fdr == pytest.approx(1 / 3)
        assert coarse.fine_count == 2
        assert coarse.fine_ids == [0, 1]

    def test_vehicle_single_fine_macro_equals_pooled(self):
        """车辆只有 FSC 一个细类：macro 与 pooled 相同。"""
        mapping = {24: "vehicle"}
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 24)]}
        pred = {1: [_make_pred(1, [25, 25, 125, 125], 0.9, 24)]}
        ranking = evaluate_ranking_metrics(gt, pred, ["vehicle"], category_mapping=mapping)
        coarse = ranking.per_coarse["vehicle"]
        assert coarse.macro_recall == pytest.approx(coarse.pooled_recall)
        assert coarse.macro_fdr == pytest.approx(coarse.pooled_fdr)

    def test_fine_identity_error_hurts_macro_recall(self):
        """细类身份错误（错型号）在官方口径下同时伤 Recall 与 FDR。"""
        mapping = {4: "aircraft", 5: "aircraft"}
        gt = {
            1: [_make_gt(1, [0, 0, 100, 100], 4)],
            2: [_make_gt(2, [0, 0, 100, 100], 5)],
        }
        pred = {
            # 细类 4 被错报成细类 5
            1: [_make_pred(1, [0, 0, 100, 100], 0.99, 5)],
            2: [_make_pred(2, [0, 0, 100, 100], 0.9, 5)],
        }
        ranking = evaluate_ranking_metrics(gt, pred, ["aircraft"], category_mapping=mapping)
        coarse = ranking.per_coarse["aircraft"]
        # 细类 4：TP=0, FN=1 → Recall 0；细类 5：TP=1, FP=1 → Recall 1, FDR 0.5
        assert coarse.macro_recall == pytest.approx(0.5)
        assert coarse.macro_fdr == pytest.approx(0.25)
        assert coarse.pooled_recall == pytest.approx(0.5)
        assert coarse.pooled_fdr == pytest.approx(0.5)

    def test_zero_gt_fine_classes_do_not_participate(self):
        """0-GT 细类不参与 macro 平均（present_in_gt_only 策略）。"""
        mapping = {0: "ship", 1: "ship", 2: "ship"}
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 0)]}
        pred = {1: [_make_pred(1, [0, 0, 100, 100], 0.9, 0)]}
        ranking = evaluate_ranking_metrics(
            gt,
            pred,
            ["ship"],
            category_mapping=mapping,
            require_complete_taxonomy=False,
        )

        assert ranking.per_coarse["ship"].fine_count == 1
        assert ranking.per_coarse["ship"].fine_ids == [0]
        assert ranking.details["fine_average_policy"] == ("present_in_gt_only_diagnostic")
        # 细类 1、2 无 GT 无预测，不产生记录也不参与平均
        assert 1 not in ranking.per_fine
        assert 2 not in ranking.per_fine

    def test_overall_is_fine_macro_average_across_coarse(self):
        """overall = 全部参与细类的简单平均（团队内部官方口径 Overall）。"""
        mapping = {0: "ship", 4: "aircraft"}
        gt = {
            1: [_make_gt(1, [0, 0, 100, 100], 0)],
            2: [_make_gt(2, [0, 0, 100, 100], 4)],
        }
        pred = {
            1: [_make_pred(1, [0, 0, 100, 100], 0.9, 0)],
            2: [_make_pred(2, [500, 500, 600, 600], 0.8, 4)],
        }
        ranking = evaluate_ranking_metrics(gt, pred, ["ship", "aircraft"], category_mapping=mapping)
        assert ranking.per_coarse["ship"].macro_recall == pytest.approx(1.0)
        assert ranking.per_coarse["aircraft"].macro_recall == pytest.approx(0.0)
        assert ranking.overall_recall == pytest.approx(0.5)
        assert ranking.details["aggregation_policy"] == ("official_ranking_v1_6_fine_macro_average")

    def test_ranking_matches_pooled_when_single_fine_per_coarse(self):
        """细类数=1 时 macro 与 pooled 完全一致（同源验证）。"""
        mapping = {0: "ship"}
        gt = {1: [_make_gt(1, [0, 0, 100, 100], 0)]}
        pred = {1: [_make_pred(1, [0, 0, 100, 100], 0.9, 0)]}
        result = evaluate_predictions(gt, pred, ["ship"], category_mapping=mapping)
        ranking = evaluate_ranking_metrics(gt, pred, ["ship"], category_mapping=mapping)
        assert ranking.per_coarse["ship"].macro_recall == pytest.approx(result.recall)
        assert ranking.per_coarse["ship"].macro_fdr == pytest.approx(result.fdr)


class TestIoU:
    def test_perfect_overlap(self):
        assert _compute_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0

    def test_no_overlap(self):
        assert _compute_iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0

    def test_empty_box(self):
        """空框标注。"""
        assert _compute_iou([0, 0, 0, 0], [0, 0, 100, 100]) == 0.0
