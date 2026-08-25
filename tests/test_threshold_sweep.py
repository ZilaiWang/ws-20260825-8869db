"""全局阈值扫描纯函数测试。"""

import math

import pytest

from rsdet.evaluation.official_metric import OverallMetrics, RankingMetrics
from rsdet.postprocess.calibration import (
    ThresholdSweepPoint,
    build_threshold_grid,
    filter_predictions_by_score,
    select_operating_points,
    sweep_global_thresholds,
)


def _point(threshold: float, recall: float, fdr: float) -> ThresholdSweepPoint:
    return ThresholdSweepPoint(
        threshold=threshold,
        detections_kept=0,
        metrics=OverallMetrics(recall=recall, fdr=fdr),
        ranking_metrics=RankingMetrics(overall_recall=recall, overall_fdr=fdr),
    )


def test_decimal_grid_and_inclusive_filter() -> None:
    """0.30 不因浮点漂移被误删，等于阈值的分数保留。"""
    grid = build_threshold_grid(0.28, 0.32, 0.01)
    assert grid == [0.28, 0.29, 0.3, 0.31, 0.32]
    predictions = {
        1: [
            {"bbox_xyxy": [0, 0, 1, 1], "category_id": 0, "score": 0.30},
            {"bbox_xyxy": [1, 1, 2, 2], "category_id": 0, "score": 0.29},
        ]
    }
    assert len(filter_predictions_by_score(predictions, grid[2])[1]) == 1
    assert len(filter_predictions_by_score(predictions, 0.0)[1]) == 2
    assert filter_predictions_by_score(predictions, 1.0)[1] == []


def test_invalid_prediction_score_is_rejected() -> None:
    predictions = {1: [{"bbox_xyxy": [0, 0, 1, 1], "category_id": 0, "score": 1.1}]}
    with pytest.raises(ValueError, match="score"):
        filter_predictions_by_score(predictions, 0.5)


def test_sweep_reuses_official_matching() -> None:
    """扫描结果遵循官方的一对一匹配和重复框 FP 规则。"""
    gt_boxes = {
        1: [
            {"bbox_xyxy": [0, 0, 10, 10], "category_id": 0},
            {"bbox_xyxy": [20, 20, 30, 30], "category_id": 0},
        ]
    }
    pred_boxes = {
        1: [
            {"bbox_xyxy": [0, 0, 10, 10], "category_id": 0, "score": 0.9},
            {"bbox_xyxy": [20, 20, 30, 30], "category_id": 0, "score": 0.8},
            {"bbox_xyxy": [0, 0, 10, 10], "category_id": 0, "score": 0.7},
        ]
    }
    points = sweep_global_thresholds(
        gt_boxes,
        pred_boxes,
        [0.7, 0.8],
        class_names=["ship"],
        category_mapping={0: "ship"},
        iou_thresholds={"ship": 0.5},
    )

    assert points[0].detections_kept == 3
    assert points[0].metrics.recall == 1.0
    assert points[0].metrics.fdr == pytest.approx(1 / 3)
    assert points[1].detections_kept == 2
    assert points[1].metrics.recall == 1.0
    assert points[1].metrics.fdr == 0.0


def test_operating_point_caps_pass_and_tie_break() -> None:
    """三个工作点遵循固定上限、通过条件和并列规则。"""
    selections = select_operating_points(
        [
            _point(0.90, 0.75, 0.00),
            _point(0.81, 1.00, 0.20),
            _point(0.80, 1.00, 0.20),
            _point(0.70, 1.00, 0.25),
        ],
        official_recall_min=0.85,
        official_fdr_max=0.20,
        internal_recall_min=0.88,
        internal_fdr_max=0.17,
    )

    assert selections["official_best"].point.threshold == 0.81
    assert selections["official_best"].passed is True
    assert selections["internal_best"].point.threshold == 0.90
    assert selections["internal_best"].passed is False
    assert selections["recall_ceiling"].point.threshold == 0.81
    assert selections["recall_ceiling"].passed is None


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [
        (0.0, 1.0, 0.0),
        (0.0, 1.0, -0.1),
        (0.8, 0.2, 0.1),
        (math.nan, 1.0, 0.1),
        (0.0, math.inf, 0.1),
    ],
)
def test_invalid_grid_is_rejected(start: float, stop: float, step: float) -> None:
    with pytest.raises(ValueError):
        build_threshold_grid(start, stop, step)
