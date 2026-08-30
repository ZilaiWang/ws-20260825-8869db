"""Parity and adversarial tests for the official fixed-risk frontier."""

from __future__ import annotations

from rsdet.evaluation.official_frontier import min_fdr_at_recall, official_fixed_risk_frontier
from rsdet.evaluation.official_metric import evaluate_predictions_with_trace

MAPPING = {5: "aircraft"}
THRESHOLDS = {"aircraft": 0.50}


def _prediction(candidate_id, score, box):
    return {
        "image_id": 1,
        "category_id": 5,
        "score": score,
        "bbox_xyxy": box,
        "source_prediction_index": candidate_id,
    }


def test_prediction_first_matching_counterexample() -> None:
    """GT-first gets 2 TP here; the official prediction-first matcher gets 1."""

    gt = {
        1: [
            {"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]},
            {"category_id": 5, "bbox_xyxy": [60.0, 0.0, 160.0, 100.0]},
        ]
    }
    predictions = [
        # Matches both GTs but has higher IoU with GT2, so it must claim GT2.
        _prediction(10, 0.90, [32.0, 0.0, 132.0, 100.0]),
        # Matches only GT2; after P10 claims GT2 this becomes FP.
        _prediction(11, 0.80, [80.0, 0.0, 160.0, 100.0]),
    ]
    result = official_fixed_risk_frontier(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping=MAPPING,
        iou_thresholds=THRESHOLDS,
        fdr_levels=(0.60,),
        nms_iou=0.50,
    )
    assert result.total_tp == 1
    assert result.total_fp == 1
    assert result.points[0.60].tp == 1


def test_frontier_matches_official_trace_exactly() -> None:
    gt = {1: [{"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]}]}
    predictions = [
        _prediction(20, 0.90, [0.0, 0.0, 100.0, 100.0]),
        _prediction(21, 0.10, [200.0, 200.0, 300.0, 300.0]),
    ]
    frontier = official_fixed_risk_frontier(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping=MAPPING,
        iou_thresholds=THRESHOLDS,
        fdr_levels=(0.12,),
        nms_iou=0.50,
    )
    metrics, trace = evaluate_predictions_with_trace(
        gt,
        frontier.kept_predictions,
        class_names=["aircraft"],
        category_mapping=MAPPING,
        iou_thresholds=THRESHOLDS,
    )
    assert metrics.details["tp"] == frontier.total_tp == 1
    assert metrics.details["fp"] == frontier.total_fp == 1
    assert {match.prediction_index for match in trace.matches} == {20}
    # Active workpoint excludes the low-score FP.
    assert frontier.points[0.12].tp == 1
    assert frontier.points[0.12].fp == 0


def test_equal_score_block_cannot_be_partially_selected() -> None:
    gt = {1: [{"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]}]}
    predictions = [
        _prediction(30, 0.50, [0.0, 0.0, 100.0, 100.0]),
        _prediction(31, 0.50, [200.0, 200.0, 300.0, 300.0]),
    ]
    result = official_fixed_risk_frontier(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping=MAPPING,
        iou_thresholds=THRESHOLDS,
        fdr_levels=(0.12,),
        nms_iou=0.50,
    )
    point = result.points[0.12]
    assert point.recall == 0.0
    assert point.score_threshold is None
    assert point.tp == point.fp == 0


def test_candidate_id_controls_equal_score_determinism() -> None:
    gt = {1: [{"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]}]}
    predictions = [
        _prediction(40, 0.75, [0.0, 0.0, 100.0, 100.0]),
        _prediction(41, 0.75, [0.0, 0.0, 100.0, 100.0]),
    ]
    kwargs = dict(
        gt_boxes=gt,
        category_mapping=MAPPING,
        iou_thresholds=THRESHOLDS,
        fdr_levels=(0.12,),
        nms_iou=0.50,
    )
    forward = official_fixed_risk_frontier(predictions=predictions, **kwargs)
    reverse = official_fixed_risk_frontier(predictions=reversed(predictions), **kwargs)
    assert forward.kept_predictions == reverse.kept_predictions
    assert forward.points == reverse.points


def test_workpoint_ids_separate_active_counts_from_full_tail() -> None:
    gt = {1: [{"category_id": 5, "bbox_xyxy": [0.0, 0.0, 100.0, 100.0]}]}
    predictions = [
        _prediction(50, 0.90, [0.0, 0.0, 100.0, 100.0]),
        _prediction(51, 0.10, [200.0, 200.0, 300.0, 300.0]),
        _prediction(52, 0.01, [400.0, 400.0, 500.0, 500.0]),
    ]
    result = official_fixed_risk_frontier(
        gt_boxes=gt,
        predictions=predictions,
        category_mapping=MAPPING,
        iou_thresholds=THRESHOLDS,
        fdr_levels=(0.12,),
        nms_iou=0.50,
    )
    assert result.selected_candidate_ids[0.12] == (50,)
    assert result.selected_tp_candidate_ids[0.12] == (50,)
    assert result.selected_fp_candidate_ids[0.12] == ()
    assert result.total_fp == 2
    assert min_fdr_at_recall(result, recall_levels=(1.0,)) == {1.0: 0.0}
