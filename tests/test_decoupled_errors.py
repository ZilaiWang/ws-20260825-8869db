"""N0-2 定位/分类解耦模块测试。"""

import math

import pytest

from rsdet.analysis.decoupled_errors import (
    analyze_decoupled_errors,
    compute_localized_fine_accuracy,
    compute_oracle_localization,
    compute_source_group_bootstrap,
    stratify_oracle_localization,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol

# 与 configs/project.yaml 相同的 25 细类映射（4 船 + 20 飞机 + 1 车辆）。
_CATEGORY_MAPPING = {index: "ship" for index in range(0, 4)}
_CATEGORY_MAPPING.update({index: "aircraft" for index in range(4, 24)})
_CATEGORY_MAPPING[24] = "vehicle"

_PROTOCOL_CONFIG = {
    "protocol_versions": {
        "contract_version": "contract_v1",
        "eval_version": "official_eval_v1",
    },
    "task": {
        "class_names": ["ship", "aircraft", "vehicle"],
        "dataset_category_mapping": {
            str(key): value for key, value in _CATEGORY_MAPPING.items()
        },
    },
    "official_evaluation": {
        "recall_min": 0.85,
        "fdr_max": 0.20,
        "iou_thresholds": {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
    },
}

PROTOCOL = parse_evaluation_protocol(_PROTOCOL_CONFIG)


def _gt(category_id: int, bbox_xyxy: list[float], class_name: str = "") -> dict:
    return {
        "bbox_xyxy": bbox_xyxy,
        "category_id": category_id,
        "class_name": class_name or str(category_id),
    }


def _pred(category_id: int, score: float, bbox_xyxy: list[float]) -> dict:
    return {
        "bbox_xyxy": bbox_xyxy,
        "category_id": category_id,
        "score": score,
    }


class TestComputeOracleLocalization:
    def test_oracle_corrects_wrong_fine_class(self):
        """预测细类错误但大类正确时，oracle 定位仍能召回。"""
        gt = {1: [_gt(5, [0, 0, 10, 10], "TU-160")]}
        # 预测为 aircraft 大类下的另一个细类 4（同大类、细类错），框完全对齐。
        preds = {1: [_pred(4, 0.9, [0, 0, 10, 10])]}
        metrics, rows = compute_oracle_localization(
            gt, preds, protocol=PROTOCOL, threshold=0.5
        )
        assert metrics.recall == 1.0
        assert len(rows) == 1
        assert rows[0]["oracle_matched"] is True
        # 细类不同 → correct_fine False。
        assert rows[0]["correct_fine"] is False

    def test_wrong_coarse_class_not_rescued(self):
        """大类不同（ship vs aircraft）时 oracle 不召回。"""
        gt = {1: [_gt(0, [0, 0, 10, 10], "MS")]}
        preds = {1: [_pred(4, 0.9, [0, 0, 10, 10])]}
        metrics, rows = compute_oracle_localization(
            gt, preds, protocol=PROTOCOL, threshold=0.5
        )
        assert metrics.recall == 0.0
        assert rows[0]["oracle_matched"] is False

    def test_geometry_below_iou_not_matched(self):
        """IoU 低于大类阈值时不匹配。"""
        gt = {1: [_gt(24, [0, 0, 100, 100], "FSC")]}
        preds = {1: [_pred(24, 0.9, [90, 90, 100, 100])]}  # IoU 远低于 0.35
        metrics, _ = compute_oracle_localization(
            gt, preds, protocol=PROTOCOL, threshold=0.5
        )
        assert metrics.recall == 0.0

    def test_score_threshold_filters_low_score(self):
        gt = {1: [_gt(24, [0, 0, 10, 10], "FSC")]}
        preds = {1: [_pred(24, 0.1, [0, 0, 10, 10])]}
        metrics, _ = compute_oracle_localization(
            gt, preds, protocol=PROTOCOL, threshold=0.5
        )
        assert metrics.recall == 0.0


class TestComputeLocalizedFineAccuracy:
    def test_accuracy_over_matched_only(self):
        rows = [
            {"oracle_matched": True, "correct_fine": True},
            {"oracle_matched": True, "correct_fine": False},
            {"oracle_matched": False, "correct_fine": None},
        ]
        result = compute_localized_fine_accuracy(rows)
        assert result.total == 2
        assert result.correct == 1
        assert result.accuracy == 0.5


class TestStratifyOracleLocalization:
    def test_scopes_created(self):
        gt = {
            1: [_gt(24, [0, 0, 5, 5], "FSC"), _gt(24, [20, 20, 30, 30], "FSC")],
            2: [_gt(4, [0, 0, 100, 100], "TU-160")],
        }
        preds = {
            1: [_pred(24, 0.9, [0, 0, 5, 5])],
            2: [_pred(4, 0.9, [0, 0, 100, 100])],
        }
        _, rows = compute_oracle_localization(
            gt, preds, protocol=PROTOCOL, threshold=0.5
        )
        stratified = stratify_oracle_localization(rows, gt)
        assert stratified["overall"]["n_objects"] == 3
        assert stratified["overall"]["matched"] == 2
        assert math.isclose(stratified["overall"]["recall"], 2 / 3)
        # 尺寸档：5px → tiny，100px → large。
        assert stratified["size_tiny_lt16"]["n_objects"] == 2
        assert stratified["size_large_ge64"]["n_objects"] == 1
        assert stratified["class_FSC"]["n_objects"] == 2


class TestComputeSourceGroupBootstrap:
    def test_bootstrap_returns_interval(self):
        rows = [
            {"image_id": 1, "oracle_matched": True},
            {"image_id": 2, "oracle_matched": True},
            {"image_id": 3, "oracle_matched": False},
        ]
        group_of_image = {1: "g1", 2: "g2", 3: "g3"}
        result = compute_source_group_bootstrap(
            rows,
            group_of_image=group_of_image,
            iterations=100,
            seed=42,
        )
        assert result["n_groups"] == 3
        assert 0.0 <= result["point_recall"] <= 1.0
        assert result["ci_low"] <= result["point_recall"] <= result["ci_high"]

    def test_bootstrap_requires_two_groups(self):
        rows = [{"image_id": 1, "oracle_matched": True}]
        with pytest.raises(ValueError, match="至少 2"):
            compute_source_group_bootstrap(
                rows, group_of_image={1: "g1"}, iterations=10
            )


class TestAnalyzeDecoupledErrors:
    def test_full_analysis(self):
        gt = {
            1: [_gt(24, [0, 0, 10, 10], "FSC")],
            2: [_gt(5, [20, 20, 40, 40], "TU-160")],
        }
        preds = {
            1: [_pred(24, 0.9, [0, 0, 10, 10])],  # 全对
            2: [_pred(4, 0.9, [20, 20, 40, 40])],  # 类错(同大类)但定位对
        }
        result = analyze_decoupled_errors(
            gt_boxes=gt,
            predictions=preds,
            protocol=PROTOCOL,
            threshold=0.5,
            group_of_image={1: "g1", 2: "g2"},
            bootstrap_iterations=50,
        )
        assert result["oracle_localization_recall"] == 1.0
        # 2 个都几何匹配；其中 1 个细类正确。
        assert result["localized_fine_accuracy"] == 0.5
        assert result["localized_fine_total"] == 2
        assert result["source_group_bootstrap"]["n_groups"] == 2
        assert "size_small_16_32" in result["stratified"]
