"""N0-1 cross-fit 阈值基线模块测试。"""

import json
import math
from pathlib import Path

import pytest

from rsdet.analysis.crossfit_thresholds import (
    _filter_by_score,
    _merge_folds,
    evaluate_ranking_workpoint,
    evaluate_workpoint,
    load_gt_from_formal_crop_manifest,
    run_crossfit,
    scan_global_threshold,
    split_by_fold,
    split_gt_by_fold,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol

_PROTOCOL_CONFIG = {
    "protocol_versions": {
        "contract_version": "contract_v1",
        "eval_version": "official_eval_v1",
    },
    "task": {
        "class_names": ["ship", "aircraft", "vehicle"],
        "dataset_category_mapping": {"0": "ship", "4": "aircraft", "24": "vehicle"},
    },
    "official_evaluation": {
        "recall_min": 0.85,
        "fdr_max": 0.20,
        "latency_max_seconds": 20.0,
        "iou_thresholds": {"ship": 0.50, "aircraft": 0.50, "vehicle": 0.35},
    },
}

PROTOCOL = parse_evaluation_protocol(_PROTOCOL_CONFIG)


def _gt(image_id: int, category_id: int, bbox_xyxy: list[float]) -> dict:
    return {"bbox_xyxy": bbox_xyxy, "category_id": category_id}


def _pred(image_id: int, category_id: int, score: float, bbox_xyxy: list[float]) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "score": score,
        "bbox_xyxy": bbox_xyxy,
    }


class TestFilterAndMerge:
    def test_filter_by_score(self):
        preds = {
            1: [
                _pred(1, 24, 0.9, [0, 0, 10, 10]),
                _pred(1, 24, 0.1, [0, 0, 10, 10]),
            ]
        }
        kept = _filter_by_score(preds, 0.5)
        assert len(kept[1]) == 1
        assert kept[1][0]["score"] == 0.9

    def test_filter_empty_image_dropped(self):
        preds = {1: [_pred(1, 24, 0.1, [0, 0, 10, 10])]}
        kept = _filter_by_score(preds, 0.5)
        assert kept == {}

    def test_merge_folds(self):
        folded = {
            0: {1: ["a"]},
            1: {2: ["b"]},
        }
        merged = _merge_folds(folded, [0, 1])
        assert merged == {1: ["a"], 2: ["b"]}


class TestSplit:
    def test_split_by_fold(self):
        preds = {
            1: ["p1"],
            2: ["p2"],
            3: ["p3"],
        }
        image_folds = {1: 0, 2: 1, 3: 0}
        folded = split_by_fold(preds, image_folds)
        assert set(folded[0]) == {1, 3}
        assert set(folded[1]) == {2}

    def test_split_gt_by_fold(self):
        gt = {1: ["g1"], 2: ["g2"]}
        image_folds = {1: 0, 2: 1}
        folded = split_gt_by_fold(gt, image_folds)
        assert set(folded[0]) == {1}
        assert set(folded[1]) == {2}


class TestLoadGtFromManifest:
    def test_loads_tight_rows_only(self, tmp_path: Path):
        path = tmp_path / "manifest.csv"
        path.write_text(
            "crop_policy,annotation_uid,formal_image_id,gt_x0,gt_y0,gt_x1,gt_y1,class_id\n"
            "tight,ann_1,100,0,0,10,10,24\n"
            "context_1p25,ann_1,100,0,0,10,10,24\n"
            "tight,ann_2,101,5,5,20,20,4\n"
            "tight,ann_3,102,1,1,9,9,0\n",
            encoding="utf-8",
        )
        boxes = load_gt_from_formal_crop_manifest(path, expected_images=3, expected_annotations=3)
        assert set(boxes) == {100, 101, 102}
        assert len(boxes[100]) == 1  # context 行被忽略
        assert boxes[100][0]["bbox_xyxy"] == [0, 0, 10, 10]
        assert boxes[100][0]["category_id"] == 24

    def test_duplicate_tight_annotation_rejected(self, tmp_path: Path):
        path = tmp_path / "manifest.csv"
        path.write_text(
            "crop_policy,annotation_uid,formal_image_id,gt_x0,gt_y0,gt_x1,gt_y1,class_id\n"
            "tight,ann_1,100,0,0,10,10,24\n"
            "tight,ann_1,100,0,0,10,10,24\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="重复"):
            load_gt_from_formal_crop_manifest(path, expected_images=1, expected_annotations=1)


class TestScanGlobalThreshold:
    def test_selects_recall_max_within_gate(self):
        """两个目标在不同阈值下都可达时，选 Recall 最大的。"""
        # 一个 GT，两个预测：高分正确、低分错误类。
        gt = {1: [_gt(1, 24, [0, 0, 10, 10])]}
        preds = {
            1: [
                _pred(1, 24, 0.9, [0, 0, 10, 10]),
                _pred(1, 4, 0.8, [0, 0, 10, 10]),
            ]
        }
        threshold, metrics, curve = scan_global_threshold(
            gt,
            preds,
            protocol=PROTOCOL,
            threshold_start=0.01,
            threshold_stop=0.95,
            threshold_step=0.05,
        )
        # 算法选"第一个 gate 达标的阈值"（保守低阈、Recall 优先）：
        # 0.81 起过滤掉 0.8 的错误类预测，Recall 1.0 / FDR 0.0。
        assert threshold == pytest.approx(0.81, abs=0.06)
        assert metrics.recall == 1.0
        assert metrics.fdr == 0.0
        assert curve

    def test_no_gate_achievable_falls_back_to_max_recall(self):
        """没有任何阈值同时满足门槛时，回退到 Recall 最大者。"""
        # 预测永远错误类，Recall=0。
        gt = {1: [_gt(1, 24, [0, 0, 10, 10])]}
        preds = {1: [_pred(1, 4, 0.9, [0, 0, 10, 10])]}
        threshold, metrics, _ = scan_global_threshold(
            gt,
            preds,
            protocol=PROTOCOL,
            threshold_start=0.01,
            threshold_stop=0.95,
            threshold_step=0.05,
        )
        assert metrics.recall == 0.0
        assert threshold == 0.01  # 保持所有预测，Recall 已为 0


class TestEvaluateWorkpoint:
    def test_basic_workpoint(self):
        gt = {1: [_gt(1, 24, [0, 0, 10, 10])]}
        preds = {1: [_pred(1, 24, 0.9, [0, 0, 10, 10])]}
        metrics = evaluate_workpoint(gt, preds, threshold=0.5, protocol=PROTOCOL)
        assert metrics.recall == 1.0
        assert metrics.fdr == 0.0

    def test_partial_taxonomy_ranking_is_explicit(self):
        gt = {1: [_gt(1, 24, [0, 0, 10, 10])]}
        preds = {1: [_pred(1, 24, 0.9, [0, 0, 10, 10])]}
        metrics = evaluate_ranking_workpoint(
            gt,
            preds,
            threshold=0.5,
            protocol=PROTOCOL,
            require_complete_taxonomy=False,
        )
        assert metrics.overall_recall == 1.0
        assert metrics.details["fine_average_policy"] == "present_in_gt_only_diagnostic"


class TestRunCrossfit:
    def _write_manifest(self, tmp_path: Path) -> Path:
        path = tmp_path / "manifest.csv"
        # 3 折、每折 2 张图、每张 1 个 GT。
        rows = [
            ("tight", f"ann_{i}", image_id, x0, y0, x0 + 10, y0 + 10, class_id)
            for i, (image_id, x0, y0, class_id) in enumerate(
                [
                    (1, 0, 0, 24),
                    (2, 5, 5, 24),
                    (3, 10, 10, 4),
                    (4, 15, 15, 4),
                    (5, 20, 20, 0),
                    (6, 25, 25, 0),
                ]
            )
        ]
        lines = ["crop_policy,annotation_uid,formal_image_id,gt_x0,gt_y0,gt_x1,gt_y1,class_id"]
        for row in rows:
            lines.append(",".join(str(value) for value in row))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _write_aggregate(self, tmp_path: Path) -> Path:
        root = tmp_path / "aggregate"
        root.mkdir(parents=True, exist_ok=True)
        (root / "oof_metadata.json").write_text(
            json.dumps(
                {
                    "model_key": "M1",
                    "seed": 42,
                    "low_score_threshold": 0.001,
                    "source_manifest_sha256": "sha256",
                }
            ),
            encoding="utf-8",
        )
        # 每张图一个 GT 匹配的预测（score 高），外加一个低分干扰预测。
        # GT 位置/类别与 manifest 一致：
        # 图1[0,0,10,10]/24, 图2[5,5,15,15]/24, 图3[10,10,20,20]/4,
        # 图4[15,15,25,25]/4, 图5[20,20,30,30]/0, 图6[25,25,35,35]/0。
        # 文件是 COCO 格式 bbox: [x, y, w, h]，load_cv3_aggregate 会转 xyxy。
        gts = {
            1: (24, [0, 0, 10, 10]),
            2: (24, [5, 5, 15, 15]),
            3: (4, [10, 10, 20, 20]),
            4: (4, [15, 15, 25, 25]),
            5: (0, [20, 20, 30, 30]),
            6: (0, [25, 25, 35, 35]),
        }
        predictions = []
        for image_id in range(1, 7):
            category_id, xyxy = gts[image_id]
            w = xyxy[2] - xyxy[0]
            h = xyxy[3] - xyxy[1]
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "score": 0.9,
                    "bbox": [xyxy[0], xyxy[1], w, h],
                }
            )
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "score": 0.05,
                    "bbox": [xyxy[0], xyxy[1], w, h],
                }
            )
        (root / "predictions_oof_low.json").write_text(json.dumps(predictions), encoding="utf-8")
        (root / "oof_images.csv").write_text(
            "image_id,relative_path,fold,group_id\n"
            "1,images/train/a.jpg,0,g1\n"
            "2,images/train/b.jpg,0,g2\n"
            "3,images/train/c.jpg,1,g3\n"
            "4,images/train/d.jpg,1,g4\n"
            "5,images/train/e.jpg,2,g5\n"
            "6,images/train/f.jpg,2,g6\n",
            encoding="utf-8",
        )
        return root

    def test_crossfit_runs_and_merges(self, tmp_path: Path):
        aggregate = self._write_aggregate(tmp_path)
        manifest = self._write_manifest(tmp_path)
        result = run_crossfit(
            aggregate_dir=aggregate,
            formal_crop_manifest_path=manifest,
            protocol=PROTOCOL,
            expected_images=6,
            expected_annotations=6,
            candidate_floor=0.001,
            threshold_start=0.001,
            threshold_stop=0.5,
            threshold_step=0.01,
            require_complete_taxonomy=False,
        )
        assert len(result["per_fold"]) == 3
        merged = result["merged_held_out"]
        # 所有 GT 都有高分正确预测，Recall 应为 1.0。
        assert merged["recall"] == pytest.approx(1.0)
        # 每折选阈应 > 0.05（排除 0.05 干扰预测，保留 0.9 高分预测）。
        for per_fold in result["per_fold"]:
            assert per_fold["selected_threshold"] > 0.05
            assert per_fold["selected_threshold"] < 0.9
        assert math.isfinite(result["threshold_dispersion"]["mean"])
        assert result["merged_held_out"]["official_gate_passed"] is True
        ranking = result["merged_held_out"]["official_ranking"]
        assert ranking["overall_macro_recall"] == pytest.approx(1.0)
        assert ranking["details"]["fine_average_policy"] == "present_in_gt_only_diagnostic"

    def test_crossfit_rejects_wrong_fold_count(self, tmp_path: Path):
        aggregate = tmp_path / "aggregate"
        aggregate.mkdir(parents=True)
        (aggregate / "oof_metadata.json").write_text(
            json.dumps({"model_key": "M1", "low_score_threshold": 0.001}),
            encoding="utf-8",
        )
        # 6 张图（与 manifest 一致）但只有 2 个 fold。
        predictions = [
            {
                "image_id": image_id,
                "category_id": 24,
                "score": 0.9,
                "bbox": [0, 0, 10, 10],
            }
            for image_id in range(1, 7)
        ]
        (aggregate / "predictions_oof_low.json").write_text(
            json.dumps(predictions), encoding="utf-8"
        )
        (aggregate / "oof_images.csv").write_text(
            "image_id,relative_path,fold,group_id\n"
            "1,images/train/a.jpg,0,g1\n"
            "2,images/train/b.jpg,0,g2\n"
            "3,images/train/c.jpg,0,g3\n"
            "4,images/train/d.jpg,1,g4\n"
            "5,images/train/e.jpg,1,g5\n"
            "6,images/train/f.jpg,1,g6\n",
            encoding="utf-8",
        )
        manifest = self._write_manifest(tmp_path)
        with pytest.raises(ValueError, match="3 折"):
            run_crossfit(
                aggregate_dir=aggregate,
                formal_crop_manifest_path=manifest,
                protocol=PROTOCOL,
                expected_images=6,
                expected_annotations=6,
                candidate_floor=0.001,
            )
