"""N0-3 对象证据 manifest 模块测试。"""

import json
import math

import pytest

from rsdet.analysis.object_evidence import (
    FP_BG,
    FP_CLS,
    FP_DUP,
    FP_LOC,
    _classify_fp,
    _size_bin,
    build_object_evidence_manifest,
    manifest_sha256,
    write_manifest_json,
)
from rsdet.evaluation.protocol import parse_evaluation_protocol

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


def _gt(category_id: int, bbox_xyxy: list[float]) -> dict:
    return {"bbox_xyxy": bbox_xyxy, "category_id": category_id}


def _pred(category_id: int, score: float, bbox_xyxy: list[float]) -> dict:
    return {"bbox_xyxy": bbox_xyxy, "category_id": category_id, "score": score}


class TestSizeBin:
    def test_bins(self):
        assert _size_bin([0, 0, 5, 5]) == "tiny_lt16"
        assert _size_bin([0, 0, 20, 20]) == "small_16_32"
        assert _size_bin([0, 0, 40, 40]) == "medium_32_64"
        assert _size_bin([0, 0, 80, 80]) == "large_ge64"


class TestClassifyFp:
    def test_clear_background(self):
        gt_lookup = {(1, 0): _gt(24, [100, 100, 110, 110])}
        record = _pred(24, 0.5, [0, 0, 10, 10])
        assert (
            _classify_fp(
            record, gt_lookup, None, PROTOCOL, pred_index=0, image_id=1
        )
            == FP_BG
        )

    def test_duplicate_same_fine_and_iou(self):
        gt_lookup = {(1, 0): _gt(24, [0, 0, 100, 100])}
        record = _pred(24, 0.5, [0, 0, 100, 100])  # 同细类同框 → 抢占
        assert (
            _classify_fp(
            record, gt_lookup, None, PROTOCOL, pred_index=0, image_id=1
        )
            == FP_DUP
        )

    def test_wrong_fine_but_overlaps(self):
        # 预测 aircraft 细类 5，GT 是 aircraft 细类 4，框重叠且 IoU 够。
        gt_lookup = {(1, 0): _gt(4, [0, 0, 100, 100])}
        record = _pred(5, 0.5, [0, 0, 100, 100])
        assert (
            _classify_fp(
            record, gt_lookup, None, PROTOCOL, pred_index=0, image_id=1
        )
            == FP_CLS
        )

    def test_poor_localization(self):
        # 同细类但 IoU 低于 0.35（车辆）。
        gt_lookup = {(1, 0): _gt(24, [0, 0, 100, 100])}
        record = _pred(24, 0.5, [98, 98, 100, 100])  # IoU 很小但 > 0
        assert (
            _classify_fp(
            record, gt_lookup, None, PROTOCOL, pred_index=0, image_id=1
        )
            == FP_LOC
        )

    def test_wrong_coarse_class(self):
        # 预测 aircraft，GT 是 ship，框重叠够 IoU → 仍是 FP_CLS（大类错也算）。
        gt_lookup = {(1, 0): _gt(0, [0, 0, 100, 100])}
        record = _pred(5, 0.5, [0, 0, 100, 100])
        assert (
            _classify_fp(
            record, gt_lookup, None, PROTOCOL, pred_index=0, image_id=1
        )
            == FP_CLS
        )


class TestBuildManifest:
    def _setup(self):
        gt = {
            1: [_gt(24, [0, 0, 10, 10]), _gt(4, [100, 100, 110, 110])],
            2: [_gt(5, [200, 200, 210, 210])],
        }
        preds = {
            1: [
                _pred(24, 0.9, [0, 0, 10, 10]),  # TP
                _pred(5, 0.8, [100, 100, 110, 110]),  # 细类错 → FP_CLS
                _pred(4, 0.1, [0, 0, 10, 10]),  # 低分，过滤
            ],
            2: [
                _pred(4, 0.9, [200, 200, 210, 210]),  # 细类错(同大类)→oracle hit
            ],
        }
        image_folds = {1: 0, 2: 1}
        checkpoint = {1: "ckpt0", 2: "ckpt1"}
        image_groups = {1: "g1", 2: "g2"}
        return gt, preds, image_folds, checkpoint, image_groups

    def test_basic_manifest(self):
        gt, preds, image_folds, ckpt, groups = self._setup()
        manifest = build_object_evidence_manifest(
            gt_boxes=gt,
            predictions=preds,
            protocol=PROTOCOL,
            threshold=0.5,
            image_folds=image_folds,
            checkpoint_sha256=ckpt,
            image_groups=groups,
        )
        # 过滤后 3 个候选。
        assert len(manifest["records"]) == 3
        assert manifest["meta"]["official_tp"] == 1
        assert manifest["summary"]["official_fp"] == 2
        assert manifest["summary"]["fp_by_type"]["FP_CLS"] == 2

        # 逐记录检查。
        by_uid = {record["proposal_uid"]: record for record in manifest["records"]}
        tp_record = [r for r in manifest["records"] if r["official_status"] == "TP"]
        assert len(tp_record) == 1
        assert tp_record[0]["checkpoint_sha256"] == "ckpt0"
        assert tp_record[0]["source_group"] == "g1"
        assert tp_record[0]["matched_iou"] is not None

        fp_records = [
            r for r in manifest["records"] if r["official_status"] != "TP"
        ]
        assert len(fp_records) == 2
        # 细类错（5 vs GT 4）→ FP_CLS。
        fp_1 = [r for r in fp_records if r["image_id"] == 1][0]
        assert fp_1["official_status"] == FP_CLS
        # oracle：飞机类 GT 被同大类不同细类命中。
        fp_2 = [r for r in fp_records if r["image_id"] == 2][0]
        assert fp_2["oracle_hit"] is True
        assert fp_2["official_status"] == FP_CLS

    def test_views(self):
        gt, preds, image_folds, ckpt, groups = self._setup()
        manifest = build_object_evidence_manifest(
            gt_boxes=gt,
            predictions=preds,
            protocol=PROTOCOL,
            threshold=0.5,
            image_folds=image_folds,
            checkpoint_sha256=ckpt,
            image_groups=groups,
        )
        views = manifest["views"]
        assert len(views["deployable_positive"]) == 1
        assert len(views["oracle_positive"]) == 3  # 全部几何命中
        # hard_negative = FP_BG 或未命中 oracle 的 FP。
        assert len(views["hard_negative"]) == 0  # 本例全是 FP_CLS 且有 oracle hit

    def test_fp_bg_goes_to_hard_negative(self):
        gt = {1: [_gt(24, [0, 0, 10, 10])]}
        preds = {1: [_pred(24, 0.9, [500, 500, 510, 510])]}  # 纯背景
        manifest = build_object_evidence_manifest(
            gt_boxes=gt,
            predictions=preds,
            protocol=PROTOCOL,
            threshold=0.5,
            image_folds={1: 0},
            checkpoint_sha256={1: "ckpt"},
        )
        record = manifest["records"][0]
        assert record["official_status"] == FP_BG
        assert record["oracle_hit"] is False
        assert len(manifest["views"]["hard_negative"]) == 1

    def test_low_score_filtered(self):
        gt = {1: [_gt(24, [0, 0, 10, 10])]}
        preds = {
            1: [
                _pred(24, 0.9, [0, 0, 10, 10]),
                _pred(24, 0.1, [0, 0, 10, 10]),
            ]
        }
        manifest = build_object_evidence_manifest(
            gt_boxes=gt,
            predictions=preds,
            protocol=PROTOCOL,
            threshold=0.5,
            image_folds={1: 0},
            checkpoint_sha256={1: "ckpt"},
        )
        assert len(manifest["records"]) == 1


class TestWriteAndHash:
    def test_write_and_sha(self, tmp_path):
        path = tmp_path / "manifest.json"
        write_manifest_json({"records": [], "views": {}, "meta": {}}, path)
        assert path.is_file()
        sha = manifest_sha256(path)
        assert len(sha) == 64
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data == {"records": [], "views": {}, "meta": {}}
