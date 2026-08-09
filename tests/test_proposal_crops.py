"""N2-1 proposal crop manifest 模块测试。"""

import csv
import json
from pathlib import Path

import pytest

from rsdet.analysis.proposal_crops import (
    BACKGROUND_CLASS_ID,
    BACKGROUND_CLASS_NAME,
    build_proposal_crop_manifest,
    load_image_path_mapping,
    split_by_fold,
    stats_by_view,
    write_proposal_crop_manifest,
)


def _formal_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "formal_manifest.csv"
    path.write_text(
        "crop_policy,annotation_uid,formal_image_id,source_relative_path,"
        "source_width,source_height,gt_x0,gt_y0,gt_x1,gt_y1,class_id\n"
        "tight,ann_1,101,images/train/a.jpg,1024,1024,0,0,10,10,24\n"
        "context_1p25,ann_1,101,images/train/a.jpg,1024,1024,0,0,10,10,24\n"
        "tight,ann_2,102,images/train/b.jpg,2048,2048,5,5,15,15,4\n",
        encoding="utf-8",
    )
    return path


def _evidence_manifest() -> dict:
    return {
        "records": [
            {
                "proposal_uid": "p1",
                "image_id": 101,
                "fold": 0,
                "source_group": "g1",
                "category_id": 24,
                "score": 0.9,
                "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
                "official_status": "TP",
                "matched_gt_category": 24,
                "matched_gt_key": "101:0",
                "oracle_hit": True,
            },
            {
                "proposal_uid": "p2",
                "image_id": 102,
                "fold": 1,
                "source_group": "g2",
                "category_id": 5,
                "score": 0.7,
                "bbox_xyxy": [5.0, 5.0, 15.0, 15.0],
                "official_status": "FP_CLS",
                "matched_gt_category": None,
                "matched_gt_key": None,
                "oracle_hit": True,  # 大类对、细类错
            },
            {
                "proposal_uid": "p3",
                "image_id": 101,
                "fold": 0,
                "source_group": "g1",
                "category_id": 24,
                "score": 0.3,
                "bbox_xyxy": [100.0, 100.0, 110.0, 110.0],
                "official_status": "FP_BG",
                "matched_gt_category": None,
                "matched_gt_key": None,
                "oracle_hit": False,  # 纯背景
            },
        ]
    }


class TestLoadImagePathMapping:
    def test_maps_tight_rows(self, tmp_path: Path):
        formal = _formal_manifest(tmp_path)
        mapping = load_image_path_mapping(formal)
        assert set(mapping) == {101, 102}
        assert mapping[101]["source_relative_path"] == "images/train/a.jpg"
        assert mapping[101]["source_width"] == 1024
        assert mapping[102]["source_height"] == 2048


class TestBuildManifest:
    def test_basic_views_and_labels(self, tmp_path: Path):
        formal = _formal_manifest(tmp_path)
        rows, summary = build_proposal_crop_manifest(
            evidence_manifest=_evidence_manifest(),
            formal_manifest_path=formal,
        )
        assert summary["rows_written"] == 3
        assert summary["missing_image_paths"] == 0
        assert summary["background_candidates"] == 1

        by_uid = {row["proposal_uid"]: row for row in rows}
        # TP → deployable_positive，标签=真值 24。
        assert by_uid["p1"]["view"] == "deployable_positive"
        assert by_uid["p1"]["class_id"] == 24
        assert by_uid["p1"]["is_background"] == "false"
        # FP_CLS 但 oracle hit → oracle_positive，标签=预测类别 5。
        assert by_uid["p2"]["view"] == "oracle_positive"
        assert by_uid["p2"]["class_id"] == 5
        # FP_BG 无 oracle → hard_negative 且 background=true，标签=25。
        assert by_uid["p3"]["view"] == "hard_negative"
        assert by_uid["p3"]["class_id"] == BACKGROUND_CLASS_ID
        assert by_uid["p3"]["is_background"] == "true"
        assert by_uid["p3"]["class_name"] == BACKGROUND_CLASS_NAME
        assert by_uid["p3"]["major_class"] == "background"

    def test_view_filter(self, tmp_path: Path):
        formal = _formal_manifest(tmp_path)
        rows, summary = build_proposal_crop_manifest(
            evidence_manifest=_evidence_manifest(),
            formal_manifest_path=formal,
            include_views=["deployable_positive"],
        )
        assert summary["rows_written"] == 1
        assert rows[0]["proposal_uid"] == "p1"

    def test_missing_image_skipped(self, tmp_path: Path):
        evidence = {
            "records": [
                {
                    "proposal_uid": "px",
                    "image_id": 999,
                    "fold": 0,
                    "source_group": "gx",
                    "category_id": 24,
                    "score": 0.5,
                    "bbox_xyxy": [0, 0, 10, 10],
                    "official_status": "FP_BG",
                    "oracle_hit": False,
                }
            ]
        }
        formal = _formal_manifest(tmp_path)
        rows, summary = build_proposal_crop_manifest(
            evidence_manifest=evidence, formal_manifest_path=formal
        )
        assert rows == []
        assert summary["missing_image_paths"] == 1


class TestWriteAndSplit:
    def test_roundtrip(self, tmp_path: Path):
        formal = _formal_manifest(tmp_path)
        rows, _ = build_proposal_crop_manifest(
            evidence_manifest=_evidence_manifest(),
            formal_manifest_path=formal,
        )
        out = tmp_path / "proposal.csv"
        write_proposal_crop_manifest(rows, out)
        with out.open("r", encoding="utf-8") as handle:
            reader = list(csv.DictReader(handle))
        assert len(reader) == 3
        assert set(reader[0].keys()) >= {
            "proposal_uid", "crop_xyxy", "fold", "view", "class_id"
        }

    def test_split_and_stats(self, tmp_path: Path):
        formal = _formal_manifest(tmp_path)
        rows, _ = build_proposal_crop_manifest(
            evidence_manifest=_evidence_manifest(),
            formal_manifest_path=formal,
        )
        folded = split_by_fold(rows)
        assert set(folded) == {0, 1}
        assert len(folded[0]) == 2
        stats = stats_by_view(rows)
        assert stats["deployable_positive"]["n"] == 1
        assert stats["hard_negative"]["background"] == 1
