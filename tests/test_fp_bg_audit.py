"""N0-4 FP_BG 人工语义审计模块测试。"""

import json
from pathlib import Path

import pytest

from rsdet.analysis.fp_bg_audit import (
    ADMISSIBLE_BACKGROUND_LABELS,
    LABEL_CLEAR_BACKGROUND,
    LABEL_DUPLICATE,
    LABEL_INVALID_CROP,
    LABEL_PLAUSIBLE_UNLABELED,
    LABEL_POOR_LOCALIZATION,
    audit_samples_to_csv,
    audit_samples_to_json,
    build_fp_bg_sample_pool,
    compute_audit_summary,
    load_audit_labels,
    sample_fp_bg_audit,
    score_bin_of,
)

_MAPPING = {0: "MS", 4: "TU-160", 24: "FSC"}


def _manifest_with_fp_bg(n: int = 30) -> dict:
    """构造含 FP_BG 候选的 manifest。"""
    records = []
    for index in range(n):
        fold = index % 3
        category_id = [0, 4, 24][index % 3]
        records.append(
            {
                "proposal_uid": f"p{index}",
                "image_id": index + 1,
                "fold": fold,
                "category_id": category_id,
                "class_name": "",
                "score": [0.05, 0.2, 0.5][index % 3],
                "bbox_xyxy": [0, 0, 10, 10],
                "size_bin": "small_16_32",
                "source_group": f"g{fold}",
                "official_status": "FP_BG",
            }
        )
    return {"records": records}


class TestScoreBin:
    def test_bins(self):
        assert score_bin_of(0.05) == "low"
        assert score_bin_of(0.20) == "mid"
        assert score_bin_of(0.50) == "high"
        assert score_bin_of(0.35) == "high"


class TestBuildPool:
    def test_selects_fp_bg_only(self):
        manifest = {
            "records": [
                {"official_status": "FP_BG", "proposal_uid": "a"},
                {"official_status": "TP", "proposal_uid": "b"},
                {"official_status": "FP_CLS", "proposal_uid": "c"},
            ]
        }
        pool = build_fp_bg_sample_pool(manifest)
        assert len(pool) == 1
        assert pool[0]["proposal_uid"] == "a"


class TestSampleAudit:
    def test_samples_have_repeat_controls(self):
        manifest = _manifest_with_fp_bg(60)
        samples, summary = sample_fp_bg_audit(
            manifest,
            category_mapping=_MAPPING,
            max_per_stratum=10,
            repeat_control_fraction=0.20,
            seed=42,
        )
        assert summary["pool_size"] == 60
        positives = [s for s in samples if not s.is_repeat_control]
        repeats = [s for s in samples if s.is_repeat_control]
        assert positives
        assert repeats
        assert all(s.repeat_of for s in repeats)
        # 每个正卡都有唯一 audit_uid。
        assert len({s.audit_uid for s in samples}) == len(samples)

    def test_deterministic_seed(self):
        manifest = _manifest_with_fp_bg(60)
        _, summary_a = sample_fp_bg_audit(manifest, category_mapping=_MAPPING, seed=7)
        _, summary_b = sample_fp_bg_audit(manifest, category_mapping=_MAPPING, seed=7)
        assert summary_a == summary_b

    def test_class_name_filled_from_mapping(self):
        manifest = _manifest_with_fp_bg(9)
        samples, _ = sample_fp_bg_audit(manifest, category_mapping=_MAPPING, max_per_stratum=5)
        class_names = {s.class_name for s in samples}
        assert "FSC" in class_names

    def test_empty_pool_rejected(self):
        with pytest.raises(ValueError, match="抽检池为空"):
            sample_fp_bg_audit({"records": []}, category_mapping=_MAPPING)

    def test_max_per_stratum_limits(self):
        # 全部属同一大类 FSC；fold = i % 3、score = [0.05, 0.2, 0.5][(i//3) % 3]
        # 使 fold 与分数分位独立变化 → 3 折 × 3 分位 = 9 层，每层若干条。
        manifest = {
            "records": [
                {
                    "proposal_uid": f"p{i}",
                    "image_id": i,
                    "fold": i % 3,
                    "category_id": 24,
                    "class_name": "",
                    "score": [0.05, 0.2, 0.5][(i // 3) % 3],
                    "bbox_xyxy": [0, 0, 10, 10],
                    "size_bin": "small",
                    "source_group": "g",
                    "official_status": "FP_BG",
                }
                for i in range(60)
            ]
        }
        samples, summary = sample_fp_bg_audit(
            manifest, category_mapping=_MAPPING, max_per_stratum=2
        )
        assert summary["strata_sampled"] == 9
        assert summary["sampled_positives"] <= 18  # 9 层 × 2


class TestCsvRoundTrip:
    def test_round_trip_with_labels(self, tmp_path: Path):
        manifest = _manifest_with_fp_bg(30)
        samples, _ = sample_fp_bg_audit(manifest, category_mapping=_MAPPING, max_per_stratum=5)
        csv_path = tmp_path / "audit.csv"
        audit_samples_to_csv(samples, csv_path)
        # 填上标签后读回。
        lines = csv_path.read_text(encoding="utf-8").splitlines()
        header = lines[0]
        assert "label" in header
        assert "audit_uid" in header

        # 手工填标签。
        rows = load_audit_labels(csv_path)
        assert len(rows) == len(samples)
        assert all(row["label"] == "" for row in rows)

    def test_invalid_label_rejected(self, tmp_path: Path):
        import csv

        manifest = _manifest_with_fp_bg(10)
        samples, _ = sample_fp_bg_audit(manifest, category_mapping=_MAPPING, max_per_stratum=3)
        csv_path = tmp_path / "audit.csv"
        audit_samples_to_csv(samples, csv_path)

        # 用 csv 模块把第一行的 label 改成非法值。
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        rows[0]["label"] = "BOGUS"
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        with pytest.raises(ValueError, match="未知人工标签"):
            load_audit_labels(csv_path)


class TestJsonOutput:
    def test_json_contains_summary_and_samples(self, tmp_path: Path):
        manifest = _manifest_with_fp_bg(30)
        samples, summary = sample_fp_bg_audit(
            manifest, category_mapping=_MAPPING, max_per_stratum=5
        )
        path = tmp_path / "audit.json"
        audit_samples_to_json(samples, path, summary=summary)
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["summary"]["total_samples"] == len(samples)
        assert len(data["samples"]) == len(samples)


class TestAuditSummary:
    def test_counts_and_consistency(self):
        rows = [
            {"proposal_uid": "p1", "label": LABEL_CLEAR_BACKGROUND, "repeat_of": ""},
            {"proposal_uid": "p1", "label": LABEL_CLEAR_BACKGROUND, "repeat_of": "p1"},
            {"proposal_uid": "p2", "label": LABEL_PLAUSIBLE_UNLABELED, "repeat_of": ""},
            {"proposal_uid": "p2", "label": LABEL_DUPLICATE, "repeat_of": "p2"},
            {"proposal_uid": "p3", "label": LABEL_INVALID_CROP, "repeat_of": ""},
            {"proposal_uid": "p4", "label": LABEL_POOR_LOCALIZATION, "repeat_of": ""},
            {"proposal_uid": "p5", "label": "", "repeat_of": ""},
        ]
        result = compute_audit_summary(rows)
        assert result["labeled"] == 6
        assert result["unlabeled"] == 1
        assert result["label_counts"][LABEL_CLEAR_BACKGROUND] == 2
        assert result["repeat_pairs_total"] == 2
        assert result["repeat_pairs_consistent"] == 1
        assert result["repeat_pairs_incomplete"] == 0
        assert result["repeat_consistency_rate"] == 0.5

    def test_repeat_without_primary_label_is_incomplete(self):
        rows = [
            {"proposal_uid": "p1", "label": "", "repeat_of": ""},
            {"proposal_uid": "p1", "label": LABEL_CLEAR_BACKGROUND, "repeat_of": "p1"},
        ]
        result = compute_audit_summary(rows)
        assert result["repeat_pairs_total"] == 1
        assert result["repeat_pairs_incomplete"] == 1
        assert result["repeat_consistency_rate"] == 0.0

    def test_admissible_labels(self):
        assert LABEL_CLEAR_BACKGROUND in ADMISSIBLE_BACKGROUND_LABELS
        assert LABEL_PLAUSIBLE_UNLABELED not in ADMISSIBLE_BACKGROUND_LABELS
