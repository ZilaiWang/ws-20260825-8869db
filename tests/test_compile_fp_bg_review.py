from __future__ import annotations

import csv
from pathlib import Path

import pytest

from rsdet.analysis.fp_bg_audit import LABEL_CLEAR_BACKGROUND, LABEL_DUPLICATE
from scripts.compile_fp_bg_review import compile_review


def _write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fixture(tmp_path: Path, labels: tuple[str, str]) -> tuple[Path, Path, Path]:
    decisions = tmp_path / "decisions.csv"
    mapping = tmp_path / "mapping.csv"
    audit = tmp_path / "audit.csv"
    _write(
        decisions,
        ["card_id", "label", "labeler", "notes"],
        [
            {"card_id": "card-0000", "label": labels[0], "labeler": "r", "notes": ""},
            {"card_id": "card-0001", "label": labels[1], "labeler": "r", "notes": ""},
        ],
    )
    _write(
        mapping,
        ["card_id", "audit_uid", "proposal_uid", "image_id", "relative_path"],
        [
            {"card_id": "card-0000", "audit_uid": "a0", "proposal_uid": "p0", "image_id": 1, "relative_path": "a.jpg"},
            {"card_id": "card-0001", "audit_uid": "a1", "proposal_uid": "p0", "image_id": 1, "relative_path": "a.jpg"},
        ],
    )
    fields = [
        "audit_uid", "proposal_uid", "image_id", "fold", "category_id", "class_name",
        "score", "bbox_xyxy", "is_repeat_control", "repeat_of",
    ]
    _write(
        audit,
        fields,
        [
            {"audit_uid": "a0", "proposal_uid": "p0", "image_id": 1, "fold": 0, "category_id": 4, "class_name": "aircraft", "score": 0.2, "bbox_xyxy": "[1,2,3,4]", "is_repeat_control": False, "repeat_of": ""},
            {"audit_uid": "a1", "proposal_uid": "p0", "image_id": 1, "fold": 0, "category_id": 4, "class_name": "aircraft", "score": 0.2, "bbox_xyxy": "[1,2,3,4]", "is_repeat_control": True, "repeat_of": "p0"},
        ],
    )
    return decisions, mapping, audit


def test_compile_admits_consistent_clear_background(tmp_path: Path) -> None:
    decisions, mapping, audit = _fixture(
        tmp_path, (LABEL_CLEAR_BACKGROUND, LABEL_CLEAR_BACKGROUND)
    )
    joined, whitelist, decision = compile_review(
        decisions_path=decisions,
        mapping_path=mapping,
        audit_path=audit,
        minimum_repeat_consistency=0.85,
    )
    assert len(joined) == 2
    assert len(whitelist) == 1
    assert decision["background_whitelist_admission"] is True
    assert decision["repeat_consistency_rate"] == 1.0


def test_compile_rejects_disagreeing_repeat(tmp_path: Path) -> None:
    decisions, mapping, audit = _fixture(
        tmp_path, (LABEL_CLEAR_BACKGROUND, LABEL_DUPLICATE)
    )
    _, whitelist, decision = compile_review(
        decisions_path=decisions,
        mapping_path=mapping,
        audit_path=audit,
        minimum_repeat_consistency=0.85,
    )
    assert whitelist == []
    assert decision["background_whitelist_admission"] is False
    assert decision["conflicting_proposal_count"] == 1


def test_compile_rejects_missing_decision(tmp_path: Path) -> None:
    decisions, mapping, audit = _fixture(
        tmp_path, (LABEL_CLEAR_BACKGROUND, LABEL_CLEAR_BACKGROUND)
    )
    rows = list(csv.DictReader(decisions.open(newline="", encoding="utf-8")))
    _write(decisions, list(rows[0]), rows[:1])
    with pytest.raises(ValueError, match="card_id 集合不一致"):
        compile_review(
            decisions_path=decisions,
            mapping_path=mapping,
            audit_path=audit,
            minimum_repeat_consistency=0.85,
        )
