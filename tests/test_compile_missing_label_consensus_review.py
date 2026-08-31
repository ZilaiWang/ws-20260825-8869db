from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.compile_missing_label_consensus_review import main

FIELDS = [
    "candidate_id",
    "fold",
    "file_name",
    "category_id",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "primary_score",
    "support_score",
    "support_iou",
    "agreement_product",
    "maximum_gt_iou",
    "human_decision",
    "review_note",
]


def _write(path: Path, decision: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "candidate_id": "F0-1-2",
                "fold": 0,
                "file_name": "images/train/a.jpg",
                "category_id": 24,
                "bbox_x1": 1,
                "bbox_y1": 2,
                "bbox_x2": 11,
                "bbox_y2": 12,
                "primary_score": 0.5,
                "support_score": 0.6,
                "support_iou": 0.7,
                "agreement_product": 0.3,
                "maximum_gt_iou": 0.0,
                "human_decision": decision,
                "review_note": "test",
            }
        )


def test_incomplete_review_fails_closed(tmp_path: Path, monkeypatch) -> None:
    review = tmp_path / "review.csv"
    _write(review, "")
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv", ["compile", "--review-csv", str(review), "--output-dir", str(output)]
    )
    assert main() == 2
    decision = json.loads((output / "decision.json").read_text())
    assert decision["annotation_patch_admission"] is False


def test_completed_review_writes_versioned_patch(tmp_path: Path, monkeypatch) -> None:
    review = tmp_path / "review.csv"
    _write(review, "confirmed_missing")
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv", ["compile", "--review-csv", str(review), "--output-dir", str(output)]
    )
    assert main() == 0
    decision = json.loads((output / "decision.json").read_text())
    assert decision["annotation_patch_admission"] is True
    records = json.loads((output / "confirmed_missing_v1.json").read_text())
    assert records[0]["category_id"] == 24


def test_compact_decision_overlay_must_cover_and_compiles(
    tmp_path: Path, monkeypatch
) -> None:
    review = tmp_path / "review.csv"
    _write(review, "")
    overlay = tmp_path / "decisions.csv"
    overlay.write_text(
        "candidate_id,human_decision,review_note\n"
        "F0-1-2,ambiguous_ignore,visible object with uncertain class\n",
        encoding="utf-8",
    )
    output = tmp_path / "out"
    monkeypatch.setattr(
        "sys.argv",
        [
            "compile",
            "--review-csv",
            str(review),
            "--decision-csv",
            str(overlay),
            "--output-dir",
            str(output),
        ],
    )
    assert main() == 0
    decision = json.loads((output / "decision.json").read_text())
    assert decision["ignore_patch_admission"] is True
    assert decision["decision_csv_sha256"]
