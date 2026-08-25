from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image

from rsdet.analysis.fp_bg_review import (
    load_formal_image_index,
    load_review_cards,
    render_review_card,
    write_review_outputs,
)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_join_render_and_write(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    image_path = data_root / "images" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (100, 80), (100, 120, 140)).save(image_path)
    formal = tmp_path / "formal.csv"
    fields = [
        "formal_image_id",
        "source_relative_path",
        "annotation_uid",
        "class_id",
        "gt_x0",
        "gt_y0",
        "gt_x1",
        "gt_y1",
    ]
    row = {
        "formal_image_id": 7,
        "source_relative_path": "images/sample.jpg",
        "annotation_uid": "ann-1",
        "class_id": 4,
        "gt_x0": 20,
        "gt_y0": 20,
        "gt_x1": 50,
        "gt_y1": 50,
    }
    _write_csv(formal, fields, [row, row])
    audit = tmp_path / "audit.csv"
    _write_csv(
        audit,
        [
            "audit_uid",
            "proposal_uid",
            "image_id",
            "category_id",
            "class_name",
            "score",
            "bbox_xyxy",
        ],
        [
            {
                "audit_uid": "hidden-repeat-metadata",
                "proposal_uid": "proposal-1",
                "image_id": 7,
                "category_id": 4,
                "class_name": "aircraft",
                "score": 0.2,
                "bbox_xyxy": "[18, 18, 52, 52]",
            }
        ],
    )
    index = load_formal_image_index(formal)
    assert len(index[7][1]) == 1
    cards = load_review_cards(audit, index)
    assert cards[0].card_id == "card-0000"
    card_image = render_review_card(cards[0], Image.open(image_path), panel_size=64)
    assert card_image.size == (192, 116)
    summary = write_review_outputs(cards, data_root, tmp_path / "review")
    assert summary["card_count"] == 1
    assert summary["automatic_background_admission"] is False
    decision = (tmp_path / "review" / "manual_review_decisions.csv").read_text()
    mapping = (tmp_path / "review" / "sealed_card_mapping.csv").read_text()
    assert "hidden-repeat-metadata" not in decision
    assert "proposal-1" not in decision
    assert "hidden-repeat-metadata" in mapping
    rendered = (tmp_path / "review" / "cards" / "card-0000.jpg").read_bytes()
    assert b"hidden-repeat-metadata" not in rendered
