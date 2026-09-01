from __future__ import annotations

from rsdet.data.background_regularization import (
    make_source_diverse_groups,
    select_training_background_rows,
)


def _row(index: int, source: str) -> dict[str, object]:
    return {
        "image_id": index,
        "candidate_key": f"{index:04d}",
        "source_file_name": f"images/train/{source}",
    }


def test_select_training_background_rows_excludes_validation_sources() -> None:
    rows = [_row(0, "train.jpg"), _row(1, "val.jpg")]
    selected, summary = select_training_background_rows(
        rows,
        train_images=["/data/images/train/train.jpg"],
        val_images=["/data/images/train/val.jpg"],
    )
    assert [row["image_id"] for row in selected] == [0]
    assert summary == {"train_row_count": 1, "val_row_count": 1, "unknown_row_count": 0}


def test_source_diverse_groups_are_deterministic_and_unique() -> None:
    rows = [
        _row(0, "a.jpg"),
        _row(1, "a.jpg"),
        _row(2, "b.jpg"),
        _row(3, "c.jpg"),
        _row(4, "d.jpg"),
        _row(5, "e.jpg"),
        _row(6, "f.jpg"),
        _row(7, "g.jpg"),
    ]
    groups, leftovers = make_source_diverse_groups(rows)
    assert len(groups) == 2
    assert not leftovers
    for group in groups:
        sources = [row["source_file_name"] for row in group]
        assert len(sources) == len(set(sources)) == 4
    groups_again, leftovers_again = make_source_diverse_groups(list(reversed(rows)))
    assert groups == groups_again
    assert leftovers == leftovers_again
