from scripts.replace_prediction_fold import replace_fold


def _row(image_id: int, fold: int, score: float) -> dict:
    return {
        "image_id": image_id,
        "source_fold": fold,
        "score": score,
        "category_id": 24,
        "bbox": [0, 0, 1, 1],
    }


def test_replace_prediction_fold_changes_only_requested_fold() -> None:
    base = [_row(10, 0, 0.1), _row(20, 1, 0.2), _row(30, 2, 0.3)]
    candidate = [_row(20, 1, 0.9), _row(21, 1, 0.8)]
    output = replace_fold(base, candidate, 1)
    assert {(row["image_id"], row["score"]) for row in output} == {
        (10, 0.1),
        (20, 0.9),
        (21, 0.8),
        (30, 0.3),
    }
