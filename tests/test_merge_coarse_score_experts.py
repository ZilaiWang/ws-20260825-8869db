from __future__ import annotations

import pytest

from scripts.merge_coarse_score_experts import align_scores, merge_scores


def _row(image, category, score):
    return {
        "image_id": image,
        "category_id": category,
        "bbox": [1.0, 2.0, 3.0, 4.0],
        "score": score,
        "source_fold": 0,
    }


def test_align_scores_is_order_independent_and_exact():
    reference = [_row(1, 0, 0.1), _row(2, 4, 0.2)]
    expert = [_row(2, 4, 0.8), _row(1, 0, 0.9)]
    assert align_scores(reference, expert) == [0.9, 0.8]
    with pytest.raises(ValueError, match="missing"):
        align_scores(reference, expert[:1])


def test_merge_scores_uses_only_configured_coarse_expert():
    reference = [_row(1, 0, 0.1), _row(2, 4, 0.2), _row(3, 24, 0.3)]
    output = merge_scores(
        reference,
        {"direct": [0.4, 0.5, 0.6], "identity": [0.7, 0.8, 0.9]},
        routes={"ship": "identity", "aircraft": "identity", "vehicle": "direct"},
        category_mapping={0: "ship", 4: "aircraft", 24: "vehicle"},
    )
    assert [row["score"] for row in output] == [0.7, 0.8, 0.6]
    assert [row["score_expert"] for row in output] == ["identity", "identity", "direct"]
