from __future__ import annotations

import pytest

from scripts.rerank_cv3_pseudo_with_tta_support import rerank_with_support


def _prediction(*, score: float, x: float = 0.0, category: int = 4) -> dict[str, object]:
    return {
        "image_id": 1,
        "category_id": category,
        "bbox": [x, 0.0, 10.0, 10.0],
        "score": score,
        "source_fold": 0,
    }


def test_supported_candidate_uses_geometric_score() -> None:
    output, counts = rerank_with_support(
        [_prediction(score=0.81)],
        [_prediction(score=0.49)],
        match_iou=0.5,
        unsupported_factor=0.25,
    )
    assert counts == {"supported": 1, "unsupported": 0}
    assert output[0]["score"] == pytest.approx(0.63)
    assert output[0]["tta_supported"] is True


def test_support_is_same_class_and_one_to_one() -> None:
    identity = [_prediction(score=0.8)]
    augmented = [
        _prediction(score=0.9),
        _prediction(score=0.7, x=0.5),
        _prediction(score=0.6, category=5),
    ]
    output, counts = rerank_with_support(
        identity,
        augmented,
        match_iou=0.5,
        unsupported_factor=0.25,
    )
    assert counts == {"supported": 1, "unsupported": 2}
    assert sum(bool(item["tta_supported"]) for item in output) == 1
    unsupported_scores = sorted(
        float(item["score"]) for item in output if not item["tta_supported"]
    )
    assert unsupported_scores == pytest.approx([0.15, 0.175])


@pytest.mark.parametrize("factor", [-0.1, 1.1])
def test_invalid_factor_is_rejected(factor: float) -> None:
    with pytest.raises(ValueError):
        rerank_with_support(
            [], [], match_iou=0.5, unsupported_factor=factor
        )
