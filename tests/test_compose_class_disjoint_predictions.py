import pytest

from scripts.compose_class_disjoint_predictions import compose


def _row(image_id: int, category_id: int, score: float) -> dict:
    return {
        "image_id": image_id,
        "category_id": category_id,
        "score": score,
        "bbox": [1, 2, 3, 4],
    }


def test_compose_uses_exact_label_owner() -> None:
    primary = [_row(1, 0, 0.5), _row(1, 24, 0.9)]
    expert = [_row(1, 0, 0.8), _row(1, 24, 0.6)]
    result = compose(
        primary,
        expert,
        primary_labels=frozenset(range(24)),
        expert_labels=frozenset({24}),
    )
    assert [(row["category_id"], row["score"]) for row in result] == [
        (24, 0.6),
        (0, 0.5),
    ]


def test_compose_rejects_overlap_or_gap() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        compose([], [], primary_labels=frozenset(range(25)), expert_labels=frozenset({24}))
    with pytest.raises(ValueError, match="cover"):
        compose([], [], primary_labels=frozenset(range(23)), expert_labels=frozenset({24}))


def test_compose_applies_independent_branch_thresholds() -> None:
    primary = [_row(1, 0, 0.59), _row(1, 1, 0.61), _row(1, 24, 0.99)]
    expert = [_row(1, 0, 0.99), _row(1, 24, 0.49), _row(1, 24, 0.51)]
    result = compose(
        primary,
        expert,
        primary_labels=frozenset(range(24)),
        expert_labels=frozenset({24}),
        primary_threshold=0.60,
        expert_threshold=0.50,
    )
    assert [(row["category_id"], row["score"]) for row in result] == [
        (1, 0.61),
        (24, 0.51),
    ]


def test_compose_rejects_invalid_branch_threshold() -> None:
    with pytest.raises(ValueError, match="expert_threshold"):
        compose(
            [],
            [],
            primary_labels=frozenset(range(24)),
            expert_labels=frozenset({24}),
            expert_threshold=1.01,
        )
