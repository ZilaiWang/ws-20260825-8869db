from __future__ import annotations

from scripts.train_pseudo_coarse_binary_verifier import (
    balanced_binary_batches,
    binary_target,
)


def test_balanced_binary_batches_balance_labels_and_fine_classes() -> None:
    rows = [
        {"is_foreground": "0", "support_gt_category_id": "", "score": "0.1"},
        {"is_foreground": "0", "support_gt_category_id": "", "score": "0.9"},
        {"is_foreground": "1", "support_gt_category_id": "2", "score": "0.1"},
        {"is_foreground": "1", "support_gt_category_id": "3", "score": "0.9"},
    ]
    batches = balanced_binary_batches(
        rows, batch_size=8, batches_per_epoch=4, seed=42
    )
    assert len(batches) == 4
    for batch in batches:
        assert sum(binary_target(rows[index]) for index in batch) == 4
        assert len(batch) == 8


def test_balanced_binary_batches_requires_both_labels() -> None:
    rows = [{"is_foreground": "1", "support_gt_category_id": "3"}]
    try:
        balanced_binary_batches(rows, batch_size=4, batches_per_epoch=1, seed=1)
    except ValueError as error:
        assert "positive and negative" in str(error)
    else:
        raise AssertionError("missing negatives must fail")


def test_score_sqrt_sampling_focuses_on_high_score_negatives() -> None:
    rows = [
        {"is_foreground": "0", "support_gt_category_id": "", "score": "0.0001"},
        {"is_foreground": "0", "support_gt_category_id": "", "score": "0.81"},
        {"is_foreground": "1", "support_gt_category_id": "2", "score": "0.01"},
    ]
    batches = balanced_binary_batches(
        rows,
        batch_size=20,
        batches_per_epoch=100,
        seed=42,
        negative_sampling="score_sqrt",
    )
    sampled_negatives = [
        index
        for batch in batches
        for index in batch
        if not binary_target(rows[index])
    ]
    assert sampled_negatives.count(1) > 20 * sampled_negatives.count(0)
