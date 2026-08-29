from __future__ import annotations

import numpy as np
import pytest

from scripts.rerank_cv3_pseudo_with_open_set_verifier import foreground_statistics
from scripts.train_pseudo_open_set_verifier import (
    BACKGROUND_CLASS_ID,
    balanced_batch_indices,
    target_class,
)


def test_target_class_uses_support_gt_and_explicit_background() -> None:
    assert target_class({"is_foreground": "0", "support_gt_category_id": ""}) == 25
    assert target_class({"is_foreground": "1", "support_gt_category_id": "17"}) == 17
    with pytest.raises(ValueError, match="lacks support"):
        target_class({"is_foreground": "1", "support_gt_category_id": ""})


def test_balanced_batches_are_half_background_and_cover_coarse_classes() -> None:
    rows = []
    for category in (0, 4, 5, 24):
        rows.extend(
            {"is_foreground": "1", "support_gt_category_id": str(category)}
            for _ in range(4)
        )
    rows.extend(
        {"is_foreground": "0", "support_gt_category_id": ""} for _ in range(16)
    )
    batches = balanced_batch_indices(rows, batch_size=12, batches_per_epoch=4, seed=7)
    assert len(batches) == 4
    for batch in batches:
        targets = [target_class(rows[index]) for index in batch]
        assert targets.count(BACKGROUND_CLASS_ID) == 6
        assert len(targets) == 12


def test_coarse_balanced_background_sampling_covers_all_groups() -> None:
    rows = []
    for category, coarse in ((0, "ship"), (4, "aircraft"), (24, "vehicle")):
        rows.extend(
            {
                "is_foreground": "1",
                "support_gt_category_id": str(category),
                "coarse": coarse,
            }
            for _ in range(4)
        )
        rows.extend(
            {
                "is_foreground": "0",
                "support_gt_category_id": "",
                "coarse": coarse,
            }
            for _ in range(4)
        )
    batches = balanced_batch_indices(
        rows,
        batch_size=60,
        batches_per_epoch=10,
        seed=7,
        background_sampling="coarse_balanced",
    )
    background_coarse = {
        rows[index]["coarse"]
        for batch in batches
        for index in batch
        if target_class(rows[index]) == BACKGROUND_CLASS_ID
    }
    assert background_coarse == {"ship", "aircraft", "vehicle"}


def test_invalid_background_sampling_is_rejected() -> None:
    with pytest.raises(ValueError, match="background_sampling"):
        balanced_batch_indices(
            [],
            batch_size=6,
            batches_per_epoch=1,
            seed=7,
            background_sampling="invalid",
        )


def test_foreground_statistics_separates_background_and_fine_probabilities() -> None:
    probabilities = np.zeros(26, dtype=np.float64)
    probabilities[3] = 0.60
    probabilities[7] = 0.25
    probabilities[25] = 0.15
    result = foreground_statistics(probabilities, predicted_class=3)
    assert result["foreground_probability"] == pytest.approx(0.85)
    assert result["predicted_class_probability"] == pytest.approx(0.60)
    assert result["conditional_predicted_class_probability"] == pytest.approx(0.60 / 0.85)
    assert result["top1_class"] == 3
    assert result["agree"] == 1
    assert result["margin"] == pytest.approx((0.60 - 0.25) / 0.85)
