from __future__ import annotations

import pytest

from scripts.analyze_pseudo_workpoint_errors import apply_fold_thresholds


def test_apply_fold_thresholds_uses_each_images_heldout_threshold():
    predictions = {
        1: [{"score": 0.5}, {"score": 0.4}],
        2: [{"score": 0.5}, {"score": 0.4}],
        3: [{"score": 0.5}, {"score": 0.4}],
    }
    selected = apply_fold_thresholds(
        predictions,
        {1: 0, 2: 1, 3: 2},
        {0: 0.45, 1: 0.50, 2: 0.60},
    )
    assert [len(selected[key]) for key in (1, 2, 3)] == [1, 1, 0]


def test_apply_fold_thresholds_rejects_incomplete_contract():
    with pytest.raises(ValueError, match="folds 0, 1 and 2"):
        apply_fold_thresholds({}, {}, {0: 0.5})
