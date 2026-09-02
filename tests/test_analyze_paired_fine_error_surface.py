import pytest

from scripts.analyze_paired_fine_error_surface import (
    _crossfit_thresholds,
    _filter_crossfit,
)


def test_crossfit_filter_uses_each_images_held_out_fold_threshold() -> None:
    predictions = {
        10: [{"score": 0.2}, {"score": 0.6}],
        11: [{"score": 0.2}, {"score": 0.6}],
        12: [{"score": 0.2}, {"score": 0.6}],
    }
    filtered = _filter_crossfit(
        predictions,
        {10: 0, 11: 1, 12: 2},
        {0: 0.1, 1: 0.5, 2: 0.7},
    )
    assert [len(filtered[image_id]) for image_id in (10, 11, 12)] == [2, 1, 0]


def test_crossfit_threshold_parser_fails_closed_on_missing_fold() -> None:
    frontier = {
        "frontiers": {"0.150": {"crossfit_thresholds": {"0": 0.1, "1": 0.2}}}
    }
    with pytest.raises(ValueError, match="folds 0, 1 and 2"):
        _crossfit_thresholds(frontier, "0.150")
