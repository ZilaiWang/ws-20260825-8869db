from __future__ import annotations

import pytest

from scripts.evaluate_pseudo_with_frozen_thresholds import (
    absolute_score_thresholds_from_frontier,
    thresholds_from_frontier,
)


def test_thresholds_from_frontier() -> None:
    payload = {
        "frontiers": {
            "0.150": {"crossfit_thresholds": {"0": 0.4, "1": 0.5, "2": 0.6}}
        }
    }
    assert thresholds_from_frontier(payload, 0.15) == {0: 0.4, 1: 0.5, 2: 0.6}
    with pytest.raises(ValueError):
        thresholds_from_frontier(payload, 0.2)


def test_absolute_score_thresholds_from_frontier() -> None:
    payload = {
        "absolute_score_crossfit": {
            "crossfit_thresholds": {"0": 0.11, "1": 0.22, "2": 0.33}
        }
    }
    assert absolute_score_thresholds_from_frontier(payload) == {
        0: 0.11,
        1: 0.22,
        2: 0.33,
    }
