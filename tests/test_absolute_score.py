from __future__ import annotations

import pytest

from rsdet.evaluation.absolute_score import (
    competition_score,
    fdr_points,
    latency_points,
    recall_points,
    score_coarse_interpretations,
)


def test_published_piecewise_boundaries() -> None:
    assert recall_points(0.0) == 0.0
    assert recall_points(0.85) == pytest.approx(60.0)
    assert recall_points(1.0) == pytest.approx(100.0)
    assert fdr_points(0.0) == pytest.approx(100.0)
    assert fdr_points(0.2) == pytest.approx(60.0)
    assert fdr_points(1.0) == pytest.approx(0.0)
    assert latency_points(0.0) == 100.0
    assert latency_points(20.0) == 80.0
    assert latency_points(20.0001) == 0.0


def test_weighted_total() -> None:
    result = competition_score(1.0, 0.0, 0.0)
    assert result["total_score"] == pytest.approx(100.0)


def test_coarse_interpretations_and_pooled_counts() -> None:
    rows = {
        "ship": {"recall": 0.9, "fdr": 0.1, "tp": 90, "fp": 10, "fn": 10},
        "aircraft": {"recall": 1.0, "fdr": 0.0, "tp": 100, "fp": 0, "fn": 0},
        "vehicle": {"recall": 0.5, "fdr": 0.5, "tp": 5, "fp": 5, "fn": 5},
    }
    result = score_coarse_interpretations(rows, 2.0)
    assert result["aggregation_is_not_stated_in_public_formula"] is True
    assert set(result["mean_per_coarse_score"]["per_coarse"]) == {
        "ship",
        "aircraft",
        "vehicle",
    }
    assert result["pooled_counts"]["tp"] == 195
    assert result["pooled_counts"]["fp"] == 15
    assert result["pooled_counts"]["fn"] == 15


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_invalid_rates_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        recall_points(value)
