from __future__ import annotations

import numpy as np

from scripts.calibrate_pixel_score_with_support_prior import (
    combine_probability_with_prior,
    crossfit_support_prior_scores,
    smoothed_rate,
)


def test_probability_is_unchanged_when_prior_matches_global() -> None:
    value = combine_probability_with_prior(0.4, support_rate=0.2, global_rate=0.2)
    assert np.isclose(value, 0.4)
    assert smoothed_rate(1, 2, alpha=1.0) == 0.5


def test_support_prior_is_crossfit_and_monotone() -> None:
    rows = []
    labels = []
    for fold in (0, 1, 2):
        rows.extend(
            [
                {"source_fold": fold, "source_support_count": 1, "score": 0.5},
                {"source_fold": fold, "source_support_count": 4, "score": 0.5},
            ]
        )
        labels.extend([0, 1])
    outputs, audits = crossfit_support_prior_scores(
        rows, np.asarray(labels, dtype=np.int64), alpha=1.0
    )
    assert len(audits) == 3
    for start in range(0, 6, 2):
        assert outputs[start + 1]["score"] > outputs[start]["score"]
