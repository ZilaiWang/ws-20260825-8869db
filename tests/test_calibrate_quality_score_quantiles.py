from __future__ import annotations

import numpy as np

from scripts.calibrate_quality_score_quantiles import quantile_map


def test_quantile_map_is_monotone_and_handles_duplicate_knots() -> None:
    values = np.asarray([0.0, 0.2, 0.5, 1.0])
    x = np.asarray([0.0, 0.2, 0.2, 1.0])
    y = np.asarray([0.1, 0.3, 0.5, 0.9])
    output = quantile_map(values, x, y)
    assert np.all(np.diff(output) >= 0.0)
    assert np.allclose(output[[0, -1]], [0.1, 0.9])
