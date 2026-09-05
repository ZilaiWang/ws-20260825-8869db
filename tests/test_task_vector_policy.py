from __future__ import annotations

import pytest

from rsdet.experiments.task_vector_policy import (
    percentile,
    score_from_fine_counts,
    select_conservative_alpha,
    stress_incremental_vehicle_fp,
)


def test_conservative_alpha_breaks_score_ties_downward() -> None:
    assert select_conservative_alpha({0.0: 70.0, 0.125: 71.0, 0.25: 71.0}) == 0.125


def test_stress_multiplies_only_positive_incremental_fp() -> None:
    baseline = {24: {"tp": 8, "fp": 2, "fn": 2}}
    candidate = {24: {"tp": 9, "fp": 4, "fn": 1}}
    stressed = stress_incremental_vehicle_fp(baseline, candidate, multiplier=6)
    assert stressed[24] == {"tp": 9, "fp": 14, "fn": 1}
    improved = {24: {"tp": 9, "fp": 1, "fn": 1}}
    assert stress_incremental_vehicle_fp(baseline, improved)[24]["fp"] == 1


def test_score_from_fine_counts_uses_fine_macro() -> None:
    mapping = {0: "ship", 1: "ship", 2: "aircraft", 3: "vehicle"}
    counts = {
        0: {"tp": 10, "fp": 0, "fn": 0},
        1: {"tp": 0, "fp": 1, "fn": 1},
        2: {"tp": 10, "fp": 0, "fn": 0},
        3: {"tp": 10, "fp": 0, "fn": 0},
    }
    result = score_from_fine_counts(counts, mapping)
    assert result["per_coarse"]["ship"]["recall"] == pytest.approx(0.5)
    assert result["per_coarse"]["ship"]["fdr"] == pytest.approx(0.5)


def test_percentile_linear_interpolation() -> None:
    assert percentile([0, 10], 0.1) == pytest.approx(1.0)
