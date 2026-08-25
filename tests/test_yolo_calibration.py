"""Tests for the strict cross-fit YOLO calibration module."""

from __future__ import annotations

import pytest

from rsdet.postprocess.yolo_calibration import (
    ClassSpatialStatistics,
    SpatialAnnotation,
    calibrate_binary_score,
    estimate_class_spatial_statistics,
    estimate_fractal_dimension,
    filter_by_coarse_thresholds,
)
from scripts.y1_crossfit_calibration import build_decision


def test_fractal_dimension_neutral_for_tiny_class() -> None:
    assert estimate_fractal_dimension([(0.1, 0.1)] * 3) == 1.0


def test_fractal_dimension_is_bounded() -> None:
    points = [(x / 10.0, y / 10.0) for x in range(10) for y in range(10)]
    value = estimate_fractal_dimension(points)
    assert 1.0 < value <= 2.0


def test_statistics_use_only_selected_folds() -> None:
    records = []
    for category_id in range(25):
        records.extend(
            [
                SpatialAnnotation(category_id * 10 + 1, 0, category_id, 0.1, 0.1),
                SpatialAnnotation(category_id * 10 + 2, 1, category_id, 0.2, 0.2),
                SpatialAnnotation(category_id * 10 + 3, 2, category_id, 0.3, 0.3),
            ]
        )
    stats = estimate_class_spatial_statistics(records, source_folds=(0, 2))
    assert stats.source_folds == (0, 2)
    assert stats.counts == (2,) * 25
    assert sum(stats.priors) == pytest.approx(1.0)


def test_prior_calibration_preserves_probability_range() -> None:
    stats = ClassSpatialStatistics(
        counts=(10, 1),
        priors=(10 / 11, 1 / 11),
        fractal_dimensions=(1.8, 1.0),
        source_folds=(0, 1),
    )
    common = calibrate_binary_score(
        0.5,
        category_id=0,
        statistics=stats,
        beta=3.0,
        spatial_lambda=0.0,
    )
    rare = calibrate_binary_score(
        0.5,
        category_id=1,
        statistics=stats,
        beta=3.0,
        spatial_lambda=0.0,
    )
    assert 0.0 <= common <= 1.0
    assert 0.0 <= rare <= 1.0
    assert rare > common


def test_coarse_filtering() -> None:
    predictions = {
        1: [
            {"category_id": 0, "score": 0.2},
            {"category_id": 4, "score": 0.2},
            {"category_id": 24, "score": 0.2},
        ]
    }
    kept = filter_by_coarse_thresholds(
        predictions,
        {"ship": 0.1, "aircraft": 0.3, "vehicle": 0.1},
    )
    assert [item["category_id"] for item in kept[1]] == [0, 24]


def test_decision_selects_prior_without_fractal_increment() -> None:
    baseline = {
        "recall": 0.918,
        "fdr": 0.199,
        "official_gate_passed": True,
        "official_ranking": {"overall_macro_fdr": 0.238},
        "delta_vs_c0": {"recall": 0.0, "fdr": 0.0, "overall_macro_fdr": 0.0},
        "fold_direction_vs_c0": [],
    }
    prior = {
        "recall": 0.914,
        "fdr": 0.159,
        "official_gate_passed": True,
        "official_ranking": {"overall_macro_fdr": 0.208},
        "delta_vs_c0": {
            "recall": -0.004,
            "fdr": -0.04,
            "overall_macro_fdr": -0.03,
        },
        "fold_direction_vs_c0": [{"fdr_delta": -0.01}] * 3,
    }
    proxy = {
        **prior,
        "recall": 0.9139,
        "official_ranking": {"overall_macro_fdr": 0.2082},
    }
    result = {
        "merged_held_out": {
            "C0_global": baseline,
            "C1_coarse": {
                **baseline,
                "official_gate_passed": False,
                "delta_vs_c0": {
                    "recall": -0.001,
                    "fdr": 0.003,
                    "overall_macro_fdr": 0.002,
                },
            },
            "C2_prior": prior,
            "C3_fractal_proxy": proxy,
        }
    }
    decision = build_decision(result)
    assert decision["selected_method"] == "C2_prior"
    assert decision["methods"]["C2_prior"]["admitted"] is True
    assert decision["methods"]["C3_fractal_proxy"]["admitted"] is False
