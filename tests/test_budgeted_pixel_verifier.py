from __future__ import annotations

import numpy as np
import pytest

from scripts.simulate_budgeted_pixel_verifier import (
    align_pixel_scores,
    build_router_features,
    normalize_records,
    select_per_image_budget,
    validity_thresholds,
)


def _rows():
    return [
        {
            "image_id": 1,
            "category_id": 0,
            "bbox": [0.0, 0.0, 10.0, 20.0],
            "score": 0.8,
            "source_fold": 0,
            "source_variant": "UNION",
        },
        {
            "image_id": 1,
            "category_id": 24,
            "bbox": [50.0, 50.0, 5.0, 5.0],
            "score": 0.2,
            "source_fold": 0,
            "source_variant": "COPH",
        },
        {
            "image_id": 2,
            "category_id": 4,
            "bbox": [20.0, 30.0, 15.0, 10.0],
            "score": 0.4,
            "source_fold": 1,
            "source_variant": "UNION",
        },
    ]


def test_alignment_rejects_missing_and_preserves_order():
    base = normalize_records(_rows())
    pixel = [{**row, "score": value} for row, value in zip(reversed(_rows()), (0.3, 0.2, 0.1))]
    scores = align_pixel_scores(base, pixel)
    assert scores.tolist() == pytest.approx([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="missing"):
        align_pixel_scores(base, pixel[:-1])


def test_router_features_are_deployable_and_finite():
    records = normalize_records(_rows())
    mapping = {0: "ship", 4: "aircraft", 24: "vehicle"}
    matrix, columns = build_router_features(
        records,
        category_mapping=mapping,
        image_sizes={1: (100.0, 100.0), 2: (100.0, 100.0)},
    )
    assert matrix.shape == (3, len(columns))
    assert np.isfinite(matrix).all()
    assert matrix[1, columns.index("variant_coph")] == 1.0
    assert matrix[1, columns.index("fine_24")] == 1.0


def test_per_image_budget_is_exact_and_stable():
    records = normalize_records(_rows())
    selected = select_per_image_budget(records, np.asarray([0.5, 0.5, 0.1]), 1)
    assert selected.tolist() == [True, False, True]
    assert not select_per_image_budget(records, np.ones(3), 0).any()
    with pytest.raises(ValueError, match="non-negative"):
        select_per_image_budget(records, np.ones(3), -1)


def test_validity_thresholds_return_ordered_finite_values():
    labels = np.asarray([1, 1, 0, 0], dtype=np.int64)
    probabilities = np.asarray([0.9, 0.8, 0.4, 0.1], dtype=np.float64)
    thresholds = validity_thresholds(labels, probabilities, targets=(0.0, 0.5))
    assert thresholds == sorted(set(thresholds))
    assert all(0.0 < value < 1.0 for value in thresholds)
