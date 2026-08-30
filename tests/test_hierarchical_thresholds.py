from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from rsdet.evaluation.hierarchical_thresholds import (
    filter_by_thresholds,
    select_threshold,
    shrink_threshold,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_cv3_oof_hierarchical_thresholds.py"
SPEC = importlib.util.spec_from_file_location("hierarchical_threshold_cli", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_select_threshold_maximizes_recall_under_fdr() -> None:
    points = [
        {"threshold": 0.1, "overall_recall": 0.95, "overall_fdr": 0.20},
        {"threshold": 0.3, "overall_recall": 0.91, "overall_fdr": 0.14},
        {"threshold": 0.5, "overall_recall": 0.84, "overall_fdr": 0.05},
    ]
    assert select_threshold(points, 0.15)["threshold"] == pytest.approx(0.3)


def test_shrink_threshold_uses_anchor_for_tiny_class() -> None:
    selected, weight = shrink_threshold(
        0.1,
        0.5,
        evidence=4,
        prior_strength=50,
        minimum_evidence=10,
    )
    assert weight == 0.0
    assert selected == pytest.approx(0.5)


def test_shrink_threshold_moves_toward_fine_with_evidence() -> None:
    selected, weight = shrink_threshold(
        0.1,
        0.5,
        evidence=100,
        prior_strength=50,
        minimum_evidence=10,
    )
    assert weight == pytest.approx(2 / 3)
    assert 0.1 < selected < 0.5


def test_filter_by_thresholds_preserves_empty_images() -> None:
    predictions = {
        1: [
            {"category_id": 0, "score": 0.2},
            {"category_id": 1, "score": 0.8},
        ],
        2: [],
    }
    filtered = filter_by_thresholds(predictions, {0: 0.3, 1: 0.7})
    assert filtered == {1: [{"category_id": 1, "score": 0.8}], 2: []}


def test_filter_by_thresholds_rejects_missing_class() -> None:
    with pytest.raises(ValueError, match="missing thresholds"):
        filter_by_thresholds({1: [{"category_id": 3, "score": 0.9}]}, {0: 0.2})


def test_legacy_bbox_xyxy_loader(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "image_id": 7,
                    "category_id": 2,
                    "bbox_xyxy": [1.0, 2.0, 4.0, 6.0],
                    "score": 0.8,
                }
            ]
        ),
        encoding="utf-8",
    )
    assert MODULE._load_predictions_compat(path) == {
        7: [
            {
                "bbox_xyxy": [1.0, 2.0, 4.0, 6.0],
                "category_id": 2,
                "score": 0.8,
            }
        ]
    }
