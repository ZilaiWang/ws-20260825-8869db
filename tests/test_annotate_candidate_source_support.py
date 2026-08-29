from __future__ import annotations

import numpy as np

from scripts.annotate_candidate_source_support import (
    annotate_source_support,
    pairwise_iou,
)


def _row(x: float, *, score: float = 0.8, category: int = 3) -> dict:
    return {
        "image_id": 1,
        "category_id": category,
        "bbox": [x, 0.0, 10.0, 10.0],
        "score": score,
    }


def test_pairwise_iou_handles_empty_and_overlap() -> None:
    empty = np.empty((0, 4), dtype=np.float64)
    one = np.asarray([[0.0, 0.0, 10.0, 10.0]])
    assert pairwise_iou(empty, one).shape == (0, 1)
    result = pairwise_iou(one, np.asarray([[5.0, 0.0, 15.0, 10.0]]))
    assert np.isclose(result[0, 0], 1.0 / 3.0)


def test_source_support_is_same_fine_and_no_gt() -> None:
    candidates = [_row(0.0)]
    sources = [
        ("y5_rot", [_row(0.0, score=0.7), _row(0.0, score=0.99, category=2)]),
        ("m3_id", [_row(1.0, score=0.6)]),
        ("coph", [_row(8.0, score=0.9)]),
    ]
    output = annotate_source_support(candidates, sources, support_iou=0.5)
    row = output[0]
    assert row["source_support_count"] == 2
    assert row["heterogeneous_support"] == 1
    assert np.isclose(row["source_support_score_sum"], 1.3)
    assert np.isclose(row["support_y5_rot_score"], 0.7)
    assert row["support_coph_score"] == 0.0


def test_source_labels_must_be_unique() -> None:
    try:
        annotate_source_support([_row(0.0)], [("x", []), ("x", [])], support_iou=0.5)
    except ValueError as error:
        assert "unique" in str(error)
    else:
        raise AssertionError("duplicate labels must fail")
