from copy import deepcopy

import pytest

from rsdet.postprocess.aircraft_relabel import relabel_aircraft


def case():
    rows = {
        1: [
            {"category_id": 24, "score": 0.85, "bbox_xyxy": [0, 0, 10, 10]},
            {"category_id": 4, "score": 0.6, "bbox_xyxy": [20, 0, 30, 10]},
        ]
    }
    bundle = {
        "image_id": 1,
        "prediction_index": 1,
        "old_category": 4,
        "probabilities": [0.0, 0.9, 0.1] + [0.0] * 17,
    }
    return rows, bundle


def test_relabel_preserves_scores_boxes_bypass_and_inputs():
    rows, bundle = case()
    before = deepcopy(rows)
    result = relabel_aircraft(rows, [bundle])
    assert rows == before
    assert result[1][0] == rows[1][0]
    assert result[1][1] == dict(rows[1][1], category_id=5)
    assert relabel_aircraft(rows, [bundle], min_probability=0.91) == rows


@pytest.mark.parametrize(
    "change", ["duplicate", "missing", "stale", "nonair", "nan", "length", "sum"]
)
def test_reject_invalid_bundles(change):
    rows, bundle = case()
    bundles = [bundle]
    if change == "duplicate":
        bundles.append(deepcopy(bundle))
    if change == "missing":
        bundles = []
    if change == "stale":
        bundle["old_category"] = 5
    if change == "nonair":
        bundle["prediction_index"] = 0
    if change == "nan":
        bundle["probabilities"][0] = float("nan")
    if change == "length":
        bundle["probabilities"].pop()
    if change == "sum":
        bundle["probabilities"][0] = 0.5
    with pytest.raises(ValueError):
        relabel_aircraft(rows, bundles)


def test_same_label_nms_after_relabel_and_empty_input():
    rows, bundle = case()
    rows[1].append(dict(rows[1][1], category_id=5, score=0.5))
    other = dict(bundle, prediction_index=2, old_category=5)
    result = relabel_aircraft(rows, [bundle, other])
    assert len(result[1]) == 2
    assert result[1][1]["score"] == 0.6
    assert relabel_aircraft({1: []}, []) == {1: []}
