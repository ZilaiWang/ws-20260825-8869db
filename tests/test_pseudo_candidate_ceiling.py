from scripts.analyze_cv3_pseudo_candidate_ceiling import _iou, _quantiles


def test_iou_xywh() -> None:
    assert _iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert _iou([0, 0, 10, 10], [10, 10, 2, 2]) == 0.0
    assert abs(_iou([0, 0, 10, 10], [5, 0, 10, 10]) - 1 / 3) < 1e-12


def test_quantiles_empty_and_order_independent() -> None:
    assert _quantiles([])["p50"] is None
    result = _quantiles([4.0, 1.0, 3.0, 2.0])
    assert result["min"] == 1.0
    assert result["max"] == 4.0
    assert result["p50"] in {2.0, 3.0}
