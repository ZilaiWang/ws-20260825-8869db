from scripts.analyze_resolution_route_thresholds import select_point


def _point(threshold: float, quality: float, recall: float, fdr_ok: bool = True) -> dict:
    return {
        "expert_threshold": threshold,
        "mean_quality_delta": quality,
        "worst_vehicle_recall_delta_pp": recall,
        "all_fold_fdr_not_worse": fdr_ok,
    }


def test_select_point_prefers_quality_inside_recall_guard() -> None:
    protected, oracle = select_point(
        [
            _point(0.48, 1.2, -1.5),
            _point(0.52, 1.8, -2.1),
            _point(0.56, 2.5, -3.0),
        ],
        max_vehicle_recall_drop_pp=2.2,
    )
    assert protected["expert_threshold"] == 0.52
    assert oracle["expert_threshold"] == 0.56


def test_select_point_fails_closed_without_positive_safe_point() -> None:
    protected, oracle = select_point(
        [_point(0.48, -0.1, -1.0), _point(0.52, 2.0, -3.0)],
        max_vehicle_recall_drop_pp=2.2,
    )
    assert protected is None
    assert oracle["expert_threshold"] == 0.52
