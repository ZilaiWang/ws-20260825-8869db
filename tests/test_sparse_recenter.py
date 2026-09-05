from rsdet.pipeline.sparse_recenter import (
    RecenterCandidate,
    boundary_risk_requests,
    centered_window,
    cluster_same_fine,
    select_windows,
)


def _candidate(score: float, tile_id: int, *, x: float = 100.0) -> RecenterCandidate:
    return RecenterCandidate(
        box=(x, 100.0, x + 100.0, 200.0),
        local_box=(4.0, 100.0, 104.0, 200.0),
        score=score,
        label=24,
        tile_id=tile_id,
        tile_width=1024,
        tile_height=1024,
        internal_edges=(True, False, True, True),
    )


def test_risk_request_and_centered_window_are_deterministic() -> None:
    clusters = cluster_same_fine([_candidate(0.50, 0), _candidate(0.40, 1)])
    requests = boundary_risk_requests(
        clusters,
        rescue_floor=0.25,
        output_threshold=0.536,
        overlap_size=256,
    )
    assert len(requests) == 1
    assert "risk_band" in requests[0].reasons
    assert "poor_context" in requests[0].reasons
    window = centered_window(requests[0], image_width=10000, image_height=10000, window_size=1024)
    assert window.x_start == 0
    assert window.y_start == 0


def test_window_selection_honours_budget_and_deduplicates() -> None:
    clusters = cluster_same_fine(
        [_candidate(0.50, 0), _candidate(0.49, 1, x=120.0), _candidate(0.48, 2, x=5000.0)]
    )
    requests = boundary_risk_requests(
        clusters,
        rescue_floor=0.25,
        output_threshold=0.536,
        overlap_size=256,
    )
    windows = select_windows(requests, image_width=10000, image_height=10000, max_windows=2)
    assert len(windows) == 2
    assert windows[0].priority >= windows[1].priority
