from scripts.analyze_quality_route_cv3 import _coarse_utility, _merge_routes


def test_merge_routes_preserves_identity_bypass_rows() -> None:
    routes = {
        "ship": {1: [{"category_id": 0}]},
        "aircraft": {1: [{"category_id": 4}]},
        "vehicle": {1: [{"category_id": 24}]},
    }
    assert [row["category_id"] for row in _merge_routes(routes, {1})[1]] == [0, 4, 24]


def test_coarse_utility_is_sum_of_two_platform_subscores() -> None:
    payload = {
        "score_payload": {
            "per_coarse": {
                "ship": {"recall_points": 81.0, "fdr_points": 73.0}
            }
        }
    }
    assert _coarse_utility(payload, "ship") == 154.0
