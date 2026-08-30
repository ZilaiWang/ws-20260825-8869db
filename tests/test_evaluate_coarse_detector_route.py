from scripts.evaluate_coarse_detector_route import route_predictions


def test_route_predictions_uses_primary_except_specialist_class() -> None:
    primary = {
        1: [
            {"category_id": 0, "score": 0.7},
            {"category_id": 24, "score": 0.9},
        ]
    }
    specialist = {
        1: [
            {"category_id": 0, "score": 0.99},
            {"category_id": 24, "score": 0.8},
        ]
    }
    routed = route_predictions(
        primary,
        specialist,
        category_mapping={0: "ship", 24: "vehicle"},
        primary_threshold=0.6,
        specialist_threshold=0.75,
    )
    assert routed == {1: [{"category_id": 0, "score": 0.7}, {"category_id": 24, "score": 0.8}]}


def test_route_predictions_applies_separate_thresholds() -> None:
    routed = route_predictions(
        {2: [{"category_id": 0, "score": 0.4}]},
        {2: [{"category_id": 24, "score": 0.5}]},
        category_mapping={0: "ship", 24: "vehicle"},
        primary_threshold=0.5,
        specialist_threshold=0.6,
    )
    assert routed == {2: []}
