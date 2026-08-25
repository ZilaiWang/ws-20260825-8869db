from scripts.yolo_fast_screen import _clip_prediction_box, build_screen_decision


def test_clip_prediction_box_filters_degenerate_after_image_clipping():
    assert _clip_prediction_box([-5, 2, -1, 8], width=100, height=100) is None
    assert _clip_prediction_box([2, 101, 8, 110], width=100, height=100) is None
    assert _clip_prediction_box([-5, 2, 8, 110], width=100, height=100) == [
        0.0,
        2.0,
        8.0,
        98.0,
    ]


def _metrics(**overrides):
    values = {
        "overall_recall": 0.91,
        "overall_fdr": 0.18,
        "macro_recall": 0.86,
        "macro_fdr": 0.21,
        "vehicle_recall": 0.62,
        "vehicle_fdr": 0.20,
        "vehicle_floor_recall": 0.70,
        "vehicle_no_candidate": 30,
    }
    values.update(overrides)
    return values


def test_fast_screen_strong_vehicle_signal_enters_formal_queue():
    decision = build_screen_decision(
        _metrics(),
        _metrics(
            overall_recall=0.912,
            overall_fdr=0.185,
            macro_recall=0.861,
            vehicle_recall=0.655,
            vehicle_floor_recall=0.74,
            vehicle_no_candidate=24,
        ),
    )
    assert decision["next_action"] == "promising_for_formal_cv3"
    assert decision["formal_admission"] is False


def test_fast_screen_marginal_signal_requires_second_fold():
    decision = build_screen_decision(
        _metrics(),
        _metrics(vehicle_recall=0.642, vehicle_no_candidate=29),
    )
    assert decision["next_action"] == "promising_for_second_screen_fold"


def test_fast_screen_stops_unsafe_candidate():
    decision = build_screen_decision(
        _metrics(),
        _metrics(overall_recall=0.88, vehicle_recall=0.67, vehicle_no_candidate=20),
    )
    assert decision["next_action"] == "stop_candidate"
    assert decision["checks"]["overall_recall_safety"] is False
