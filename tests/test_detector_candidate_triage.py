from scripts.triage_detector_candidate import triage


def _fixed(quality_shift: float = 0.0) -> dict:
    per_coarse = {}
    for coarse in ("ship", "aircraft", "vehicle"):
        per_coarse[coarse] = {
            "macro_recall": 0.8 + quality_shift / 100,
            "macro_fdr": 0.1,
            "recall_points": 80.0 + quality_shift,
            "fdr_points": 80.0,
        }
    return {
        "threshold": 0.5,
        "input_sha256": {"gt": "same"},
        "platform": {
            "metric_protocol": "platform_observed_20260831",
            "per_coarse": per_coarse,
        },
    }


def _frontier(quality: float) -> dict:
    return {"quality_oracle": {"selection_quality_score": quality}}


def test_triage_stops_when_fixed_and_oracle_both_decline() -> None:
    result = triage(
        _fixed(),
        _fixed(-1.0),
        _frontier(60.0),
        _frontier(59.0),
        positive_deadband=0.5,
    )
    assert result["fixed_quality"]["delta"] < 0
    assert result["single_split_oracle_diagnostic"]["delta"] < 0
    assert result["next_action"] == "stop_ranking_and_fixed_workpoint_both_worse"
    assert result["automatic_full_or_submission_admission"] is False


def test_triage_marks_calibration_only_when_oracle_recovers() -> None:
    result = triage(
        _fixed(),
        _fixed(-0.1),
        _frontier(60.0),
        _frontier(61.0),
        positive_deadband=0.5,
    )
    assert result["next_action"].startswith("calibration_diagnostic_only")


def test_triage_routes_clear_gain_to_confirmation_not_full() -> None:
    result = triage(
        _fixed(),
        _fixed(2.0),
        _frontier(60.0),
        _frontier(61.0),
        positive_deadband=0.5,
    )
    assert result["next_action"] == "candidate_for_frozen_hard_and_sentinel_confirmation"
    assert result["automatic_full_or_submission_admission"] is False
