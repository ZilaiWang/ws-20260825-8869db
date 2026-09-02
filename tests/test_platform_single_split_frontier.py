import pytest

from scripts.analyze_single_split_official_frontier import (
    _select_platform_point,
    _select_quality_point,
)
from scripts.decide_peer_normal_screen import decide


def _point(
    threshold: float,
    *,
    pooled_recall: float,
    platform_recall: float,
    platform_fdr: float,
) -> dict:
    return {
        "threshold": threshold,
        "overall_recall": pooled_recall,
        "overall_fdr": 0.1,
        "platform_gate_recall": platform_recall,
        "platform_gate_fdr": platform_fdr,
    }


def _frontier(
    gate_recall: float,
    gate_fdr: float,
    *,
    ship_recall: float = 0.7,
) -> dict:
    per_coarse_recall = {
        "ship": ship_recall,
        "aircraft": 0.9,
        "vehicle": 3 * gate_recall - ship_recall - 0.9,
    }
    point = {
        "pooled_recall": 0.9,
        "pooled_fdr": 0.1,
        "fine25_macro_recall": 0.85,
        "platform": {
            "metric_protocol": "platform_observed_20260831",
            "gate_recall": gate_recall,
            "gate_fdr": gate_fdr,
            "per_coarse": {
                name: {"macro_recall": value, "macro_fdr": gate_fdr}
                for name, value in per_coarse_recall.items()
            },
        },
    }
    return {
        "input_sha256": {"gt": "same"},
        "frontiers": {"0.150": point},
    }


def test_frontier_selects_platform_metric_not_pooled_metric() -> None:
    selected = _select_platform_point(
        [
            _point(
                0.1,
                pooled_recall=0.99,
                platform_recall=0.70,
                platform_fdr=0.14,
            ),
            _point(
                0.2,
                pooled_recall=0.90,
                platform_recall=0.80,
                platform_fdr=0.14,
            ),
        ],
        0.15,
    )
    assert selected["threshold"] == pytest.approx(0.2)


def test_quality_oracle_uses_six_subscore_quality_not_pooled_recall() -> None:
    selected = _select_quality_point(
        [
            {**_point(0.1, pooled_recall=0.99, platform_recall=0.7, platform_fdr=0.1),
             "platform_quality_score": 70.0},
            {**_point(0.2, pooled_recall=0.90, platform_recall=0.8, platform_fdr=0.1),
             "platform_quality_score": 80.0},
        ]
    )
    assert selected["threshold"] == pytest.approx(0.2)


def test_normal_screen_admits_platform_gain_with_stable_fdr() -> None:
    result = decide(
        _frontier(0.75, 0.18),
        _frontier(0.76, 0.17, ship_recall=0.705),
    )
    assert result["status"] == "complete_platform_screen_admitted"
    assert result["next_action"] == "run_fixed_hard_sentinel_tiled_screen"


def test_normal_screen_rejects_pooled_gain_when_platform_recall_drops() -> None:
    baseline = _frontier(0.75, 0.18)
    candidate = _frontier(0.74, 0.17)
    candidate["frontiers"]["0.150"]["pooled_recall"] = 0.99
    result = decide(baseline, candidate)
    assert result["status"] == "complete_platform_screen_rejected"
    assert not result["gates"]["platform_recall_gain_ge_0p5pp"]


def test_normal_screen_fails_closed_on_legacy_frontier() -> None:
    baseline = _frontier(0.75, 0.18)
    del baseline["frontiers"]["0.150"]["platform"]
    with pytest.raises(ValueError, match="platform_observed"):
        decide(baseline, _frontier(0.76, 0.17))
