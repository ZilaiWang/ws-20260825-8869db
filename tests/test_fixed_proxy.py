from copy import deepcopy

import pytest

from rsdet.experiments.fixed_proxy import quality_contribution, review_quality_delta


def platform():
    return {
        "metric_protocol": "platform_observed_20260831",
        "per_coarse": {
            name: {"macro_recall": 0.9, "macro_fdr": 0.1}
            for name in ("ship", "aircraft", "vehicle")
        },
    }


def test_quality_excludes_time_and_does_not_use_pooled_rates():
    value = platform()
    assert quality_contribution(value) == pytest.approx(65.71428571428571)
    value.update(latency_seconds=19, pooled_recall=0.01)
    assert quality_contribution(value) == pytest.approx(65.71428571428571)


def test_strict_direction_band_and_never_formal_admission():
    baseline = platform()
    assert not review_quality_delta(baseline, baseline, stage="sentinel", minimum=0)[
        "direction_pass"
    ]
    candidate = deepcopy(baseline)
    candidate["per_coarse"]["aircraft"]["macro_fdr"] = 0.05
    hard = review_quality_delta(baseline, candidate, stage="hard", minimum=0.5)
    assert hard["next_action"] == "evaluate_frozen_sentinel"
    assert hard["delta_quality"] == pytest.approx(10 / 7)
    sentinel = review_quality_delta(baseline, candidate, stage="sentinel", minimum=0)
    assert sentinel["next_action"] == "review_deployment_cost_and_risk"
    assert not hard["formal_admission"] and not sentinel["formal_admission"]


def test_reject_invalid_protocol_or_nonfinite_rate():
    value = platform()
    with pytest.raises(ValueError):
        quality_contribution(dict(value, metric_protocol="official_ranking_v1_6"))
    value["per_coarse"]["ship"]["macro_fdr"] = float("nan")
    with pytest.raises(ValueError):
        quality_contribution(value)
    with pytest.raises(ValueError):
        review_quality_delta(platform(), platform(), stage="blind", minimum=0)
