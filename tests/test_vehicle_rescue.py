from copy import deepcopy

import pytest

from rsdet.experiments.fixed_proxy import quality_contribution, review_quality_delta
from rsdet.postprocess.vehicle_rescue import append_vehicle_rescue


def row(label=24, score=0.8, box=(0, 0, 10, 10)):
    return {"category_id": label, "score": score, "bbox_xyxy": list(box)}


def test_incumbents_preserved_auxiliary_nonvehicle_ignored():
    base = {1: [row(0), row(5), row(24, 0.6)], 2: []}
    original = deepcopy(base)
    aux = {1: [row(24, 0.99), row(0, box=(20, 20, 30, 30)),
               row(24, box=(40, 40, 50, 50))], 2: [row()]}
    output, stats = append_vehicle_rescue(base, aux)
    assert base == original
    assert output[1][:3] == base[1]
    assert output[1][3:] == [row(24, box=(40, 40, 50, 50))]
    assert output[2] == [row()]
    assert stats == {"auxiliary_vehicle": 3, "suppressed_overlap": 1, "added_vehicle": 2}
    output[1][0]["bbox_xyxy"][0] = 99
    assert base == original


def test_deterministic_self_dedup_and_empty_views():
    a, b = row(box=(20, 20, 30, 30)), row(box=(0, 0, 10, 10))
    one, _ = append_vehicle_rescue({1: []}, {1: [a, b, a]})
    two, _ = append_vehicle_rescue({1: []}, {1: [b, a, a]})
    assert one == two == {1: [b, a]}
    assert append_vehicle_rescue({1: []}, {1: []})[0] == {1: []}


@pytest.mark.parametrize("bad", [row(score=float("nan")), row(label=25),
                                  row(box=(0, 0, 0, 3)), row(box=(0, 0, float("inf"), 3))])
def test_invalid_input_rejected_even_for_ignored_category(bad):
    with pytest.raises(ValueError):
        append_vehicle_rescue({1: []}, {1: [bad]})


def test_universe_and_threshold_validation():
    with pytest.raises(ValueError):
        append_vehicle_rescue({1: []}, {2: []})
    for t in (0, -1, 1.01, float("nan")):
        with pytest.raises(ValueError):
            append_vehicle_rescue({}, {}, dedup_iou=t)


def platform(r=0.9, f=0.1):
    return {"metric_protocol": "platform_observed_20260831",
            "per_coarse": {c: {"macro_recall": r, "macro_fdr": f}
                           for c in ("ship", "aircraft", "vehicle")}}


def test_fixed_review_has_no_automatic_deployment_or_all_rates_gate():
    baseline, candidate = platform(), platform()
    candidate["per_coarse"]["vehicle"]["macro_recall"] = 0.94
    assert quality_contribution(candidate) > quality_contribution(baseline)
    result = review_quality_delta(baseline, candidate, stage="hard", minimum=0.5)
    assert result["direction_pass"]
    assert result["next_action"] == "evaluate_frozen_sentinel"
    assert not result["formal_admission"]
    assert not review_quality_delta(baseline, baseline, stage="sentinel", minimum=0)["direction_pass"]
    with pytest.raises(ValueError):
        quality_contribution({"metric_protocol": "legacy_pooled"})
