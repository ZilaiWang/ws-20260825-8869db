import copy

import pytest

from rsdet.contracts import Prediction
from rsdet.postprocess.recall_calibration import (
    TARGETS,
    apply_recall_thresholds,
    select_recall_thresholds,
)
from rsdet.postprocess.thresholds import filter_prediction_by_thresholds


def example():
    curves = {c: [{"threshold": t, "overall_recall": r, "overall_fdr": f}
                  for t, r, f in ((.001, 1., .99), (.35, .8, .10), (.4, .7, .09), (.5, .5, .1), (.7, .4, .05))]
              for c in TARGETS}
    support = {c: (200, 10) for c in TARGETS}
    return curves, support


def test_lower_only_actual_macro_with_support_caps():
    curves, support = example()
    fit = select_recall_thresholds(curves, incumbent=.5, support=support)
    assert all(0 < t <= .5 for t in fit["thresholds"].values())
    assert all(fit["thresholds"][c] == .5 for c in range(4, 24))
    assert all(fit["thresholds"][c] < .5 for c in TARGETS)
    assert all(a["proposed_coarse_r_f_quality"][1] <= .15 for a in fit["fine_audit"].values())


@pytest.mark.parametrize("support_value", [(9, 5), (100, 1)])
def test_insufficient_support_abstains(support_value):
    curves, support = example()
    support[0] = support_value
    fit = select_recall_thresholds(curves, incumbent=.5, support=support)
    assert fit["thresholds"][0] == .5
    assert not fit["fine_audit"]["0"]["accepted_on_normal_fit"]


def test_excessive_false_positive_cost_abstains():
    curves, support = example()
    for ps in curves.values():
        for p in ps:
            if p["threshold"] < .5:
                p["overall_fdr"] = .9
    fit = select_recall_thresholds(curves, incumbent=.5, support=support)
    assert all(t == .5 for t in fit["thresholds"].values())


@pytest.mark.parametrize("mutation", ["nan", "missing_incumbent", "duplicate", "negative_support"])
def test_bad_inputs_rejected(mutation):
    curves, support = example()
    if mutation == "nan":
        curves[0][0]["overall_recall"] = float("nan")
    elif mutation == "missing_incumbent":
        curves[0] = [p for p in curves[0] if p["threshold"] != .5]
    elif mutation == "duplicate":
        curves[0].append(copy.deepcopy(curves[0][0]))
    else:
        support[0] = (-1, 5)
    with pytest.raises(ValueError):
        select_recall_thresholds(curves, incumbent=.5, support=support)


def test_shared_deployment_filter_exact_parity_and_aircraft_bypass():
    thresholds = {c: (.4 if c in TARGETS else .5) for c in range(25)}
    rows = [{"category_id": c, "score": s, "bbox_xyxy": [c, 0, c+1, 1]}
            for c in range(25) for s in (.399, .4, .499, .5, .9)]
    original = copy.deepcopy(rows)
    out = apply_recall_thresholds({7: rows}, {7: 2}, {2: thresholds})[7]
    prediction = Prediction("7", [r["bbox_xyxy"] for r in rows],
                            [r["score"] for r in rows], [r["category_id"] for r in rows])
    shared = filter_prediction_by_thresholds(prediction, global_threshold=.5, fine_thresholds=thresholds)
    assert list(zip(shared.boxes_xyxy, shared.scores, shared.labels, strict=True)) == [
        (r["bbox_xyxy"], r["score"], r["category_id"]) for r in out]
    assert rows == original
    assert [r for r in out if 4 <= r["category_id"] < 24] == [r for r in rows if 4 <= r["category_id"] < 24 and r["score"] >= .5]
    assert all(r in out for r in rows if r["score"] >= .5)
