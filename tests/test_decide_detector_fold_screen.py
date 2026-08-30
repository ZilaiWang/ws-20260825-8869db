from copy import deepcopy

import pytest

from scripts.decide_detector_fold_screen import decide


def _frontier() -> dict:
    per_floor = {
        "ship": {"recall": 0.90},
        "aircraft": {"recall": 0.97},
        "vehicle": {"recall": 0.80},
    }
    per_fdr = {
        "ship": {"recall": 0.80},
        "aircraft": {"recall": 0.90},
        "vehicle": {"recall": 0.60},
    }
    return {
        "status": "complete_diagnostic_only",
        "input_sha256": {"gt": "abc", "pred": "pred"},
        "score_floor_metrics": {"per_coarse": per_floor},
        "frontiers": {
            "0.150": {"recall": 0.85, "fdr": 0.149, "per_coarse": per_fdr}
        },
    }


def test_admits_only_when_all_frozen_gates_pass() -> None:
    baseline = _frontier()
    candidate = deepcopy(baseline)
    candidate["score_floor_metrics"]["per_coarse"]["vehicle"]["recall"] += 0.02
    candidate["frontiers"]["0.150"]["recall"] += 0.01
    candidate["frontiers"]["0.150"]["per_coarse"]["ship"]["recall"] += 0.01
    candidate["frontiers"]["0.150"]["per_coarse"]["aircraft"]["recall"] += 0.001
    candidate["frontiers"]["0.150"]["per_coarse"]["vehicle"]["recall"] += 0.02

    result = decide(baseline, candidate)
    assert result["status"] == "screen_admitted_for_cv3"
    assert all(result["gates"].values())


def test_rejects_coarse_collapse_despite_pooled_and_floor_gains() -> None:
    baseline = _frontier()
    candidate = deepcopy(baseline)
    candidate["score_floor_metrics"]["per_coarse"]["ship"]["recall"] += 0.02
    candidate["frontiers"]["0.150"]["recall"] += 0.01
    candidate["frontiers"]["0.150"]["per_coarse"]["vehicle"]["recall"] -= 0.006

    result = decide(baseline, candidate)
    assert result["status"] == "screen_rejected"
    assert result["gates"]["coarse_recall_protection"] is False


def test_rejects_mismatched_ground_truth() -> None:
    baseline = _frontier()
    candidate = deepcopy(baseline)
    candidate["input_sha256"]["gt"] = "different"
    with pytest.raises(ValueError, match="GT SHA256"):
        decide(baseline, candidate)
