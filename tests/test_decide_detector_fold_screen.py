from copy import deepcopy

import pytest

from scripts.decide_detector_fold_screen import decide


def _frontier() -> dict:
    per_floor = {
        "ship": {"macro_recall": 0.90},
        "aircraft": {"macro_recall": 0.97},
        "vehicle": {"macro_recall": 0.80},
    }
    per_fdr = {
        "ship": {"macro_recall": 0.80},
        "aircraft": {"macro_recall": 0.90},
        "vehicle": {"macro_recall": 0.60},
    }
    return {
        "status": "complete_diagnostic_only",
        "input_sha256": {"gt": "abc", "pred": "pred"},
        "score_floor_metrics": {
            "platform": {
                "metric_protocol": "platform_observed_20260831",
                "per_coarse": per_floor,
                "gate_recall": 0.89,
                "gate_fdr": 0.2,
            }
        },
        "frontiers": {
            "0.150": {
                "platform": {
                    "metric_protocol": "platform_observed_20260831",
                    "gate_recall": sum(x["macro_recall"] for x in per_fdr.values()) / 3,
                    "gate_fdr": 0.149,
                    "per_coarse": per_fdr,
                }
            }
        },
    }


def test_admits_only_when_all_frozen_gates_pass() -> None:
    baseline = _frontier()
    candidate = deepcopy(baseline)
    candidate["score_floor_metrics"]["platform"]["per_coarse"]["vehicle"]["macro_recall"] += 0.02
    candidate["frontiers"]["0.150"]["platform"]["gate_recall"] += 0.01
    for name, gain in {"ship": 0.01, "aircraft": 0.001, "vehicle": 0.02}.items():
        candidate["frontiers"]["0.150"]["platform"]["per_coarse"][name][
            "macro_recall"
        ] += gain
    candidate["frontiers"]["0.150"]["platform"]["gate_fdr"] -= 0.001

    result = decide(baseline, candidate)
    assert result["status"] == "screen_admitted_for_cv3"
    assert all(result["gates"].values())


def test_rejects_coarse_collapse_despite_platform_and_floor_gains() -> None:
    baseline = _frontier()
    candidate = deepcopy(baseline)
    candidate["score_floor_metrics"]["platform"]["per_coarse"]["ship"]["macro_recall"] += 0.02
    candidate["frontiers"]["0.150"]["platform"]["gate_recall"] += 0.01
    candidate["frontiers"]["0.150"]["platform"]["per_coarse"]["vehicle"][
        "macro_recall"
    ] -= 0.006

    result = decide(baseline, candidate)
    assert result["status"] == "screen_rejected"
    assert result["gates"]["coarse_macro_recall_protection"] is False


def test_rejects_mismatched_ground_truth() -> None:
    baseline = _frontier()
    candidate = deepcopy(baseline)
    candidate["input_sha256"]["gt"] = "different"
    with pytest.raises(ValueError, match="GT SHA256"):
        decide(baseline, candidate)
