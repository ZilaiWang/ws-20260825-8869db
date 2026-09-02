from scripts.decide_peer_fixed_benchmarks import decide


def _metrics(recall: float, macro: float, vehicle: float, *, fdr: float = 0.15) -> dict:
    coarse_recall = {"ship": 0.8, "aircraft": 0.95, "vehicle": vehicle}
    coarse_fdr = {"ship": fdr, "aircraft": fdr, "vehicle": fdr}
    return {
        "recall": recall,
        "pooled_recall": recall,
        "pooled_fdr": fdr,
        "macro_recall": macro,
        "fine25_macro_recall": macro,
        "fdr": fdr,
        "per_coarse": {
            "ship": {"recall": 0.8},
            "aircraft": {"recall": 0.95},
            "vehicle": {"recall": vehicle},
        },
        "platform": {
            "metric_protocol": "platform_observed_20260831",
            "gate_recall": sum(coarse_recall.values()) / 3,
            "gate_fdr": sum(coarse_fdr.values()) / 3,
            "per_coarse": {
                name: {
                    "macro_recall": coarse_recall[name],
                    "macro_fdr": coarse_fdr[name],
                }
                for name in coarse_recall
            },
        },
    }


def _hard(recall: float, macro: float, vehicle: float) -> dict:
    return {
        "input_sha256": {"gt": "hard"},
        "frontiers": {"0.150": _metrics(recall, macro, vehicle)},
    }


def _sentinel(recall: float, macro: float, vehicle: float) -> dict:
    return {
        "input_sha256": {"gt": "sentinel"},
        **_metrics(recall, macro, vehicle),
    }


def test_fixed_screen_admits_robust_gain() -> None:
    result = decide(
        _hard(0.80, 0.78, 0.60),
        _hard(0.81, 0.79, 0.62),
        _sentinel(0.78, 0.76, 0.58),
        _sentinel(0.781, 0.761, 0.581),
    )
    assert result["status"] == "screen_admitted"


def test_fixed_screen_rejects_sentinel_regression() -> None:
    result = decide(
        _hard(0.80, 0.78, 0.60),
        _hard(0.81, 0.79, 0.62),
        _sentinel(0.78, 0.76, 0.58),
        _sentinel(0.77, 0.75, 0.57),
    )
    assert result["status"] == "screen_rejected"
    assert not result["gates"]["sentinel_platform_recall_nondegrade"]


def test_fixed_screen_rejects_platform_fdr_regression() -> None:
    result = decide(
        _hard(0.80, 0.78, 0.60),
        {
            "input_sha256": {"gt": "hard"},
            "frontiers": {"0.150": _metrics(0.82, 0.80, 0.62, fdr=0.16)},
        },
        _sentinel(0.78, 0.76, 0.58),
        _sentinel(0.781, 0.761, 0.581),
    )
    assert result["status"] == "screen_rejected"
    assert not result["gates"]["hard_platform_fdr_nondegrade"]


def test_fixed_screen_fails_closed_on_legacy_payload() -> None:
    legacy = _hard(0.80, 0.78, 0.60)
    del legacy["frontiers"]["0.150"]["platform"]
    try:
        decide(
            legacy,
            _hard(0.81, 0.79, 0.61),
            _sentinel(0.78, 0.76, 0.58),
            _sentinel(0.781, 0.761, 0.581),
        )
    except ValueError as exc:
        assert "platform_observed" in str(exc)
    else:
        raise AssertionError("legacy metrics must not authorize a platform gate")
