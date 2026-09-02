from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_cv3_oof_pseudo_frontier.py"
SPEC = importlib.util.spec_from_file_location("pseudo_frontier_analysis", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_admission_payload_supports_custom_fdr_grid() -> None:
    results = {
        "0.130": {"crossfit": {"platform_gate_recall": 0.94, "platform_gate_fdr": 0.129}},
        "0.140": {"crossfit": {"platform_gate_recall": 0.95, "platform_gate_fdr": 0.139}},
        "0.145": {"crossfit": {"platform_gate_recall": 0.951, "platform_gate_fdr": 0.144}},
        "0.150": {"crossfit": {"platform_gate_recall": 0.952, "platform_gate_fdr": 0.149}},
    }

    payload = MODULE._admission_payload(results, [0.13, 0.14, 0.145, 0.15])

    assert payload["selected_level"] == pytest.approx(0.15)
    assert payload["passed"] is True
    assert payload["stretch_selected_level"] == pytest.approx(0.13)
    assert payload["stretch_passed"] is False


def test_admission_payload_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MODULE._admission_payload({}, [])


def test_absolute_score_selector_uses_constraints_and_published_objective() -> None:
    def point(threshold: float, recall: float, fdr: float) -> dict:
        return {
            "threshold": threshold,
            "platform_gate_recall": recall,
            "platform_gate_fdr": fdr,
            **{
                f"{name}_macro_recall": recall
                for name in ("ship", "aircraft", "vehicle")
            },
            **{
                f"{name}_macro_fdr": fdr
                for name in ("ship", "aircraft", "vehicle")
            },
        }

    points = [
        point(0.1, 0.95, 0.25),
        point(0.2, 0.90, 0.17),
        point(0.3, 0.88, 0.10),
    ]
    selected = MODULE._select_absolute_score_threshold(
        points,
        latency_seconds=3.0,
        minimum_recall=0.87,
        maximum_fdr=0.18,
    )
    assert selected["constraint_feasible"] is True
    assert selected["threshold"] == pytest.approx(0.3)
