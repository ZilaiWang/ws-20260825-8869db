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
        "0.130": {"crossfit": {"recall": 0.94, "fdr": 0.129}},
        "0.140": {"crossfit": {"recall": 0.95, "fdr": 0.139}},
        "0.145": {"crossfit": {"recall": 0.951, "fdr": 0.144}},
        "0.150": {"crossfit": {"recall": 0.952, "fdr": 0.149}},
    }

    payload = MODULE._admission_payload(results, [0.13, 0.14, 0.145, 0.15])

    assert payload["selected_level"] == pytest.approx(0.15)
    assert payload["passed"] is True
    assert payload["stretch_selected_level"] == pytest.approx(0.13)
    assert payload["stretch_passed"] is False


def test_admission_payload_rejects_empty_grid() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MODULE._admission_payload({}, [])
