from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_sprint20_frozen_route.py"
    spec = importlib.util.spec_from_file_location("evaluate_sprint20_frozen_route", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_count_delta_reports_candidate_minus_baseline() -> None:
    module = _module()
    baseline = {"per_fine": {"2": {"tp": 3, "fp": 2, "fn": 1}}}
    candidate = {"per_fine": {"2": {"tp": 4, "fp": 1, "fn": 0}}}

    assert module._count_delta(baseline, candidate) == {
        "2": {"tp": 1, "fp": -1, "fn": -1}
    }
