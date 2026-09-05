from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_sprint20_head_probe.py"
    spec = importlib.util.spec_from_file_location("analyze_sprint20_head_probe", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _point(threshold, recall, fdr, fp, quality):
    return {
        "threshold": threshold,
        "platform_gate_recall": recall,
        "platform_gate_fdr": fdr,
        "platform_quality_score": quality,
        "fp": fp,
    }


def test_frontier_selection_obeys_constraints_before_quality():
    module = _load_script()
    curve = [
        _point(0.1, 0.95, 0.20, 30, 90),
        _point(0.2, 0.92, 0.12, 20, 80),
        _point(0.3, 0.90, 0.10, 10, 85),
    ]
    assert module._best_under_fdr(curve, 0.12)["threshold"] == pytest.approx(0.2)
    assert module._best_under_fp(curve, 15)["threshold"] == pytest.approx(0.3)


def test_fine_delta_retains_tp_fp_fn_signs():
    module = _load_script()
    baseline = {"24": {"tp": 8, "fp": 3, "fn": 2, "recall": 0.8, "fdr": 3 / 11}}
    candidate = {"24": {"tp": 9, "fp": 5, "fn": 1, "recall": 0.9, "fdr": 5 / 14}}
    delta = module._fine_delta(baseline, candidate)["24"]
    assert delta["delta_tp"] == 1
    assert delta["delta_fp"] == 2
    assert delta["delta_fn"] == -1
    assert delta["delta_recall"] == pytest.approx(0.1)
