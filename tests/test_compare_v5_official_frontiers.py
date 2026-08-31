from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "compare_v5_official_frontiers.py"
    spec = importlib.util.spec_from_file_location("compare_v5_official_frontiers", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_crossfit_selects_requested_level() -> None:
    module = _load_script()
    expected = {"recall": 0.9}
    payload = {"frontiers": {"0.150": {"crossfit": expected}}}
    assert module._crossfit(payload, "0.150") is expected
