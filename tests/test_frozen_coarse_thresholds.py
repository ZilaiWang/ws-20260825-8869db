from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / (
        "evaluate_pseudo_with_frozen_coarse_thresholds.py"
    )
    spec = importlib.util.spec_from_file_location("frozen_coarse", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_thresholds_parse_nested_per_coarse_contract() -> None:
    module = _load_script()
    frontier = {
        "frontiers": {
            "0.150": {
                "crossfit_thresholds": {
                    str(fold): {
                        coarse: {"threshold": value}
                        for coarse, value in {
                            "ship": 0.1 + fold,
                            "aircraft": 0.2 + fold,
                            "vehicle": 0.3 + fold,
                        }.items()
                    }
                    for fold in range(3)
                }
            }
        }
    }
    parsed = module._thresholds(frontier, 0.15)
    assert parsed[2] == {"ship": 2.1, "aircraft": 2.2, "vehicle": 2.3}


def test_thresholds_reject_global_frontier_schema() -> None:
    module = _load_script()
    frontier = {
        "frontiers": {
            "0.150": {"crossfit_thresholds": {"0": 0.1, "1": 0.2, "2": 0.3}}
        }
    }
    with pytest.raises(ValueError, match="per-coarse"):
        module._thresholds(frontier, 0.15)
