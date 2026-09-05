from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts/analyze_sprint20_oof_routing.py"
    spec = importlib.util.spec_from_file_location("analyze_sprint20_oof_routing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(label: int, score: float, x: float) -> dict:
    return {
        "category_id": label,
        "score": score,
        "bbox_xyxy": [x, 0.0, x + 1.0, 1.0],
    }


def test_deployment_like_route_only_replaces_ship23():
    module = _load_script()
    oto = {
        1: [
            _row(0, 0.54, 0),
            _row(2, 0.99, 2),
            _row(4, 0.80, 4),
            _row(24, 0.70, 24),
        ]
    }
    otm = {1: [_row(2, 0.55, 20), _row(3, 0.53, 30), _row(24, 0.99, 40)]}
    routed = module._route_fixed_primary_ship23(
        oto,
        otm,
        {1},
        primary_threshold=0.536,
        alternative_threshold=0.546,
    )[1]
    assert [(row["category_id"], row["score"]) for row in routed] == [
        (0, 0.54),
        (4, 0.80),
        (24, 0.70),
        (2, 0.55),
    ]
    assert all(row["bbox_xyxy"][0] != 2 for row in routed)


def test_fixed_primary_filter_includes_boundary():
    module = _load_script()
    pred = {1: [_row(2, 0.536, 1), _row(3, 0.5359, 2)]}
    assert module._filter_fixed_threshold(pred, {1}, 0.536)[1] == [pred[1][0]]


def test_macro_fdr_is_derived_from_serialized_counts():
    module = _load_script()
    metrics = {
        "per_fine": {
            "0": {"tp": 3, "fp": 1, "fn": 0},
            "1": {"tp": 0, "fp": 0, "fn": 2},
        }
    }
    assert module._macro_fdr(metrics, (0, 1)) == 0.125
