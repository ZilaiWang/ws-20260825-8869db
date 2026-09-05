from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/compare_sprint20_runtime.py"
    spec = importlib.util.spec_from_file_location("compare_sprint20_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_canonical_filters_declared_labels() -> None:
    rows = {
        1: [
            {"category_id": 2, "score": 0.7, "bbox_xyxy": [0, 0, 1, 1]},
            {"category_id": 4, "score": 0.8, "bbox_xyxy": [2, 2, 3, 3]},
        ]
    }

    assert _module()._canonical(rows, {4}) == [(1, 4, 0.8, (2.0, 2.0, 3.0, 3.0))]
