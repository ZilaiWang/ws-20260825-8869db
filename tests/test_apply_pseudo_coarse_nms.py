from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "apply_pseudo_coarse_nms.py"
SPEC = importlib.util.spec_from_file_location("apply_pseudo_coarse_nms", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_coarse_name_contract() -> None:
    assert MODULE.coarse_name(0) == "ship"
    assert MODULE.coarse_name(4) == "aircraft"
    assert MODULE.coarse_name(24) == "vehicle"


def test_apply_policy_uses_coarse_threshold() -> None:
    rows = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [1, 1, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 24, "bbox": [0, 0, 10, 10], "score": 0.7},
        {"image_id": 1, "category_id": 24, "bbox": [3, 3, 10, 10], "score": 0.6},
    ]
    output = MODULE.apply_policy(
        rows, thresholds={"ship": 0.6, "aircraft": 0.5, "vehicle": 0.3}
    )
    assert [row["score"] for row in output] == [0.9, 0.7]
