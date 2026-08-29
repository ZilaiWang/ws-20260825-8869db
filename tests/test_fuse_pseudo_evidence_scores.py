from __future__ import annotations

import importlib.util
import math
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fuse_pseudo_evidence_scores.py"
SPEC = importlib.util.spec_from_file_location("fuse_pseudo_evidence_scores", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_geometric_score_equal_weights() -> None:
    value = MODULE.geometric_score({"a": 0.25, "b": 1.0}, ["a", "b"], [1.0, 1.0])
    assert math.isclose(value, 0.5)


def test_geometric_score_normalizes_weights() -> None:
    first = MODULE.geometric_score({"a": 0.2, "b": 0.8}, ["a", "b"], [1.0, 3.0])
    second = MODULE.geometric_score({"a": 0.2, "b": 0.8}, ["a", "b"], [0.25, 0.75])
    assert math.isclose(first, second)
