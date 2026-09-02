from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from rsdet.analysis.oof_detection import FormalGroundTruth, GroundTruthObject

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_single_split_paired_scale.py"
SPEC = importlib.util.spec_from_file_location("paired_scale", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_size_bins_are_left_closed() -> None:
    assert MODULE._size_bin((0.0, 0.0, 47.0, 47.0)) == "lt48"
    assert MODULE._size_bin((0.0, 0.0, 48.0, 48.0)) == "48to80"
    assert MODULE._size_bin((0.0, 0.0, 80.0, 80.0)) == "80to128"
    assert MODULE._size_bin((0.0, 0.0, 128.0, 128.0)) == "ge128"


def test_paired_recovery_counts_object_transitions() -> None:
    objects = {
        (1, 0): GroundTruthObject(
            "a", 1, 0, 0, "g", 24, "vehicle", (0.0, 0.0, 32.0, 32.0)
        ),
        (1, 1): GroundTruthObject(
            "b", 1, 1, 0, "g", 3, "ship", (0.0, 0.0, 64.0, 64.0)
        ),
        (2, 0): GroundTruthObject(
            "c", 2, 0, 0, "h", 9, "aircraft", (0.0, 0.0, 96.0, 96.0)
        ),
    }
    formal = FormalGroundTruth({}, objects, frozenset({1, 2}), 3)
    baseline = SimpleNamespace(
        matches=(
            SimpleNamespace(image_id=1, ground_truth_index=1),
            SimpleNamespace(image_id=2, ground_truth_index=0),
        )
    )
    candidate = SimpleNamespace(
        matches=(
            SimpleNamespace(image_id=1, ground_truth_index=0),
            SimpleNamespace(image_id=2, ground_truth_index=0),
        )
    )

    result = MODULE._paired_recovery(formal, baseline, candidate)

    assert result["by_size"]["lt48"]["candidate_only"] == 1
    assert result["by_size"]["48to80"]["baseline_only"] == 1
    assert result["by_size"]["80to128"]["both"] == 1
    assert result["by_coarse"]["vehicle"]["delta_recall_pp"] == 100.0
    assert result["by_coarse"]["ship"]["delta_recall_pp"] == -100.0
