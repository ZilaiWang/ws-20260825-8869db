from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "merge_coarse_crop_verifier_experts.py"
SPEC = importlib.util.spec_from_file_location("merge_coarse_crop", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(category: int, value: float) -> dict[str, object]:
    row: dict[str, object] = {
        "image_id": 1,
        "category_id": category,
        "bbox": [category, 0, 1, 1],
        "source_fold": 0,
        "detector_score": 0.8,
    }
    for field in MODULE.CROP_FIELDS:
        row[field] = value
    return row


def test_route_evidence_selects_vehicle_expert() -> None:
    natural = [_row(0, 0.1), _row(4, 0.2), _row(24, 0.3)]
    balanced = [_row(0, 0.7), _row(4, 0.8), _row(24, 0.9)]
    output = MODULE.route_evidence(
        natural,
        {"natural": natural, "balanced": balanced},
        routes={"ship": "natural", "aircraft": "natural", "vehicle": "balanced"},
    )
    assert [row["score"] for row in output] == [0.1, 0.2, 0.9]
    assert [row["crop_evidence_expert"] for row in output] == [
        "natural",
        "natural",
        "balanced",
    ]
