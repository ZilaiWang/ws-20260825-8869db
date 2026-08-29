from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_pseudo_pixel_oer.py"
SPEC = importlib.util.spec_from_file_location("train_pseudo_pixel_oer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


MAPPING = {
    **{index: "ship" for index in range(4)},
    **{index: "aircraft" for index in range(4, 24)},
    24: "vehicle",
}


def _record(category: int, top1: int) -> dict[str, object]:
    return {
        "image_id": 1,
        "category_id": category,
        "bbox_xyxy": [1.0, 2.0, 11.0, 22.0],
        "score": 0.7,
        "detector_score": 0.6,
        "fold": 0,
        "model_key": "Y5",
        "stable_order": 3,
        "crop_class_probability": 0.2,
        "crop_top1": 0.8,
        "crop_margin": 0.5,
        "crop_entropy": 1.2,
        "crop_top1_class": top1,
        "detector_crop_agree": int(category == top1),
    }


def test_dual_hypothesis_adds_only_same_coarse_alternative() -> None:
    same = MODULE.expand_hypotheses(
        [_record(4, 5)], category_mapping=MAPPING, dual_hypothesis=True
    )
    cross = MODULE.expand_hypotheses(
        [_record(4, 0)], category_mapping=MAPPING, dual_hypothesis=True
    )

    assert [row["category_id"] for row in same] == [4, 5]
    assert [row["hypothesis_is_relabel"] for row in same] == [0, 1]
    assert len(cross) == 1


def test_vehicle_has_no_alternative() -> None:
    rows = MODULE.expand_hypotheses(
        [_record(24, 4)], category_mapping=MAPPING, dual_hypothesis=True
    )
    assert len(rows) == 1
    assert rows[0]["category_id"] == 24


def test_pixel_features_are_finite_and_stable_width() -> None:
    rows = MODULE.expand_hypotheses(
        [_record(4, 5)], category_mapping=MAPPING, dual_hypothesis=True
    )
    matrix = MODULE.build_pixel_features(rows, category_mapping=MAPPING)
    assert matrix.shape == (2, len(MODULE.PIXEL_COLUMNS))
    assert np.isfinite(matrix).all()
