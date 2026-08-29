from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_multifamily_cv3_pseudo_eval.py"
SPEC = importlib.util.spec_from_file_location("run_multifamily_cv3_pseudo_eval", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_build_pipeline_config_matches_frozen_safe_contract() -> None:
    config = MODULE.build_pipeline_config(batch_size=4, score_floor=0.03)
    assert config.tile_size == 1024
    assert config.overlap == 256
    assert config.fusion == "safe"
    assert config.score_threshold == pytest.approx(0.03)
    assert config.max_detections == 4000


@pytest.mark.parametrize("batch_size,score", [(0, 0.03), (4, -0.1), (4, 1.1)])
def test_build_pipeline_config_rejects_invalid_values(
    batch_size: int, score: float
) -> None:
    with pytest.raises(ValueError):
        MODULE.build_pipeline_config(batch_size=batch_size, score_floor=score)


def test_prediction_to_coco_converts_xyxy_and_skips_degenerate() -> None:
    prediction = SimpleNamespace(
        boxes_xyxy=[[10, 20, 40, 70], [1, 2, 1, 5]],
        scores=[0.8, 0.4],
        labels=[3, 4],
    )
    rows = MODULE.prediction_to_coco(prediction, image_id=9, source_fold=2)
    assert rows == [
        {
            "image_id": 9,
            "category_id": 3,
            "bbox": [10.0, 20.0, 30.0, 50.0],
            "score": 0.8,
            "source_fold": 2,
        }
    ]
