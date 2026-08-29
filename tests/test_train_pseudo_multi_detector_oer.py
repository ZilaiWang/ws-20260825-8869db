from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/train_pseudo_multi_detector_oer.py"
SPEC = importlib.util.spec_from_file_location("train_pseudo_multi_oer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_records_filters_floor_and_preserves_fold(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "image_id": 7,
                    "category_id": 24,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.20,
                    "source_fold": 2,
                },
                {
                    "image_id": 7,
                    "category_id": 24,
                    "bbox": [1.0, 2.0, 3.0, 4.0],
                    "score": 0.01,
                    "source_fold": 2,
                },
            ]
        ),
        encoding="utf-8",
    )

    rows = MODULE._load_records(
        path, model_key="M3", score_floor=0.03, stable_offset=100
    )

    assert len(rows) == 1
    assert rows[0]["fold"] == 2
    assert rows[0]["bbox_xyxy"] == [1.0, 2.0, 4.0, 6.0]
    assert rows[0]["stable_order"] == 100


def test_load_records_rejects_unknown_fold(tmp_path: Path) -> None:
    path = tmp_path / "predictions.json"
    path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 0,
                    "bbox": [0, 0, 10, 10],
                    "score": 0.9,
                    "source_fold": 3,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="source_fold"):
        MODULE._load_records(path, model_key="Y5", score_floor=0.0, stable_offset=0)


def test_to_coco_roundtrip_box() -> None:
    rows = MODULE._to_coco(
        [
            {
                "image_id": 1,
                "category_id": 4,
                "bbox_xyxy": [2.0, 3.0, 7.0, 11.0],
                "score": 0.8,
                "fold": 0,
                "model_key": "Y5",
            }
        ]
    )

    assert rows[0]["bbox"] == [2.0, 3.0, 5.0, 8.0]
    assert rows[0]["source_fold"] == 0
