from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/rerank_cv3_pseudo_with_multi_detector_oer.py"
)
SPEC = importlib.util.spec_from_file_location("rerank_multi_oer", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_pseudo_records_filters_and_normalizes() -> None:
    rows = [
        {
            "image_id": 3,
            "category_id": 7,
            "bbox": [10, 20, 30, 40],
            "score": 0.4,
            "source_fold": 2,
        },
        {
            "image_id": 3,
            "category_id": 8,
            "bbox": [1, 2, 3, 4],
            "score": 0.01,
            "source_fold": 2,
        },
    ]
    records = MODULE.load_pseudo_records(
        rows, model_key="M3", score_floor=0.03, stable_offset=100
    )
    assert records == [
        {
            "image_id": 3,
            "fold": 2,
            "category_id": 7,
            "bbox_xyxy": [10.0, 20.0, 40.0, 60.0],
            "score": 0.4,
            "detector_score": 0.4,
            "model_key": "M3",
            "stable_order": 100,
        }
    ]


def test_to_coco_rows_preserves_fold_and_model() -> None:
    rows = MODULE.to_coco_rows(
        [
            {
                "image_id": 4,
                "fold": 1,
                "category_id": 2,
                "bbox_xyxy": [5, 6, 15, 26],
                "score": 0.75,
                "model_key": "Y5",
            }
        ]
    )
    assert rows[0]["bbox"] == [5.0, 6.0, 10.0, 20.0]
    assert rows[0]["source_fold"] == 1
    assert rows[0]["source_model"] == "Y5"
