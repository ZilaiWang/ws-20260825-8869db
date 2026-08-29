from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/export_multi_detector_oer_coco.py"
SPEC = importlib.util.spec_from_file_location("export_multi_oer", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_to_coco() -> None:
    rows = MODULE.to_coco(
        [
            {
                "image_id": 2,
                "category_id": 4,
                "bbox_xyxy": [1, 3, 11, 23],
                "score": 0.9,
                "fold": 1,
                "model_key": "M3",
            }
        ]
    )
    assert rows == [
        {
            "image_id": 2,
            "category_id": 4,
            "bbox": [1.0, 3.0, 10.0, 20.0],
            "score": 0.9,
            "source_fold": 1,
            "source_model": "M3",
        }
    ]
