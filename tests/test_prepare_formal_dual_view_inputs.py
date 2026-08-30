from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_formal_dual_view_inputs.py"
    spec = importlib.util.spec_from_file_location("prepare_formal_dual_view_inputs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _train_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_dual_view_metric_verifier.py"
    spec = importlib.util.spec_from_file_location("train_dual_view_metric_verifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_registry_rejects_inconsistent_rows(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "manifest.csv"
    fields = [
        "formal_image_id",
        "fold",
        "group_id",
        "source_relative_path",
        "source_width",
        "source_height",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            dict(
                formal_image_id=1,
                fold=0,
                group_id="g0",
                source_relative_path="images/train/a.jpg",
                source_width=100,
                source_height=80,
            )
        )
        writer.writerow(
            dict(
                formal_image_id=1,
                fold=1,
                group_id="g1",
                source_relative_path="images/train/a.jpg",
                source_width=100,
                source_height=80,
            )
        )
    try:
        module._image_registry(path)
    except ValueError as error:
        assert "inconsistent image metadata" in str(error)
    else:
        raise AssertionError("inconsistent formal image rows must be rejected")


def test_normalize_candidates_freezes_fold_and_xywh() -> None:
    module = _module()
    registry = {
        7: {
            "id": 7,
            "fold": 2,
            "group_id": "g7",
            "file_name": "images/train/7.jpg",
            "width": 100,
            "height": 80,
        }
    }
    rows = module._normalize_candidates(
        [
            {
                "proposal_uid": "p0",
                "image_id": 7,
                "category_id": 3,
                "bbox_xyxy": [1.5, 2.5, 11.5, 22.5],
                "score": 0.75,
            }
        ],
        registry=registry,
        label="evidence",
    )
    assert rows == [
        {
            "proposal_uid": "p0",
            "proposal_index": 0,
            "image_id": 7,
            "category_id": 3,
            "bbox": [1.5, 2.5, 10.0, 20.0],
            "score": 0.75,
            "source_fold": 2,
            "source_model": "evidence",
        }
    ]


def test_precache_views_matches_direct_renderer() -> None:
    module = _train_module()
    image = Image.fromarray(
        np.arange(32 * 32 * 3, dtype=np.uint16).reshape(32, 32, 3).astype(np.uint8)
    )
    records = [{"image_id": 1, "bbox": [5.0, 6.0, 10.0, 8.0]}]
    cache = module._precache_views(
        records=records,
        images={1: image},
        resolution=16,
        context_ratio=1.75,
        workers=1,
    )
    expected = module.render_seven_channel_view(
        image,
        module._xywh_to_xyxy(records[0]["bbox"]),
        resolution=16,
        context_ratio=1.75,
    )
    assert cache.dtype == np.uint8
    assert cache.shape == (1, 7, 16, 16)
    assert np.array_equal(cache[0], expected)
