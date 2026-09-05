from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/split_sprint20_shared_cache.py"
    spec = importlib.util.spec_from_file_location("split_sprint20_shared_cache", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_preserves_alignment_and_selects_each_head() -> None:
    module = _module()
    cache = {
        "head": "shared",
        "role": "full_seen",
        "images": [
            {
                "image_id": 1,
                "pixel_sha256": "pixels",
                "prediction": {"scores": [0.7]},
                "otm_prediction": {"scores": [0.8]},
            }
        ],
    }

    pair = module.split_shared(cache)

    assert pair["oto"]["images"][0]["prediction"]["scores"] == [0.7]
    assert pair["otm"]["images"][0]["prediction"]["scores"] == [0.8]
    assert pair["oto"]["cache_source"] == "deployment_shared_forward"
    assert "otm_prediction" not in pair["otm"]["images"][0]


@pytest.mark.parametrize("cache", [{}, {"head": "shared", "images": []}])
def test_split_rejects_empty_or_wrong_cache(cache: dict) -> None:
    with pytest.raises(ValueError):
        _module().split_shared(cache)
