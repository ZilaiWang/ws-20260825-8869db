from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _module():
    path = Path(__file__).resolve().parents[1] / "scripts/aggregate_sprint20_head_caches.py"
    spec = importlib.util.spec_from_file_location("aggregate_sprint20_head_caches", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_shared_fold_is_split_without_leaking_the_other_head(tmp_path: Path) -> None:
    module = _module()
    fold = tmp_path / "fold_0"
    fold.mkdir()
    payload = {
        "head": "shared",
        "role": "outer_oof_short",
        "images": [
            {
                "image_id": 7,
                "file_name": "seven.jpg",
                "pixel_sha256": "pixels",
                "prediction": {"image_id": 7, "scores": [0.7]},
                "otm_prediction": {"image_id": 7, "scores": [0.8]},
                "fold": 0,
            }
        ],
    }
    (fold / "shared.json").write_text(json.dumps(payload), encoding="utf-8")

    pair = module._load_fold_pair(tmp_path, 0, "shared")

    assert pair["oto"]["head"] == "oto"
    assert pair["otm"]["head"] == "otm"
    assert pair["oto"]["images"][0]["prediction"]["scores"] == [0.7]
    assert pair["otm"]["images"][0]["prediction"]["scores"] == [0.8]
    assert "otm_prediction" not in pair["oto"]["images"][0]
    assert "otm_prediction" not in pair["otm"]["images"][0]


def test_shared_fold_requires_deployment_identity(tmp_path: Path) -> None:
    module = _module()
    fold = tmp_path / "fold_1"
    fold.mkdir()
    (fold / "shared.json").write_text(
        json.dumps({"head": "native", "role": "outer_oof_short", "images": []}),
        encoding="utf-8",
    )

    try:
        module._load_fold_pair(tmp_path, 1, "shared")
    except ValueError as exc:
        assert "Invalid shared fold 1 identity" in str(exc)
    else:
        raise AssertionError("invalid shared cache was accepted")
