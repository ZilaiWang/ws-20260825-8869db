from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.submission.competition import (
    _SubmissionYoloDetector,
    discover_images,
    load_submission_config,
    run_submission,
    validate_result_payload,
)


def _config(path: Path) -> Path:
    payload = {
        "device": "cuda:0",
        "model": {
            "family": "yolo",
            "weight_path": "/app/models/model.pt",
            "expected_sha256": "a" * 64,
            "imgsz": 1024,
            "confidence": 0.001,
            "iou": 0.7,
            "max_detections": 500,
        },
        "pipeline": {
            "tile_size": 1280,
            "overlap": 256,
            "batch_size": 8,
            "fusion": "global",
            "score_threshold": 0.051,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_submission_config_rejects_relative_model_path(tmp_path: Path) -> None:
    path = _config(tmp_path / "config.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["weight_path"] = "models/model.pt"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="绝对路径"):
        load_submission_config(path)


def test_load_submission_config_validates_rot90_views(tmp_path: Path) -> None:
    path = _config(tmp_path / "config.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["model"]["rot90_views"] = [0, 1, 2, 3]
    payload["model"]["tta_nms_iou"] = 0.55
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_submission_config(path)["model"]["rot90_views"] == [0, 1, 2, 3]

    payload["model"]["rot90_views"] = [1, 2, 3]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="恒等视图"):
        load_submission_config(path)


@pytest.mark.parametrize(
    ("rotation", "rotated_box"),
    [
        (0, [10.0, 20.0, 40.0, 60.0]),
        (1, [20.0, 60.0, 60.0, 90.0]),
        (2, [60.0, 20.0, 90.0, 60.0]),
        (3, [20.0, 10.0, 60.0, 40.0]),
    ],
)
def test_invert_rot90_box(rotation: int, rotated_box: list[float]) -> None:
    # Source canvas is width=100, height=80; source box is fixed below.
    restored = _SubmissionYoloDetector._invert_rot90_box(
        rotated_box, rotation, width=100, height=80
    )
    assert restored == pytest.approx([10.0, 20.0, 40.0, 60.0])


def test_discover_images_first_level_and_sorted(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    Image.new("RGB", (8, 6), "white").save(tmp_path / "b.PNG")
    Image.new("RGB", (8, 6), "black").save(tmp_path / "a.jpg")
    Image.new("RGB", (8, 6), "red").save(tmp_path / "nested" / "c.jpg")
    (tmp_path / "note.txt").write_text("ignored", encoding="utf-8")
    assert [path.name for path in discover_images(tmp_path)] == ["a.jpg", "b.PNG"]


def test_discover_images_rejects_duplicate_stems(tmp_path: Path) -> None:
    Image.new("RGB", (8, 6), "white").save(tmp_path / "same.jpg")
    Image.new("RGB", (8, 6), "black").save(tmp_path / "same.png")
    with pytest.raises(ValueError, match="主名重复"):
        discover_images(tmp_path)


def test_run_submission_writes_official_payload(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (20, 10), "white").save(input_dir / "000001.bmp")

    class FakeDetector:
        def __init__(self, config):
            assert config["model"]["family"] == "yolo"

        def predict(self, image):
            assert image.mode == "RGB"
            return [
                {
                    "category_id": 24,
                    "category_name": FINE_NAMES[24],
                    "score": 0.75,
                    "bbox": [1.0, 2.0, 9.0, 8.0],
                }
            ]

    payload = run_submission(
        input_dir,
        output_dir,
        _config(tmp_path / "config.json"),
        detector_factory=FakeDetector,
    )
    persisted = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert persisted == payload
    assert payload["status"] == "success"
    assert payload["images"][0]["image_id"] == "000001"
    assert payload["images"][0]["file_name"] == "000001.bmp"
    assert payload["images"][0]["width"] == 20
    assert payload["images"][0]["height"] == 10
    assert payload["images"][0]["run_end_timestamp"] > 0
    assert validate_result_payload(payload) == {"images": 1, "objects": 1}


def test_validate_result_payload_rejects_out_of_bounds_bbox() -> None:
    payload = {
        "status": "success",
        "images": [
            {
                "image_id": "x",
                "file_name": "x.png",
                "width": 10,
                "height": 10,
                "run_end_timestamp": 1,
                "objects": [
                    {
                        "category_id": 0,
                        "category_name": FINE_NAMES[0],
                        "score": 0.5,
                        "bbox": [0.0, 0.0, 11.0, 5.0],
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValueError, match="超出原图边界"):
        validate_result_payload(payload)
