from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.export_coarse_coco_to_yolo import export_yolo


def test_export_coarse_coco_to_yolo(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    image_root = dataset / "images" / "train"
    image_root.mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(image_root / "a.png")
    Image.new("RGB", (100, 80)).save(image_root / "b.png")
    coco = tmp_path / "data.json"
    coco.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a.png", "width": 100, "height": 80},
                    {"id": 2, "file_name": "b.png", "width": 100, "height": 80},
                ],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 0, "bbox": [10, 20, 20, 20]}
                ],
                "categories": [
                    {"id": 0, "name": "aircraft"},
                    {"id": 1, "name": "ship"},
                ],
            }
        ),
        encoding="utf-8",
    )
    audit = export_yolo(coco, dataset, "train")
    assert audit["image_count"] == 2
    assert audit["empty_image_count"] == 1
    assert (dataset / "labels/train/a.txt").read_text().strip() == (
        "0 0.20000000 0.37500000 0.20000000 0.25000000"
    )
    assert (dataset / "labels/train/b.txt").read_text() == ""
    assert "aircraft" in (dataset / "dataset.yaml").read_text()
