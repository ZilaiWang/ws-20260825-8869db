from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.build_native_tiling_probe import build_probe


def test_build_probe_selects_only_images_changed_by_tiling(tmp_path: Path) -> None:
    root = tmp_path / "images" / "train"
    root.mkdir(parents=True)
    for name, size in (("small.jpg", (800, 800)), ("probe.jpg", (1100, 900)), ("huge.jpg", (1400, 900))):
        Image.new("RGB", size).save(root / name)
    document = {
        "images": [
            {"id": 1, "file_name": "images/train/small.jpg", "width": 800, "height": 800},
            {"id": 2, "file_name": "images/train/probe.jpg", "width": 1100, "height": 900},
            {"id": 3, "file_name": "images/train/huge.jpg", "width": 1400, "height": 900},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 0, "bbox": [1, 2, 3, 4]},
            {"id": 2, "image_id": 2, "category_id": 5, "bbox": [1, 2, 3, 4]},
            {"id": 3, "image_id": 3, "category_id": 24, "bbox": [1, 2, 3, 4]},
        ],
        "categories": [{"id": index, "name": str(index)} for index in range(25)],
    }
    output, audit = build_probe(
        json.loads(json.dumps(document)),
        image_root=tmp_path,
        split_tile_size=1024,
        whole_tile_size=1280,
    )
    assert [row["id"] for row in output["images"]] == [2]
    assert [row["id"] for row in output["annotations"]] == [2]
    assert audit["images"] == 1
    assert audit["annotations_by_coarse"] == {"aircraft": 1}
