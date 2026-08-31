from __future__ import annotations

from pathlib import Path

from PIL import Image

from rsdet.external.slicing import slice_coco


def test_slice_is_scale_preserving_and_annotation_is_unique(tmp_path: Path) -> None:
    image_root = tmp_path / "source"
    image_root.mkdir()
    Image.new("RGB", (180, 100)).save(image_root / "a.png")
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 180, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [70, 30, 30, 20]}
        ],
        "categories": [{"id": 2, "name": "vehicle"}],
    }
    output, audit = slice_coco(
        payload,
        image_root,
        tmp_path / "tiles",
        tile_size=100,
        overlap=20,
        min_visibility=0.7,
        empty_tiles_per_image=1,
    )
    assert len(output["annotations"]) == 1
    assert audit["output_annotation_count"] == 1
    assert audit["resize_policy"] == "none"
    for row in output["images"]:
        with Image.open(tmp_path / "tiles" / row["file_name"]) as image:
            assert image.size == (row["width"], row["height"])


def test_slice_empty_sampling_is_deterministic(tmp_path: Path) -> None:
    image_root = tmp_path / "source"
    image_root.mkdir()
    Image.new("RGB", (180, 180)).save(image_root / "a.png")
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 180, "height": 180}],
        "annotations": [],
        "categories": [{"id": 2, "name": "vehicle"}],
    }
    first, _ = slice_coco(
        payload, image_root, tmp_path / "one", tile_size=100, overlap=20, empty_tiles_per_image=2
    )
    second, _ = slice_coco(
        payload, image_root, tmp_path / "two", tile_size=100, overlap=20, empty_tiles_per_image=2
    )
    assert [row["file_name"] for row in first["images"]] == [
        row["file_name"] for row in second["images"]
    ]
