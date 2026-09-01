from __future__ import annotations

from pathlib import Path

from PIL import Image

from rsdet.external.slicing import slice_coco


def test_slice_is_scale_preserving_and_rejects_low_visibility_copy(tmp_path: Path) -> None:
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


def test_slice_labels_every_sufficiently_visible_overlap_copy(tmp_path: Path) -> None:
    image_root = tmp_path / "source"
    image_root.mkdir()
    Image.new("RGB", (180, 100)).save(image_root / "a.png")
    payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 180, "height": 100}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 2, "bbox": [82, 30, 10, 20]}
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
        empty_tiles_per_image=0,
    )
    assert len(output["images"]) == 2
    assert len(output["annotations"]) == 2
    assert {row["source_annotation_id"] for row in output["annotations"]} == {1}
    assert audit["duplicated_source_annotation_count"] == 1
    assert audit["maximum_tile_instances_per_source_annotation"] == 2
    assert audit["dropped_visibility_count"] == 0


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


def test_parallel_slice_matches_serial_output_and_pixels(tmp_path: Path) -> None:
    image_root = tmp_path / "source"
    image_root.mkdir()
    payload = {"images": [], "annotations": [], "categories": [{"id": 2, "name": "vehicle"}]}
    for image_id, color in ((2, "red"), (1, "blue"), (3, "green")):
        name = f"{image_id}.png"
        Image.new("RGB", (180, 180), color=color).save(image_root / name)
        payload["images"].append(
            {"id": image_id, "file_name": name, "width": 180, "height": 180}
        )
        payload["annotations"].append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [70, 70, 30, 30],
            }
        )
    serial, serial_audit = slice_coco(
        payload, image_root, tmp_path / "serial", tile_size=100, overlap=20, workers=1
    )
    parallel, parallel_audit = slice_coco(
        payload, image_root, tmp_path / "parallel", tile_size=100, overlap=20, workers=3
    )
    assert serial == parallel
    assert {**serial_audit, "workers": 3} == parallel_audit
    for row in serial["images"]:
        with Image.open(tmp_path / "serial" / row["file_name"]) as first:
            with Image.open(tmp_path / "parallel" / row["file_name"]) as second:
                assert first.tobytes() == second.tobytes()
