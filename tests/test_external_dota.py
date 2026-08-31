from __future__ import annotations

from pathlib import Path

from PIL import Image

from rsdet.external.dota import import_dota, parse_dota_label


def test_dota_import_maps_and_clips(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "P0001.png")
    (labels / "P0001.txt").write_text(
        "imagesource:GoogleEarth\n"
        "gsd:0.5\n"
        "-2 10 40 10 40 30 -2 30 plane 0\n"
        "50 50 70 50 70 70 50 70 small-vehicle 1\n",
        encoding="utf-8",
    )
    payload, audit = import_dota(images, labels)
    assert len(payload["images"]) == 1
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["category_id"] == 0
    assert payload["annotations"][0]["bbox"] == [0.0, 10.0, 40.0, 20.0]
    assert audit["dropped_difficult"] == 1


def test_dota_parser_counts_invalid(tmp_path: Path) -> None:
    label = tmp_path / "bad.txt"
    label.write_text("bad line\n0 0 1 0 1 1 0 1 ship 0\n", encoding="utf-8")
    objects, invalid = parse_dota_label(label)
    assert len(objects) == 1
    assert invalid == 1


def test_dota_import_allows_full_labels_with_image_subset(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (10, 10)).save(images / "P0001.png")
    valid = "0 0 5 0 5 5 0 5 ship 0\n"
    (labels / "P0001.txt").write_text(valid)
    (labels / "P0002.txt").write_text(valid)
    payload, audit = import_dota(images, labels, require_exact_stem_set=False)
    assert len(payload["images"]) == 1
    assert audit["labels_without_images_count"] == 1


def test_dota_scene_region_is_kept_as_background_not_giant_foreground(tmp_path: Path) -> None:
    images = tmp_path / "images"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    Image.new("RGB", (100, 80)).save(images / "P0001.png")
    (labels / "P0001.txt").write_text(
        "0 0 90 0 90 70 0 70 harbor 0\n"
        "10 10 20 10 20 20 10 20 ship 0\n"
    )
    payload, audit = import_dota(images, labels)
    assert len(payload["annotations"]) == 1
    assert payload["annotations"][0]["category_id"] == 1
    assert audit["dropped_scene_structure"] == 1
