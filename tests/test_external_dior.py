from __future__ import annotations

from pathlib import Path

from PIL import Image

from rsdet.external.dior import import_dior


def _xml(objects: str) -> str:
    return f"<annotation><size><width>100</width><height>80</height></size>{objects}</annotation>"


def _object(name: str, xmin: int, ymin: int, xmax: int, ymax: int) -> str:
    return (
        f"<object><name>{name}</name><difficult>0</difficult><bndbox>"
        f"<xmin>{xmin}</xmin><ymin>{ymin}</ymin><xmax>{xmax}</xmax>"
        f"<ymax>{ymax}</ymax></bndbox></object>"
    )


def test_dior_import_maps_compact_objects_and_keeps_scene_pixels(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    annotation_root = tmp_path / "xml"
    image_root.mkdir()
    annotation_root.mkdir()
    Image.new("RGB", (100, 80)).save(image_root / "00001.jpg")
    (annotation_root / "00001.xml").write_text(
        _xml(
            _object("airplane", 1, 1, 20, 10)
            + _object("vehicle", 30, 20, 40, 30)
            + _object("harbor", 1, 1, 100, 80)
        )
    )
    payload, audit = import_dior(image_root, annotation_root)
    assert len(payload["images"]) == 1
    assert [row["category_id"] for row in payload["annotations"]] == [0, 2]
    assert payload["annotations"][0]["bbox"] == [0.0, 0.0, 20.0, 10.0]
    assert audit["dropped_scene_structure"] == 1
    assert audit["coarse_category_counts"] == {"aircraft": 1, "vehicle": 1}


def test_dior_split_is_exact_and_deterministic(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    annotation_root = tmp_path / "xml"
    image_root.mkdir()
    annotation_root.mkdir()
    for stem in ("a", "b"):
        Image.new("RGB", (16, 16)).save(image_root / f"{stem}.jpg")
        (annotation_root / f"{stem}.xml").write_text(
            _xml(_object("ship", 1, 1, 8, 8))
        )
    split = tmp_path / "split.txt"
    split.write_text("b\n")
    payload, audit = import_dior(image_root, annotation_root, split_file=split)
    assert [row["file_name"] for row in payload["images"]] == ["b.jpg"]
    assert audit["annotation_count"] == 1
