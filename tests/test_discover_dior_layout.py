from pathlib import Path

from scripts.discover_dior_layout import discover_dior_layout


def test_discover_dior_layout_prefers_complete_trainval(tmp_path: Path) -> None:
    images = tmp_path / "DIOR" / "JPEGImages-trainval"
    xml = tmp_path / "DIOR" / "Annotations"
    splits = tmp_path / "DIOR" / "ImageSets" / "Main"
    images.mkdir(parents=True)
    xml.mkdir(parents=True)
    splits.mkdir(parents=True)
    for stem in ("1", "2", "3"):
        (images / f"{stem}.jpg").write_bytes(b"image")
        (xml / f"{stem}.xml").write_text("<annotation/>", encoding="utf-8")
    (splits / "train.txt").write_text("1\n", encoding="utf-8")
    (splits / "trainval.txt").write_text("1\n2\n3\n", encoding="utf-8")
    result = discover_dior_layout(tmp_path)
    assert result["split_name"] == "trainval.txt"
    assert result["selected_image_count"] == 3
    assert Path(result["image_root"]) == images.resolve()
    assert Path(result["annotation_root"]) == xml.resolve()
