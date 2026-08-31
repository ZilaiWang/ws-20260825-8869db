from __future__ import annotations

from pathlib import Path

from scripts.build_yolo_smoke_dataset import build_smoke_list


def test_smoke_list_covers_classes_before_filling(tmp_path: Path) -> None:
    lines = []
    for index, label in enumerate((0, 0, 1, 2)):
        image = tmp_path / "images" / "train" / f"{index}.jpg"
        target = tmp_path / "labels" / "train" / f"{index}.txt"
        image.parent.mkdir(parents=True, exist_ok=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"image")
        target.write_text(f"{label} 0.5 0.5 0.1 0.1\n")
        lines.append(str(image))
    selected, audit = build_smoke_list(lines, 3)
    assert [Path(value).stem for value in selected] == ["0", "2", "3"]
    assert audit["covered_class_ids"] == [0, 1, 2]
