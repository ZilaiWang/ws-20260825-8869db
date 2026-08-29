from __future__ import annotations

import json
from pathlib import Path

from scripts.train_y5_coarse_detector import coarse_id, materialize_coarse_dataset


def test_coarse_id_contract() -> None:
    assert [coarse_id(value) for value in (0, 3, 4, 23, 24)] == [0, 0, 1, 1, 2]


def test_materialize_coarse_dataset_keeps_symlink_label_space(tmp_path: Path) -> None:
    data = tmp_path / "data"
    image = data / "images" / "train" / "sample.jpg"
    label = data / "labels" / "train" / "sample.txt"
    image.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    image.write_bytes(b"image")
    label.write_text("3 0.5 0.5 0.2 0.2\n17 0.4 0.4 0.1 0.1\n24 0.2 0.2 0.1 0.1\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"image_id": 7, "relative_path": "images/train/sample.jpg", "fold": 1}
                ]
            }
        )
    )

    dataset, audit = materialize_coarse_dataset(manifest, data, 0, tmp_path / "out")

    train_path = Path((tmp_path / "out" / "train_images.txt").read_text().strip())
    assert train_path.is_symlink()
    assert "out/images/train" in str(train_path)
    assert (tmp_path / "out" / "labels" / "train" / "image_00007.txt").read_text() == (
        "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n2 0.2 0.2 0.1 0.1\n"
    )
    assert audit["class_counts"] == {"ship": 1, "aircraft": 1, "vehicle": 1}
    assert dataset.is_file()
