import json
from pathlib import Path

from scripts.build_macroexpert_fold_view import materialize


def _sample(root: Path, image_id: int, fold: int, labels: list[int]) -> dict:
    image = root / "images" / "train" / f"{image_id}.jpg"
    label = root / "labels" / "train" / f"{image_id}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"jpg")
    label.write_text("".join(f"{value} .5 .5 .1 .1\n" for value in labels))
    return {"image_id": image_id, "relative_path": f"images/train/{image_id}.jpg",
            "fold": fold, "group_id": f"g{fold}"}


def test_materialize_remaps_repeats_and_keeps_split_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "data"
    samples = [
        _sample(root, 1, 1, [0, 4]),
        _sample(root, 2, 1, [24]),
        _sample(root, 3, 0, [3, 23, 24]),
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"samples": samples}))
    output = tmp_path / "view"

    audit = materialize(
        manifest=manifest, data_root=root, output=output, fold=0,
        aircraft_only_keep=0.0, seed=42, repeats={0: 3, 1: 2, 2: 1, 3: 1, 4: 2},
    )

    assert audit["materialized_image_counts"] == {"train": 5, "val": 1}
    assert audit["cross_split_group_count"] == 0
    train_labels = [p.read_text() for p in sorted((output / "labels/train").iterdir())]
    assert any(text.startswith("0 ") and "5 " in text for text in train_labels)
    assert any(text.startswith("4 ") for text in train_labels)
    assert (output / "labels/val/img000003_r00.txt").read_text().splitlines()[1].startswith("5 ")
