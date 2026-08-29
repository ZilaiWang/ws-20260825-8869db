import json
from pathlib import Path

import pytest

from scripts.train_y5_background_complete import materialize_dataset


def _write_image_label(root: Path, stem: str) -> Path:
    image = root / "images" / "train" / f"{stem}.jpg"
    label = root / "labels" / "train" / f"{stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image")
    label.write_text("", encoding="utf-8")
    return image


def test_materialize_dataset_rejects_heldout_hard_negative(tmp_path: Path) -> None:
    data = tmp_path / "data"
    official = _write_image_label(data, "official")
    hard_root = tmp_path / "hard"
    hard = _write_image_label(hard_root, "hard")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"samples": [{"relative_path": str(official.relative_to(data)), "fold": 1}]}
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {"held_out_fold": 0, "source_folds": [0, 1], "tiles": [{"image": str(hard)}]}
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="held-out fold"):
        materialize_dataset(
            manifest=manifest,
            data_root=data,
            hard_negative_summary=summary,
            held_out_fold=0,
            output_dir=tmp_path / "out",
        )
