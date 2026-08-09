from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    check=False,
    capture_output=True,
    text=True,
)
if torch_probe.returncode != 0:  # pragma: no cover - environment dependent
    reason = (torch_probe.stderr or torch_probe.stdout).strip().splitlines()
    pytest.skip(
        f"PyTorch is unavailable: {reason[-1] if reason else 'import failed'}",
        allow_module_level=True,
    )
import torch  # noqa: E402

from rsdet.data.bhcdetr_dataset import (  # noqa: E402
    BHCDETRDataset,
    BHCDetrDataset,
    bhcdetr_collate,
)


def _write_sample(
    root: Path,
    *,
    split: str,
    name: str,
    size: tuple[int, int] = (100, 50),
    color: tuple[int, int, int] = (255, 0, 0),
    label: str = "3 0.5 0.5 0.4 0.4\n",
) -> str:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    relative = f"images/{split}/{name}.png"
    Image.new("RGB", size, color=color).save(root / relative)
    (label_dir / f"{name}.txt").write_text(label, encoding="utf-8")
    return relative


def test_directory_dataset_letterboxes_normalizes_and_keeps_stable_ids(
    tmp_path: Path,
) -> None:
    _write_sample(tmp_path, split="val", name="z")
    _write_sample(tmp_path, split="val", name="a", color=(0, 255, 0))
    dataset = BHCDetrDataset(
        tmp_path,
        split="val",
        image_size=200,
        training=False,
    )

    assert [item.relative_path for item in dataset.samples] == [
        "images/val/a.png",
        "images/val/z.png",
    ]
    assert [item.image_id for item in dataset.samples] == [1, 2]
    view = dataset[1][0]
    assert view["image"].shape == (3, 200, 200)
    assert view["image"].dtype == torch.float32
    assert view["padding_mask"].shape == (200, 200)
    assert view["padding_mask"].dtype == torch.bool
    assert view["padding_mask"][:50].all()
    assert not view["padding_mask"][50:150].any()
    assert view["padding_mask"][150:].all()
    assert torch.allclose(
        view["target"]["boxes"],
        torch.tensor([[0.5, 0.5, 0.4, 0.2]]),
        atol=1e-6,
    )
    assert view["target"]["labels"].tolist() == [3]
    assert view["target"]["image_id"].item() == 2
    expected_red = torch.tensor([(1.0 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    assert torch.allclose(view["image"][:, 100, 100], expected_red, atol=1e-6)


def test_training_views_are_deterministic_per_epoch_and_collate_flattens(
    tmp_path: Path,
) -> None:
    _write_sample(tmp_path, split="train", name="a", size=(40, 30))
    _write_sample(tmp_path, split="train", name="b", size=(40, 30), color=(0, 0, 255))
    dataset = BHCDETRDataset(
        tmp_path,
        split="train",
        image_size=64,
        views_per_image=2,
        flip_probability=0.5,
        max_shift_ratio=0.2,
        seed=17,
    )
    first = dataset[0]
    repeated = dataset[0]
    assert len(first) == 2
    assert [view["target"]["view_id"].item() for view in first] == [0, 1]
    for left, right in zip(first, repeated):
        assert torch.equal(left["image"], right["image"])
        assert left["target"]["augmentation"] == right["target"]["augmentation"]

    collated = bhcdetr_collate([dataset[0], dataset[1]])
    assert collated["images"].shape == (4, 3, 64, 64)
    assert collated["padding_masks"].shape == (4, 64, 64)
    assert len(collated["targets"]) == 4
    assert [target["source_image_id"].item() for target in collated["targets"]] == [
        1,
        1,
        2,
        2,
    ]

    before = [view["target"]["augmentation"] for view in first]
    dataset.set_epoch(1)
    after = [view["target"]["augmentation"] for view in dataset[0]]
    assert before != after


def test_forced_flips_and_shift_update_boxes_and_padding_mask(tmp_path: Path) -> None:
    _write_sample(
        tmp_path,
        split="train",
        name="one",
        size=(10, 10),
        label="0 0.2 0.3 0.2 0.2\n",
    )
    dataset = BHCDetrDataset(
        tmp_path,
        split="train",
        image_size=10,
        views_per_image=1,
        flip_probability=1.0,
        max_shift_ratio=0.0,
    )
    view = dataset[0][0]
    assert view["target"]["augmentation"] == {
        "horizontal_flip": True,
        "vertical_flip": True,
        "shift_x": 0,
        "shift_y": 0,
    }
    assert torch.allclose(
        view["target"]["boxes"],
        torch.tensor([[0.8, 0.7, 0.2, 0.2]]),
        atol=1e-6,
    )
    assert not view["padding_mask"].any()


def test_json_manifest_supports_explicit_split_and_preserves_image_id(
    tmp_path: Path,
) -> None:
    train_path = _write_sample(tmp_path, split="train", name="train")
    val_path = _write_sample(tmp_path, split="val", name="val")
    manifest = tmp_path / "split.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"image_id": 91, "relative_path": train_path, "split": "train"},
                    {"image_id": 42, "relative_path": val_path, "split": "val"},
                ]
            }
        ),
        encoding="utf-8",
    )
    dataset = BHCDetrDataset(tmp_path, split="val", manifest=manifest, image_size=32)
    assert len(dataset) == 1
    assert dataset.samples[0].image_id == 42
    assert dataset.samples[0].relative_path == val_path


def test_json_fold_manifest_builds_held_out_train_and_val_views(tmp_path: Path) -> None:
    paths = [
        _write_sample(tmp_path, split="train", name=f"image_{index}")
        for index in range(3)
    ]
    manifest = tmp_path / "folds.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "image_id": index + 1,
                        "relative_path": path,
                        "fold": index,
                    }
                    for index, path in enumerate(paths)
                ]
            }
        ),
        encoding="utf-8",
    )
    train = BHCDetrDataset(
        tmp_path,
        split="train",
        manifest=manifest,
        held_out_fold=1,
        image_size=32,
    )
    val = BHCDetrDataset(
        tmp_path,
        split="val",
        manifest=manifest,
        held_out_fold=1,
        image_size=32,
    )
    assert [sample.image_id for sample in train.samples] == [1, 3]
    assert [sample.image_id for sample in val.samples] == [2]


def test_csv_materialized_split_view_is_supported(tmp_path: Path) -> None:
    path = _write_sample(tmp_path, split="train", name="csv")
    manifest = tmp_path / "fold0_train.csv"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "relative_path"])
        writer.writeheader()
        writer.writerow({"image_id": "77", "relative_path": path})
    dataset = BHCDetrDataset(
        tmp_path,
        split="train",
        manifest=manifest,
        image_size=32,
    )
    assert len(dataset) == 1
    assert dataset.samples[0].image_id == 77


def test_manifest_rejects_unsafe_relative_paths_before_opening(tmp_path: Path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "labels").mkdir()
    manifest = tmp_path / "unsafe.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "image_id": 1,
                        "relative_path": "../outside.png",
                        "split": "train",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsafe"):
        BHCDetrDataset(tmp_path, split="train", manifest=manifest)


def test_invalid_yolo_label_is_rejected_when_sample_is_loaded(tmp_path: Path) -> None:
    _write_sample(
        tmp_path,
        split="val",
        name="bad",
        label="0 0.5 0.5 -0.2 0.1\n",
    )
    dataset = BHCDetrDataset(tmp_path, split="val", image_size=32)
    with pytest.raises(ValueError, match="invalid YOLO box"):
        dataset[0]


def test_collate_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="empty"):
        bhcdetr_collate([])
