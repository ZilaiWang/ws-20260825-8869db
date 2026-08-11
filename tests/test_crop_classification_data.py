"""P0-3 crop 分类数据契约测试。"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from rsdet.data.crop_classification import (
    CropClassificationDataset,
    CropRecord,
    load_crop_records,
    render_crop,
    select_deterministic_subset,
    validate_fold_isolation,
)


def _record(
    annotation_uid: str,
    *,
    class_id: int = 0,
    fold: int = 0,
    group: str = "group_a",
) -> CropRecord:
    names = ("HM", "LQS")
    return CropRecord(
        manifest_version="v1",
        crop_id=f"crop_{annotation_uid}",
        annotation_uid=annotation_uid,
        source_image_id=f"image_{annotation_uid}",
        source_relative_path="images/train/example.png",
        source_width=4,
        source_height=4,
        source_checksum_sha256="0" * 64,
        class_id=class_id,
        class_name=names[class_id],
        major_class="ship",
        crop_policy="tight",
        crop_xyxy=(0.0, 0.0, 4.0, 4.0),
        fold=fold,
        leakage_group_id=group,
    )


def _write_manifest(path: Path) -> None:
    fields = [
        "manifest_version",
        "crop_id",
        "annotation_uid",
        "source_image_id",
        "source_relative_path",
        "source_width",
        "source_height",
        "source_checksum_sha256",
        "class_id",
        "class_name",
        "major_class",
        "crop_policy",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "fold",
        "leakage_group_id",
        "color_mode",
        "outside_policy",
        "resize_semantics",
    ]
    rows = []
    for annotation, fold, group in (("a", 0, "g0"), ("b", 1, "g1"), ("c", 2, "g2")):
        for policy in ("tight", "context_1p25", "jitter_light"):
            rows.append(
                {
                    "manifest_version": "v1",
                    "crop_id": f"crop_{annotation}_{policy}",
                    "annotation_uid": annotation,
                    "source_image_id": f"image_{annotation}",
                    "source_relative_path": "images/train/example.png",
                    "source_width": 4,
                    "source_height": 4,
                    "source_checksum_sha256": "0" * 64,
                    "class_id": 0,
                    "class_name": "HM",
                    "major_class": "ship",
                    "crop_policy": policy,
                    "crop_x0": 0,
                    "crop_y0": 0,
                    "crop_x1": 4,
                    "crop_y1": 4,
                    "fold": fold,
                    "leakage_group_id": group,
                    "color_mode": "RGB",
                    "outside_policy": "pad",
                    "resize_semantics": "direct_square_resize",
                }
            )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_selects_one_policy_and_leakage_safe_fold(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest)
    train = load_crop_records(manifest, crop_policy="tight", held_out_fold=1, split="train")
    val = load_crop_records(manifest, crop_policy="tight", held_out_fold=1, split="val")

    assert [item.annotation_uid for item in train] == ["a", "c"]
    assert [item.annotation_uid for item in val] == ["b"]
    validate_fold_isolation(train, val)


def test_fold_isolation_rejects_group_leakage() -> None:
    with pytest.raises(ValueError, match="leakage_group_id"):
        validate_fold_isolation(
            [_record("train", group="shared")],
            [_record("val", fold=1, group="shared")],
        )


def test_float_extent_render_pads_outside_with_black() -> None:
    array = np.full((4, 4, 3), 255, dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")
    crop = render_crop(image, (-2.0, 0.0, 2.0, 4.0), 4)
    result = np.asarray(crop)

    assert crop.size == (4, 4)
    assert result[:, 0].max() == 0
    assert result[:, -1].min() > 200


def test_dataset_returns_rendered_crop_label_and_stable_index(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    image_dir.mkdir(parents=True)
    Image.new("RGB", (4, 4), (10, 20, 30)).save(image_dir / "example.png")
    dataset = CropClassificationDataset([_record("a")], tmp_path, 224)

    image, label, index = dataset[0]
    assert image.size == (224, 224)
    assert (label, index) == (0, 0)


def test_smoke_subset_round_robins_classes_deterministically() -> None:
    records = [
        _record("a0", class_id=0),
        _record("a1", class_id=0),
        _record("a2", class_id=0),
        _record("b0", class_id=1),
    ]
    selected = select_deterministic_subset(records, 3)
    assert {item.class_id for item in selected} == {0, 1}
    assert [item.annotation_uid for item in selected] == ["a0", "a1", "b0"]
