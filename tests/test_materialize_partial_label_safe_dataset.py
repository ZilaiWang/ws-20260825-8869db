from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from scripts.materialize_partial_label_safe_dataset import materialize


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "data"
    (root / "images/train").mkdir(parents=True)
    (root / "labels/train").mkdir(parents=True)
    Image.new("RGB", (100, 80)).save(root / "images/train/a.jpg")
    Image.new("RGB", (100, 80)).save(root / "images/train/b.jpg")
    (root / "labels/train/a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    (root / "labels/train/b.txt").write_text("")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"image_id": 1, "relative_path": "images/train/a.jpg", "fold": 0},
                    {"image_id": 2, "relative_path": "images/train/b.jpg", "fold": 1},
                ]
            }
        )
    )
    confirmed = tmp_path / "confirmed.json"
    confirmed.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "c1",
                    "file_name": "images/train/a.jpg",
                    "bbox_xyxy": [10, 20, 30, 40],
                }
            ]
        )
    )
    ignored = tmp_path / "ignored.json"
    ignored.write_text(
        json.dumps([{"candidate_id": "i1", "file_name": "images/train/b.jpg"}])
    )
    return root, manifest, confirmed, ignored


def test_materialize_adds_confirmed_and_excludes_ambiguous(tmp_path: Path) -> None:
    root, manifest, confirmed, ignored = _fixture(tmp_path)
    output = tmp_path / "out"
    audit = materialize(
        manifest,
        root,
        confirmed,
        ignored,
        output,
        held_out_fold=None,
        add_confirmed=True,
        ambiguous_policy="exclude_image",
    )
    assert audit["image_count"] == 1
    assert audit["confirmed_boxes_added"] == 1
    lines = (output / "labels/train/image_00001.txt").read_text().splitlines()
    assert len(lines) == 2
    assert lines[-1].startswith("24 ")


def test_materialize_control_omits_patch_with_same_exclusion(tmp_path: Path) -> None:
    root, manifest, confirmed, ignored = _fixture(tmp_path)
    output = tmp_path / "out"
    audit = materialize(
        manifest,
        root,
        confirmed,
        ignored,
        output,
        held_out_fold=None,
        add_confirmed=False,
        ambiguous_policy="exclude_image",
    )
    assert audit["image_count"] == 1
    assert audit["confirmed_boxes_added"] == 0


def test_materialize_deduplicates_original_and_repeated_review_boxes(tmp_path: Path) -> None:
    root, manifest, confirmed, ignored = _fixture(tmp_path)
    (root / "labels/train/a.txt").write_text("24 0.5 0.5 0.1 0.1\n")
    confirmed.write_text(
        json.dumps(
            [
                {
                    "candidate_id": "same-as-original",
                    "file_name": "images/train/a.jpg",
                    "bbox_xyxy": [45, 36, 55, 44],
                },
                {
                    "candidate_id": "new",
                    "file_name": "images/train/a.jpg",
                    "bbox_xyxy": [10, 20, 30, 40],
                },
                {
                    "candidate_id": "duplicate-new",
                    "file_name": "images/train/a.jpg",
                    "bbox_xyxy": [10, 20, 30, 40],
                },
            ]
        )
    )
    audit = materialize(
        manifest,
        root,
        confirmed,
        ignored,
        tmp_path / "out",
        held_out_fold=None,
        add_confirmed=True,
        ambiguous_policy="exclude_image",
    )
    assert audit["confirmed_boxes_added"] == 1
    assert audit["confirmed_boxes_deduplicated_against_original"] == 1
    assert audit["confirmed_boxes_deduplicated_against_confirmed"] == 1
    assert audit["confirmed_class_contract"] == {"class_id": 24, "class_name": "FSC"}
