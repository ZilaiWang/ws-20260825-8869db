from pathlib import Path

import pytest

from scripts.infer_deim_tiled_coco import resolve_image_path


def test_resolve_image_path_accepts_direct_root(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"test")
    assert resolve_image_path(tmp_path, "image.jpg") == image


def test_resolve_image_path_finds_one_fold_image(tmp_path: Path) -> None:
    image = tmp_path / "fold_0" / "images" / "image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"test")
    assert resolve_image_path(tmp_path, "image.jpg") == image


def test_resolve_image_path_rejects_ambiguous_fold_names(tmp_path: Path) -> None:
    for fold in (0, 1):
        image = tmp_path / f"fold_{fold}" / "images" / "image.jpg"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"test")
    with pytest.raises(FileNotFoundError, match="exactly one"):
        resolve_image_path(tmp_path, "image.jpg")
