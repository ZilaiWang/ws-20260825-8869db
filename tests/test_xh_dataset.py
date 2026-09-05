from pathlib import Path

from rsdet.data.xh_dataset import XHDataset


def test_dataset_ignores_macos_appledouble_sidecars(tmp_path: Path) -> None:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    (image_dir / "sample.jpg").write_bytes(b"placeholder")
    (label_dir / "sample.txt").write_text("", encoding="utf-8")
    (image_dir / "._sample.jpg").write_bytes(b"appledouble metadata")
    (label_dir / "._sample.txt").write_bytes(b"\x00\x05appledouble metadata")

    dataset = XHDataset(tmp_path, load_images=False)

    assert len(dataset) == 1
    assert dataset.refs[0].stem == "sample"
    assert dataset.refs[0].image_path == image_dir / "sample.jpg"
    assert dataset.refs[0].label_path == label_dir / "sample.txt"
