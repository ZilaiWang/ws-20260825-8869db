from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "materialize_object_scale_detector_scenes.py"


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    image_dir = tmp_path / "images" / "train"
    label_dir = tmp_path / "labels" / "train"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    image = image_dir / "vehicle.jpg"
    Image.new("RGB", (1024, 1024), "black").save(image)
    (label_dir / "vehicle.txt").write_text("24 0.5 0.5 0.02 0.02\n")
    paths = tmp_path / "all.txt"
    paths.write_text(f"{image}\n")
    return image, paths


def test_overlap_is_rejected_without_full_training(tmp_path: Path) -> None:
    _, paths = _fixture(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-list",
            str(paths),
            "--val-list",
            str(paths),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "train/validation leakage" in result.stderr


def test_full_training_requires_and_records_identical_sets(tmp_path: Path) -> None:
    _, paths = _fixture(tmp_path)
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--train-list",
            str(paths),
            "--val-list",
            str(paths),
            "--output",
            str(output),
            "--target-classes",
            "24",
            "--vehicle-target-network-side",
            "96",
            "--full-training",
            "--dry-run",
        ],
        check=True,
    )
    import json

    summary = json.loads((output / "summary.json").read_text())
    assert summary["train_validation_overlap"] == 1
    assert summary["training_mode"] == "full_no_validation"
    assert summary["validation_used_for_training_or_selection"] is False
