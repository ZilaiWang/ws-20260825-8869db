"""预测交付校验 CLI 测试。"""

import json
import os
import subprocess
import sys
from pathlib import Path


def _run_validator(root: Path, pred: Path, gt: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    return subprocess.run(
        [
            sys.executable,
            "scripts/validate_predictions.py",
            "--pred",
            str(pred),
            "--gt",
            str(gt),
            "--project-config",
            "configs/project.yaml",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


def test_valid_prediction_file_passes(tmp_path: Path) -> None:
    root = Path(__file__).parent.parent
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    gt.write_text(
        json.dumps(
            {
                "images": [{"id": 11, "width": 100, "height": 100}],
                "annotations": [],
                "categories": [],
            }
        ),
        encoding="utf-8",
    )
    pred.write_text(
        json.dumps(
            [{"image_id": 11, "category_id": 24, "bbox": [10, 20, 30, 40], "score": 0.9}]
        ),
        encoding="utf-8",
    )

    result = _run_validator(root, pred, gt)

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS" in result.stdout + result.stderr


def test_unknown_image_id_fails(tmp_path: Path) -> None:
    root = Path(__file__).parent.parent
    gt = tmp_path / "gt.json"
    pred = tmp_path / "pred.json"
    gt.write_text(
        json.dumps({"images": [{"id": 11, "width": 100, "height": 100}]}),
        encoding="utf-8",
    )
    pred.write_text(
        json.dumps(
            [{"image_id": 12, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9}]
        ),
        encoding="utf-8",
    )

    result = _run_validator(root, pred, gt)

    assert result.returncode == 1
    assert "不在图像清单" in result.stdout + result.stderr
