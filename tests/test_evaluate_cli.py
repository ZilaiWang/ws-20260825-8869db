"""评估 CLI 的 COCO 格式兼容测试。"""

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_standard_coco_detection_list(tmp_path: Path) -> None:
    """预测文件使用标准 COCO detection 顶层列表。"""
    root = Path(__file__).parent.parent
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    output_path = tmp_path / "metrics.json"
    gt_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 24, "bbox": [0, 0, 100, 100]}
                ],
                "categories": [{"id": 24, "name": "FSC"}],
            }
        ),
        encoding="utf-8",
    )
    pred_path.write_text(
        json.dumps([{"image_id": 1, "category_id": 24, "bbox": [25, 25, 100, 100], "score": 0.9}]),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--project-config",
            "configs/project.yaml",
            "--output",
            str(output_path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["per_class"]["vehicle"]["tp"] == 1
    assert metrics["protocol_versions"] == {
        "contract_version": "contract_v1",
        "eval_version": "official_eval_v1",
    }
    assert metrics["details"]["matching_policy"] == "same_fine_category_id"
    assert metrics["detection_gate"]["passed"] is True


def test_wrong_fine_category_is_not_a_match(tmp_path: Path) -> None:
    """同属飞机大类的错误细类不能匹配。"""
    root = Path(__file__).parent.parent
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    output_path = tmp_path / "metrics.json"
    gt_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 4, "bbox": [0, 0, 100, 100]}
                ],
                "categories": [{"id": 4, "name": "A1_SU-35"}],
            }
        ),
        encoding="utf-8",
    )
    pred_path.write_text(
        json.dumps([{"image_id": 1, "category_id": 5, "bbox": [0, 0, 100, 100], "score": 0.99}]),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--project-config",
            "configs/project.yaml",
            "--output",
            str(output_path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    metrics = json.loads(output_path.read_text(encoding="utf-8"))
    assert metrics["per_class"]["aircraft"] == {
        "recall": 0.0,
        "fdr": 1.0,
        "tp": 0,
        "fp": 1,
        "fn": 1,
    }
    assert metrics["detection_gate"]["passed"] is False


def test_missing_protocol_versions_is_rejected(tmp_path: Path) -> None:
    """评估配置缺少冻结版本时不能静默执行。"""
    root = Path(__file__).parent.parent
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    config_path = tmp_path / "project.yaml"
    gt_path.write_text(json.dumps({"annotations": []}), encoding="utf-8")
    pred_path.write_text("[]", encoding="utf-8")
    config = yaml.safe_load((root / "configs/project.yaml").read_text(encoding="utf-8"))
    del config["protocol_versions"]
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True),
        encoding="utf-8",
    )

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--project-config",
            str(config_path),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert "protocol_versions" in result.stdout + result.stderr
