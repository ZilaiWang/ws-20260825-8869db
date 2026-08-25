"""全局阈值扫描 CLI 端到端测试。"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def test_threshold_sweep_outputs_and_official_reevaluation(tmp_path: Path) -> None:
    root = Path(__file__).parent.parent
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "pred.json"
    output_dir = tmp_path / "sweep"
    gt_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10]},
                    {"image_id": 1, "category_id": 0, "bbox": [20, 20, 10, 10]},
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions = [
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 1, "category_id": 0, "bbox": [20, 20, 10, 10], "score": 0.8},
        {"image_id": 1, "category_id": 0, "bbox": [0, 0, 10, 10], "score": 0.7},
    ]
    pred_path.write_text(json.dumps(predictions), encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable,
            "scripts/sweep_thresholds.py",
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--output-dir",
            str(output_dir),
            "--threshold-start",
            "0.7",
            "--threshold-stop",
            "1.0",
            "--threshold-step",
            "0.1",
            "--allow-partial-taxonomy",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    csv_path = output_dir / "threshold_sweep.csv"
    selected_path = output_dir / "selected_thresholds.yaml"
    metrics_path = output_dir / "metrics_at_selected_thresholds.json"
    assert csv_path.exists() and selected_path.exists() and metrics_path.exists()

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert {row["threshold"] for row in rows} == {"0.7", "0.8", "0.9", "1"}
    assert all(row["contract_version"] == "contract_v1" for row in rows)
    assert all(row["eval_version"] == "official_eval_v1" for row in rows)
    assert all(row["threshold_stage"] == "post_fusion" for row in rows)

    selected = yaml.safe_load(selected_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    expected_versions = {
        "contract_version": "contract_v1",
        "eval_version": "official_eval_v1",
        "ranking_version": "official_ranking_v1_6",
    }
    assert selected["protocol_versions"] == expected_versions
    assert metrics["protocol_versions"] == expected_versions
    assert selected["source"]["threshold_stage"] == "post_fusion"
    assert set(selected["workpoints"]) == {
        "official_best",
        "internal_best",
        "recall_ceiling",
    }
    official = selected["workpoints"]["official_best"]
    assert official["threshold"] == 0.8
    assert official["overall_recall"] == 1.0
    assert official["overall_fdr"] == 0.0

    filtered_path = tmp_path / "filtered.json"
    filtered_path.write_text(
        json.dumps([item for item in predictions if item["score"] >= official["threshold"]]),
        encoding="utf-8",
    )
    reevaluated_path = tmp_path / "reevaluated.json"
    reevaluate = subprocess.run(
        [
            sys.executable,
            "scripts/evaluate.py",
            "--gt",
            str(gt_path),
            "--pred",
            str(filtered_path),
            "--output",
            str(reevaluated_path),
            "--allow-partial-taxonomy",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert reevaluate.returncode == 0, reevaluate.stdout + reevaluate.stderr
    reevaluated = json.loads(reevaluated_path.read_text(encoding="utf-8"))
    assert official["overall_recall"] == reevaluated["overall_recall"]
    assert official["overall_fdr"] == reevaluated["overall_fdr"]
    assert official["details"] == reevaluated["details"]
    assert official["per_class"] == reevaluated["per_class"]
