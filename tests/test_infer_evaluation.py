"""Inference post-filtering and automatic official evaluation tests."""

import json
from pathlib import Path

from rsdet.contracts import Prediction
from scripts.infer import _automatic_evaluation, _filter_scores, _manifest_records


def test_fine_threshold_overrides_coarse_threshold() -> None:
    prediction = Prediction(
        image_id=1,
        boxes_xyxy=[[0.0, 0.0, 10.0, 10.0]],
        scores=[0.4],
        labels=[24],
    )

    filtered = _filter_scores(
        prediction,
        {"vehicle": 0.1},
        {"FSC": 0.5},
    )

    assert filtered.boxes_xyxy == []


def test_inference_manifest_selects_held_out_fold(tmp_path: Path) -> None:
    manifest = tmp_path / "cv3.json"
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {"image_id": 1, "relative_path": "images/train/a.jpg", "fold": 0},
                    {"image_id": 2, "relative_path": "images/train/b.jpg", "fold": 1},
                ]
            }
        ),
        encoding="utf-8",
    )

    records = _manifest_records(
        manifest,
        split="val",
        held_out_fold=1,
    )

    assert records == [{"image_id": 2, "relative_path": "images/train/b.jpg"}]


def test_inference_automatically_writes_official_metrics(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.json"
    prediction_path = tmp_path / "run_predictions.json"
    metrics_path = tmp_path / "run_metrics.json"
    gt_path.write_text(
        json.dumps(
            {
                "annotations": [
                    {
                        "image_id": 1,
                        "category_id": 24,
                        "bbox": [0.0, 0.0, 10.0, 10.0],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prediction_path.write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 24,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "score": 0.9,
                }
            ]
        ),
        encoding="utf-8",
    )
    config = {
        "evaluation": {
            "gt": str(gt_path),
            "project_config": "configs/project.yaml",
            "metrics_output": str(metrics_path),
        }
    }

    result = _automatic_evaluation(config, prediction_path)

    assert result is not None
    output_path, metrics = result
    assert output_path == metrics_path
    assert metrics["overall_recall"] == 1.0
    assert metrics["overall_fdr"] == 0.0
    assert metrics["detection_gate"]["passed"] is True
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics
