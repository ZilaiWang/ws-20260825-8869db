import json

from rsdet.contracts import Prediction, TileRecord
from rsdet.pipeline.large_image import PipelineConfig
from rsdet.postprocess.global_aggregation import fuse_global_predictions
from rsdet.postprocess.safe_tile_fusion import fuse_safe_tile_predictions
from rsdet.postprocess.thresholds import (
    effective_threshold,
    filter_prediction_by_thresholds,
    normalize_fine_thresholds,
)
from rsdet.submission.competition import load_submission_config


def test_threshold_priority_and_json_normalization() -> None:
    fine = normalize_fine_thresholds({"0": 0.4, "24": 0.8})
    assert fine == {0: 0.4, 24: 0.8}
    coarse = {"ship": 0.3, "aircraft": 0.2, "vehicle": 0.7}
    assert effective_threshold(0, global_threshold=0.1, coarse_thresholds=coarse, fine_thresholds=fine) == 0.4
    assert effective_threshold(1, global_threshold=0.1, coarse_thresholds=coarse, fine_thresholds=fine) == 0.3
    assert effective_threshold(24, global_threshold=0.1, coarse_thresholds=coarse, fine_thresholds=fine) == 0.8


def test_safe_global_and_direct_filter_have_per_box_parity() -> None:
    prediction = Prediction(
        0,
        [[0, 0, 10, 10], [20, 20, 30, 30], [40, 40, 50, 50]],
        [0.39, 0.41, 0.79],
        [0, 0, 24],
    )
    tile = TileRecord(0, 9, 0, 0, 100, 100)
    fine = {0: 0.4, 24: 0.8}
    direct = filter_prediction_by_thresholds(
        prediction, global_threshold=0.1, fine_thresholds=fine
    )
    safe = fuse_safe_tile_predictions(
        [prediction], [tile], parent_image_id=9, image_width=100, image_height=100,
        score_threshold=0.1, score_threshold_by_fine=fine
    )
    global_result = fuse_global_predictions(
        [prediction], [tile], parent_image_id=9, image_width=100, image_height=100,
        score_threshold=0.1, score_threshold_by_fine=fine
    )
    assert direct.boxes_xyxy == safe.boxes_xyxy == global_result.boxes_xyxy
    assert direct.scores == safe.scores == global_result.scores
    assert direct.labels == safe.labels == global_result.labels


def test_docker_config_requires_complete_fine_thresholds(tmp_path) -> None:
    payload = {
        "metric_protocol": "platform_observed_20260831",
        "model": {
            "family": "yolo", "weight_path": "/app/model.pt", "expected_sha256": "a" * 64,
            "imgsz": 1024, "confidence": 0.001, "iou": 0.7, "max_detections": 2000
        },
        "pipeline": {
            "tile_size": 1024, "overlap": 128, "batch_size": 4, "fusion": "safe",
            "score_threshold": 0.001,
            "score_threshold_by_fine": {str(index): 0.15 for index in range(25)}
        }
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload))
    loaded = load_submission_config(path)
    assert len(loaded["pipeline"]["score_threshold_by_fine"]) == 25
    del payload["pipeline"]["score_threshold_by_fine"]["24"]
    path.write_text(json.dumps(payload))
    try:
        load_submission_config(path)
    except ValueError as error:
        assert "all fine classes" in str(error)
    else:
        raise AssertionError("incomplete thresholds were accepted")


def test_pipeline_config_accepts_fine_thresholds() -> None:
    config = PipelineConfig(score_threshold_by_fine={0: 0.2})
    assert config.score_threshold_by_fine == {0: 0.2}
