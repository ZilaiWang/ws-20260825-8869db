import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rsdet.contracts import Prediction
from rsdet.data.xh_dataset import FINE_NAMES
from rsdet.submission import competition

CONFIG = (Path(__file__).resolve().parents[1] / "submission/docker/configs"
          / "progressive40_full_s1280_frozen0536_v1.json")


def test_frozen_candidate_config():
    config = competition.load_submission_config(CONFIG)
    assert config["model"]["imgsz"] == 1280
    assert config["model"]["inference_adapter"] == "shared_offline"
    assert config["model"]["expected_sha256"].startswith("b0df7981f6ad")
    assert config["source_training_checkpoint_sha256"].startswith("904c4935a854")
    assert config["pipeline"]["score_threshold"] == 0.001
    assert config["post_fusion_score_threshold"] == 0.536
    assert config["pipeline"]["max_detections"] == 4000


@pytest.mark.parametrize("value", [float("nan"), 1.1, -0.1, 0.0001])
def test_invalid_postfusion_threshold_rejected(tmp_path, value):
    config = json.loads(CONFIG.read_text())
    config["post_fusion_score_threshold"] = value
    p = tmp_path / "c.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="post_fusion"):
        competition.load_submission_config(p)


def test_shared_adapter_does_not_silently_ignore_tta(tmp_path):
    config = json.loads(CONFIG.read_text())
    config["model"]["rot90_views"] = [0, 1]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="shared_offline"):
        competition.load_submission_config(p)


def test_frozen_threshold_is_applied_after_low_floor_fusion(monkeypatch):
    detector = competition.CompetitionDetector.__new__(competition.CompetitionDetector)
    detector.config = competition.load_submission_config(CONFIG)
    detector.detector = object()
    detector.resolution_runtime = None
    detector.pipeline_config = competition._pipeline_config_from_mapping(
        detector.config["pipeline"], "pipeline")

    def fake_pipeline(*args, **kwargs):
        assert kwargs["config"].score_threshold == 0.001
        return Prediction(0, [[0, 0, 10, 10]] * 3, [0.535, 0.536, 0.537], [0, 1, 24]), None

    monkeypatch.setattr(competition, "run_pipeline", fake_pipeline)
    result = detector.predict(Image.new("RGB", (20, 20)))
    assert [x["score"] for x in result] == [0.537, 0.536]
    assert [x["category_id"] for x in result] == [24, 1]


def test_shared_adapter_is_actually_selected(monkeypatch, tmp_path):
    from rsdet.models import ultralytics_adapter

    config = competition.load_submission_config(CONFIG)
    w = tmp_path / "test.pt"
    w.write_bytes(b"dummy")
    config["model"]["weight_path"] = str(w)
    config["model"]["expected_sha256"] = competition._sha256(w)

    class FakeShared:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self._model = SimpleNamespace(names=dict(enumerate(FINE_NAMES)))

        def load(self, path):
            self.path = path

        def to(self, device):
            self.device = device

        def eval(self):
            self.eval_called = True

    monkeypatch.setattr(competition, "check_gpu", lambda _: None)
    monkeypatch.setattr(ultralytics_adapter, "UltralyticsDetector", FakeShared)
    result = competition.CompetitionDetector(config)
    assert isinstance(result.detector, FakeShared)
    assert result.detector.kwargs["imgsz"] == 1280
    assert result.detector.kwargs["confidence"] == 0.001
    assert result.detector.eval_called
