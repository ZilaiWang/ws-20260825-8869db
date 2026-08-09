"""Ultralytics 适配器在无真实模型依赖下的契约测试。"""

from pathlib import Path

import numpy as np

from rsdet.contracts import InferenceSample
from rsdet.models.ultralytics_adapter import UltralyticsDetector


class _Array:
    def __init__(self, values):
        self._values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self._values


class _Boxes:
    def __init__(self):
        self.xyxy = _Array([[1.0, 2.0, 11.0, 22.0]])
        self.conf = _Array([0.75])
        self.cls = _Array([24.0])

    def __len__(self):
        return 1


class _Result:
    boxes = _Boxes()


class _FakeModel:
    def __init__(self):
        self.kwargs = None

    def predict(self, **kwargs):
        self.kwargs = kwargs
        return [_Result() for _ in kwargs["source"]]

    def to(self, device):
        self.device = device


def test_adapter_converts_results_and_preserves_ids(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeModel()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "rsdet.models.ultralytics_adapter.create_ultralytics_model",
        lambda family, weights: fake,
    )
    detector = UltralyticsDetector(family="yolo", half=True)
    detector.load(str(checkpoint))
    detector.to("cpu")

    outputs = detector.predict([InferenceSample(17, "image.jpg", 100, 80)])

    assert outputs[0].image_id == 17
    assert outputs[0].boxes_xyxy == [[1.0, 2.0, 11.0, 22.0]]
    assert outputs[0].scores == [0.75]
    assert outputs[0].labels == [24]
    assert fake.kwargs["quantize"] is None


def test_adapter_converts_rgb_numpy_to_bgr(monkeypatch, tmp_path: Path) -> None:
    fake = _FakeModel()
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.setattr(
        "rsdet.models.ultralytics_adapter.create_ultralytics_model",
        lambda family, weights: fake,
    )
    detector = UltralyticsDetector()
    detector.load(str(checkpoint))
    rgb = np.asarray([[[10, 20, 30]]], dtype=np.uint8)

    detector.predict([InferenceSample(1, rgb, 1, 1)])

    assert fake.kwargs["source"][0].tolist() == [[[30, 20, 10]]]
    assert rgb.tolist() == [[[10, 20, 30]]]
