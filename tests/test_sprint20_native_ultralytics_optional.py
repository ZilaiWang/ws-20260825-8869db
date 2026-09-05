"""Real-library wiring test on RANDOM weights; never an accuracy experiment."""
# ruff: noqa: E402

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch is required for native head wiring")
ultralytics = pytest.importorskip(
    "ultralytics", reason="Pinned Ultralytics not installed in authoring environment"
)
pytest.importorskip("rsdet")
from ultralytics.nn.tasks import DetectionModel

from rsdet.contracts import InferenceSample
from rsdet.models.ultralytics_adapter import UltralyticsDetector
from sprint20.heads import CachedOTM, SharedHeadCapture, select_native_otm
from sprint20.policy import prediction_fingerprint


def test_native_oto_otm_and_shared_decode(tmp_path):
    if ultralytics.__version__ != "8.4.103":
        pytest.skip("This implementation is pinned to 8.4.103")
    torch.set_num_threads(1)
    torch.manual_seed(8)
    model = DetectionModel("yolo26s.yaml", nc=25, verbose=False).eval()
    with torch.no_grad():
        for tower in (model.model[-1].cv3, model.model[-1].one2one_cv3):
            for layer in tower:
                layer[-1].bias.fill_(-2.0)
    weight = tmp_path / "random_test_only.pt"
    torch.save({"model": model, "train_args": {"task": "detect"}}, weight)

    def make():
        d = UltralyticsDetector(imgsz=96, confidence=0.01, iou=0.7, max_detections=50, half=False)
        d.load(str(weight))
        d.to("cpu")
        d.eval()
        return d

    image = np.random.default_rng(9).integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
    batch = [InferenceSample(1, image, 96, 96)]
    native_oto = make().predict(batch)
    native_otm = select_native_otm(make()).predict(batch)
    shared = SharedHeadCapture(make())
    shared.begin_image()
    shared_oto = shared.predict(batch)
    shared_otm = CachedOTM(shared).predict(batch)
    assert prediction_fingerprint(native_oto[0]) == prediction_fingerprint(shared_oto[0])
    assert prediction_fingerprint(native_otm[0]) == prediction_fingerprint(shared_otm[0])
    shared.close()
