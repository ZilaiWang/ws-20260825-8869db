"""Actual Torch CPU forward, with synthetic classifier/crops; NOT P40 weights."""
# ruff: noqa: E402

import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

torch = pytest.importorskip("torch", reason="Torch is required for bounded D4 fixtures")

from sprint20.bounded_d4 import BoundedD4Runtime
from sprint20.policy import _apply_aircraft_labels, full_view_labels, prediction_fingerprint


@dataclass
class Prediction:
    image_id: int
    boxes_xyxy: list
    scores: list
    labels: list


def install_fixture_modules(monkeypatch):
    def normalize(im):
        return torch.tensor(np.asarray(im).copy()).permute(2, 0, 1).float() / 255

    def views(x):
        flipped = torch.flip(x, [-1])
        return torch.stack(
            [torch.rot90(z, k, [-2, -1]) for z in (x, flipped) for k in range(4)], 1
        ).flatten(0, 1)

    def crop(im, box, size):
        return im.crop(tuple(box)).resize((size, size))

    def nms(boxes, scores, iou):
        # Synthetic identical-box dedup is sufficient for these fixtures.
        keep = []
        seen = set()
        for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i)):
            if tuple(boxes[i]) not in seen:
                keep.append(i)
                seen.add(tuple(boxes[i]))
        return keep

    modules = {
        "rsdet.data.crop_classification": {"render_crop": crop},
        "rsdet.submission.aircraft_d4": {"_normalize": normalize, "_tensorized_d4_views": views},
        "rsdet.postprocess.nms": {"nms": nms},
    }
    for name, attrs in modules.items():
        pieces = name.split(".")
        for i in range(1, len(pieces)):
            parent = ".".join(pieces[:i])
            if parent not in sys.modules:
                mod = types.ModuleType(parent)
                mod.__path__ = []
                monkeypatch.setitem(sys.modules, parent, mod)
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        monkeypatch.setitem(sys.modules, name, mod)
    return normalize, views, crop, nms


class ToyClassifier(torch.nn.Module):
    def forward(self, x):
        result = torch.full((len(x), 25), -10.0, device=x.device)
        label = (x.mean((1, 2, 3)) > 0.5).long() + 4
        result[torch.arange(len(x)), label] = 10
        return result


def test_cpu_runtime_matches_full_d4_and_saves_views(monkeypatch):
    normalize, views, crop, nms = install_fixture_modules(monkeypatch)
    torch.set_num_threads(1)
    rgb = np.zeros((32, 64, 3), dtype=np.uint8)
    rgb[:, 32:] = 255
    p = Prediction(1, [[0, 0, 32, 32], [32, 0, 64, 32], [1, 1, 5, 5]], [0.9, 0.8, 0.7], [4, 4, 24])
    delegate = SimpleNamespace(
        model=ToyClassifier().eval(),
        device=torch.device("cpu"),
        tensorized_views=True,
        channels_last=False,
        config={"batch_objects": 64, "relabel_min_probability": 0.9, "nms_iou": 0.5},
    )
    wrapper = BoundedD4Runtime(delegate)
    out = wrapper.refine(rgb, p)
    base = torch.stack(
        [normalize(crop(Image.fromarray(rgb), box, 224)) for box in p.boxes_xyxy[:2]]
    )
    probs = (
        delegate.model(views(base)).reshape(2, 8, 25).float()[:, :, 4:24].softmax(2).mean(1).numpy()
    )
    labels = full_view_labels(probs, [0, 0])
    full = _apply_aircraft_labels(p, labels, nms)
    assert prediction_fingerprint(out) == prediction_fingerprint(full)
    assert out.labels == [4, 5, 24]
    assert wrapper.last_audit["certified_keep"] == 1
    assert wrapper.last_audit["view_evaluations"] == 10 < 16


def test_no_aircraft_bypass_without_model_calls(monkeypatch):
    install_fixture_modules(monkeypatch)
    delegate = SimpleNamespace(tensorized_views=True)
    wrapper = BoundedD4Runtime(delegate)
    p = Prediction(1, [[0, 0, 1, 1]], [0.9], [24])
    assert wrapper.refine(np.zeros((2, 2, 3), dtype=np.uint8), p) is p
