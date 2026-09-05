# ruff: noqa: E402

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="Torch is required for Sprint20 head fixtures")

from sprint20 import heads


class ToyModel:
    def __init__(self, nc=25):
        h = SimpleNamespace(
            nc=nc,
            reg_max=1,
            end2end=True,
            cv2=torch.nn.Linear(2, 2),
            cv3=torch.nn.Linear(2, 25),
            one2one_cv2=torch.nn.Linear(2, 2),
            one2one_cv3=torch.nn.Linear(2, 25),
        )
        self.model = [h]

    @property
    def end2end(self):
        return self.model[-1].end2end

    @end2end.setter
    def end2end(self, value):
        self.model[-1].end2end = value


class ToyWrapper:
    def __init__(self):
        self.model = ToyModel()
        self.predictor = None

    def predict(self, **kwargs):
        return kwargs


def test_inspection_is_before_fusion():
    w = ToyWrapper()
    info = heads.inspect_head(w)
    assert info["nc"] == 25 and info["cv2"] and info["one2one_cv3"]
    w.model.model[-1].cv3 = None
    with pytest.raises(RuntimeError, match="absent/already fused"):
        heads.assert_available(w, "otm")


def test_never_random_init_a_missing_head():
    w = ToyWrapper()
    w.model.model[-1].cv2 = None
    with pytest.raises(RuntimeError):
        heads.assert_available(w, "shared")
    assert w.model.model[-1].cv2 is None


def test_head_requires_25_classes():
    w = ToyWrapper()
    w.model.model[-1].nc = 80
    with pytest.raises(ValueError):
        heads.assert_available(w, "otm")


def test_native_head_switch_is_instance_local(monkeypatch):
    monkeypatch.setattr(heads, "require_version", lambda: "8.4.103")
    w, untouched = ToyWrapper(), ToyWrapper()
    d = SimpleNamespace(_model=w)
    heads.select_native_otm(d)
    assert not w.model.end2end
    assert w.predict(source="x")["end2end"] is False
    assert untouched.model.end2end and "end2end" not in untouched.predict()


def test_cannot_switch_a_reused_predictor(monkeypatch):
    monkeypatch.setattr(heads, "require_version", lambda: "8.4.103")
    w = ToyWrapper()
    w.predictor = object()
    with pytest.raises(RuntimeError, match="fresh"):
        heads.select_native_otm(SimpleNamespace(_model=w))


def test_pinned_version_fails_closed(monkeypatch):
    monkeypatch.setattr(heads.importlib.metadata, "version", lambda _: "8.4.104")
    with pytest.raises(RuntimeError):
        heads.require_version()


@dataclass
class Sample:
    image_id: int
    image: np.ndarray
    width: int
    height: int


def test_cache_hash_rejects_same_id_different_pixels():
    a = Sample(1, np.zeros((2, 2, 3), dtype=np.uint8), 2, 2)
    b = Sample(1, np.ones((2, 2, 3), dtype=np.uint8), 2, 2)
    assert heads.sample_key(a) != heads.sample_key(b)
    capture = SimpleNamespace(cache={heads.sample_key(a): "prediction"})
    cached = heads.CachedOTM(capture)
    assert cached.predict([a]) == ["prediction"]
    with pytest.raises(RuntimeError):
        cached.predict([a])
    with pytest.raises(RuntimeError):
        cached.predict([b])


def test_cache_is_deep_copied():
    a = Sample(1, np.zeros((2, 2, 3), dtype=np.uint8), 2, 2)
    capture = SimpleNamespace(cache={heads.sample_key(a): {"box": [1, 2, 3, 4]}})
    copied = heads.CachedOTM(capture).predict([a])[0]
    copied["box"][0] = 100
    assert capture.cache[heads.sample_key(a)]["box"][0] == 1


def test_cached_otm_satisfies_eval_contract():
    capture = SimpleNamespace(cache={})
    cached = heads.CachedOTM(capture)
    assert cached.eval() is None
