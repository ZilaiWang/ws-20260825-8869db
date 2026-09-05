from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_entrypoint():
    path = ROOT / "submission/docker/app/main.py"
    spec = importlib.util.spec_from_file_location("final_submission_main", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_entrypoint_keeps_stable_runtime_default(monkeypatch) -> None:
    module = _load_entrypoint()
    seen = {}
    monkeypatch.setattr(module, "parse_args", lambda: Namespace(input="in", output="out"))
    monkeypatch.setattr(module, "load_submission_config", lambda _path: {})
    monkeypatch.setattr(
        module,
        "run_submission",
        lambda _input, _output, _config, *, detector_factory: seen.setdefault(
            "factory", detector_factory
        ),
    )
    assert module.main() == 0
    assert seen["factory"] is module.CompetitionDetector


def test_entrypoint_selects_sprint20_only_when_configured(monkeypatch) -> None:
    module = _load_entrypoint()
    seen = {}
    monkeypatch.setattr(module, "parse_args", lambda: Namespace(input="in", output="out"))
    monkeypatch.setattr(
        module, "load_submission_config", lambda _path: {"sprint20": {"mode": "shared"}}
    )
    monkeypatch.setattr(
        module,
        "run_submission",
        lambda _input, _output, _config, *, detector_factory: seen.setdefault(
            "factory", detector_factory
        ),
    )
    assert module.main() == 0
    assert seen["factory"].__module__ == "sprint20.runtime"
