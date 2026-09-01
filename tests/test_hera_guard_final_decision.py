from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path


def _frontier(path: Path, recall: float, ship: float, vehicle: float, fdr: float = 0.15) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "frontiers": {
                    "0.150": {
                        "crossfit": {
                            "recall": recall,
                            "fdr": fdr,
                            "macro_recall": recall,
                            "per_coarse": {
                                "ship": {"recall": ship, "fdr": fdr},
                                "aircraft": {"recall": 0.99, "fdr": 0.02},
                                "vehicle": {"recall": vehicle, "fdr": fdr},
                            },
                        }
                    }
                },
            }
        )
    )


def _frozen(path: Path, recall: float, ship: float, vehicle: float, fdr: float = 0.15) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "complete",
                "metrics": {
                    "recall": recall,
                    "fdr": fdr,
                    "macro_recall": recall,
                    "per_coarse": {
                        "ship": {"recall": ship, "fdr": fdr},
                        "aircraft": {"recall": 0.99, "fdr": 0.02},
                        "vehicle": {"recall": vehicle, "fdr": fdr},
                    },
                },
            }
        )
    )


def test_three_benchmark_gate_admits_primary_gain(tmp_path: Path, monkeypatch) -> None:
    paths = {}
    for condition in ("normal", "hard", "sentinel"):
        for route in ("base", "candidate"):
            paths[(condition, route)] = tmp_path / f"{condition}-{route}.json"
    _frontier(paths[("normal", "base")], 0.90, 0.85, 0.70)
    _frontier(paths[("normal", "candidate")], 0.901, 0.851, 0.701)
    _frontier(paths[("hard", "base")], 0.85, 0.75, 0.60)
    _frontier(paths[("hard", "candidate")], 0.856, 0.756, 0.601)
    _frontier(paths[("sentinel", "base")], 0.84, 0.74, 0.59)
    _frontier(paths[("sentinel", "candidate")], 0.842, 0.742, 0.591)
    output = tmp_path / "decision.json"
    argv = ["decide"]
    for condition in ("normal", "hard", "sentinel"):
        argv += [f"--{condition}-base", str(paths[(condition, "base")])]
        argv += [f"--{condition}-candidate", str(paths[(condition, "candidate")])]
    argv += ["--output", str(output)]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        runpy.run_path("scripts/decide_hera_guard_final_candidate.py", run_name="__main__")
    except SystemExit as error:
        assert error.code == 0
    payload = json.loads(output.read_text())
    assert payload["decision"]["formal_cv3_expansion_admission"] is True
    assert payload["comparisons"]["hard"]["absolute_score_equal_latency"]["delta"] > 0
    assert payload["threshold_tuning_on_sentinel"] is True


def test_sentinel_can_use_thresholds_frozen_on_hard(tmp_path: Path, monkeypatch) -> None:
    paths = {}
    for condition in ("normal", "hard", "sentinel"):
        for route in ("base", "candidate"):
            paths[(condition, route)] = tmp_path / f"{condition}-{route}.json"
            _frontier(paths[(condition, route)], 0.9, 0.85, 0.7)
    frozen_base = tmp_path / "sentinel-base-frozen.json"
    frozen_candidate = tmp_path / "sentinel-candidate-frozen.json"
    _frozen(frozen_base, 0.84, 0.74, 0.59)
    _frozen(frozen_candidate, 0.846, 0.746, 0.591)
    output = tmp_path / "decision.json"
    argv = ["decide"]
    for condition in ("normal", "hard", "sentinel"):
        argv += [f"--{condition}-base", str(paths[(condition, "base")])]
        argv += [f"--{condition}-candidate", str(paths[(condition, "candidate")])]
    argv += [
        "--sentinel-base-frozen",
        str(frozen_base),
        "--sentinel-candidate-frozen",
        str(frozen_candidate),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    try:
        runpy.run_path("scripts/decide_hera_guard_final_candidate.py", run_name="__main__")
    except SystemExit as error:
        assert error.code == 0
    payload = json.loads(output.read_text())
    assert payload["sentinel_thresholds_frozen_from_hard"] is True
    assert payload["threshold_tuning_on_sentinel"] is False
    assert payload["comparisons"]["sentinel"]["candidate_recall"] == 0.846
