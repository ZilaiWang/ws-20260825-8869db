"""P0-3 三折汇总 CLI 测试。"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path


def _write_summary(
    root: Path,
    policy: str,
    resolution: int,
    fold: int,
    macro_recall: float,
    macro_f1: float,
) -> None:
    output = root / f"lp-{policy}-{resolution}-fold{fold}"
    output.mkdir(parents=True)
    value = {
        "condition": {
            "fold": fold,
            "policy": policy,
            "resolution": resolution,
            "regime": "linear_probe",
            "sampler": "natural",
            "seed": 42,
            "smoke": False,
            "eval_only": False,
        },
        "n_train": 100,
        "n_val": 50,
        "final_metrics": {
            "accuracy": macro_recall + 0.05,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "top5_accuracy": 0.9,
            "loss": 1.0,
            "samples_per_second": 100.0,
        },
        "aircraft20": {"macro_recall": macro_recall - 0.02},
    }
    (output / "run_summary.json").write_text(json.dumps(value), encoding="utf-8")


def test_summary_requires_three_folds_and_selects_two_conditions(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    runs = tmp_path / "runs"
    for policy, resolution, score in (
        ("tight", 224, 0.70),
        ("context_1p25", 224, 0.72),
        ("tight", 336, 0.71),
        ("context_1p25", 336, 0.69),
    ):
        for fold, offset in enumerate((-0.01, 0.0, 0.01)):
            _write_summary(runs, policy, resolution, fold, score + offset, score - 0.02)
    output = tmp_path / "aggregate"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src")
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_p03_runs.py"),
            "--runs-root",
            str(runs),
            "--output-dir",
            str(output),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with (output / "aggregate.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    selection = json.loads((output / "selection.json").read_text(encoding="utf-8"))

    assert len(rows) == 4
    assert {int(row["n_folds"]) for row in rows} == {3}
    assert selection["selected_for_fine_tune"][0] == {
        "policy": "context_1p25",
        "resolution": 224,
    }
    assert len(selection["selected_for_fine_tune"]) == 2


def test_summary_rejects_incomplete_condition(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent
    runs = tmp_path / "runs"
    _write_summary(runs, "tight", 224, 0, 0.7, 0.68)
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_p03_runs.py"),
            "--runs-root",
            str(runs),
            "--output-dir",
            str(tmp_path / "aggregate"),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "折数不完整" in result.stderr
