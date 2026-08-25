"""Regression tests for synchronization-free training metric aggregation."""

from __future__ import annotations

import subprocess
import sys

import pytest

torch_probe = subprocess.run(
    [sys.executable, "-c", "import torch"],
    check=False,
    capture_output=True,
    text=True,
)
if torch_probe.returncode != 0:  # pragma: no cover - environment dependent
    reason = (torch_probe.stderr or torch_probe.stdout).strip().splitlines()
    pytest.skip(
        f"PyTorch is unavailable: {reason[-1] if reason else 'import failed'}",
        allow_module_level=True,
    )

import torch  # noqa: E402

from rsdet.engine.trainer import _accumulate_metrics, _mean_metrics  # noqa: E402


def test_metric_aggregation_accepts_scalar_and_singleton_loss_shapes() -> None:
    sums = {}
    _accumulate_metrics(
        sums,
        {
            "loss_detection": torch.tensor([2.0]),
            "loss_bhcl": torch.tensor(3.0),
        },
    )
    _accumulate_metrics(
        sums,
        {
            "loss_detection": torch.tensor([4.0]),
            "loss_bhcl": torch.tensor(5.0),
        },
    )

    assert all(value.ndim == 0 for value in sums.values())
    metrics = _mean_metrics(sums, 2)
    assert metrics["loss_bhcl"] == pytest.approx(4.0)
    assert metrics["loss_detection"] == pytest.approx(3.0)


def test_metric_aggregation_rejects_non_scalar_loss() -> None:
    with pytest.raises(ValueError, match="must contain exactly one value"):
        _accumulate_metrics({}, {"loss_bad": torch.ones(2)})
