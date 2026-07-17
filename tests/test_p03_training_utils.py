"""P03 训练入口中不依赖 GPU 的样本权重规则测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_script_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "train_crop_classifier.py"
    spec = importlib.util.spec_from_file_location("p03_training_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _records(*counts: int):
    return [
        SimpleNamespace(class_id=class_id)
        for class_id, count in enumerate(counts)
        for _ in range(count)
    ]


def test_sqrt_inverse_weights_and_cap_are_applied_per_sample() -> None:
    module = _load_script_module()
    records = _records(100, 25, 1)
    weights = module._sampler_weights(records, exponent=0.5, cap=4.0)

    assert np.all(weights[:100] == 1.0)
    assert np.all(weights[100:125] == 2.0)
    assert weights[-1] == 4.0


def test_sampler_audit_reports_expected_class_mass_not_fake_epoch_counts() -> None:
    module = _load_script_module()
    audit = module._sampler_audit(
        _records(100, 25, 1),
        "sqrt_inverse",
        {"exponent": 0.5, "max_weight_ratio": 4.0},
    )

    probabilities = audit["expected_class_probability"]
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert np.isclose(probabilities[0], 100 / 154)
    assert np.isclose(probabilities[1], 50 / 154)
    assert np.isclose(probabilities[2], 4 / 154)
    assert "expected" in audit["note"]
