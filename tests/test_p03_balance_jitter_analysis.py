"""P03-3 配对与 jitter 分层的关键规则测试。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from rsdet.evaluation.classification import evaluate_classification


def _load_script_module():
    scripts = Path(__file__).resolve().parent.parent / "scripts"
    sys.path.insert(0, str(scripts))
    path = scripts / "analyze_p03_balance_jitter.py"
    spec = importlib.util.spec_from_file_location("p03_balance_jitter_script", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _condition(module, predictions: list[int]):
    labels = np.asarray([0, 1, 2, 3], dtype=np.int64)
    logits = np.full((4, 25), -5.0, dtype=np.float64)
    logits[np.arange(4), predictions] = 5.0
    metrics, confusion = evaluate_classification(labels, logits)
    return module.ConditionData(
        policy="tight",
        resolution=224,
        annotation_uids=np.asarray(["a", "b", "c", "d"], dtype=object),
        source_image_ids=np.asarray(["i0", "i1", "i2", "i3"], dtype=object),
        labels=labels,
        logits=logits,
        predictions=np.asarray(predictions, dtype=np.int64),
        folds=np.asarray([0, 1, 2, 0]),
        metrics=metrics,
        confusion=confusion,
    )


def test_jitter_numeric_bins_are_non_overlapping_and_exhaustive() -> None:
    module = _load_script_module()
    geometry = {
        "gt_coverage_fraction": np.asarray([0.85, 0.92, 0.97, 1.0]),
        "jitter_center_magnitude": np.asarray([0.02, 0.06, 0.09, 0.08]),
        "jitter_scale_log_magnitude": np.asarray([0.02, 0.05, 0.08, 0.07]),
        "padding_fraction": np.asarray([0.0, 0.1, 0.0, 0.2]),
        "gt_short_edge": np.asarray([32.0, 64.0, 96.0, 128.0]),
        "major_class": np.asarray(["ship", "aircraft", "vehicle", "aircraft"]),
        "source_edge_risk": np.asarray([False, True, False, True]),
    }
    definitions = dict(module._geometry_definitions(geometry))

    for prefix in ("coverage_", "center_", "scale_log_"):
        masks = [mask for name, mask in definitions.items() if name.startswith(prefix)]
        assert np.array_equal(np.stack(masks).sum(axis=0), np.ones(4, dtype=int))


def test_paired_row_uses_second_minus_first_and_cluster_bootstrap() -> None:
    module = _load_script_module()
    first = _condition(module, [0, 1, 2, 3])
    second = _condition(module, [0, 2, 2, 4])
    row = module._paired_row("first", "second", "unit", first, second, bootstrap_resamples=100)

    assert row["first_only_correct"] == 2
    assert row["second_only_correct"] == 0
    assert row["net_correct_objects"] == -2
    assert row["accuracy_delta"] == -0.5
    assert row["n_source_clusters"] == 4
    assert row["n_resamples"] == 100
