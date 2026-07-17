"""P03-1 本地配对分析的关键规则测试。"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import numpy as np

from rsdet.evaluation.classification import evaluate_classification


def _load_script_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "analyze_p03_results.py"
    spec = importlib.util.spec_from_file_location("p03_analysis_script", path)
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


def test_pairwise_counts_use_same_object_and_second_minus_first() -> None:
    module = _load_script_module()
    first = _condition(module, [0, 1, 2, 3])
    second = _condition(module, [0, 2, 2, 4])
    second.resolution = 336
    row = module._pairwise_row(first, second)

    assert row["both_correct"] == 2
    assert row["first_only_correct"] == 2
    assert row["second_only_correct"] == 0
    assert row["net_correct_objects"] == -2
    assert row["accuracy_delta"] == -0.5


def test_return_checksum_verifier_recognizes_only_known_self_entry(tmp_path: Path) -> None:
    module = _load_script_module()
    root = tmp_path / "P03-TASK-01"
    root.mkdir()
    value = root / "value.txt"
    value.write_text("verified\n", encoding="utf-8")
    value_sha = hashlib.sha256(value.read_bytes()).hexdigest()
    empty_sha = hashlib.sha256(b"").hexdigest()
    checksum = root / "RETURN_FILES_SHA256.txt"
    checksum.write_text(
        f"{empty_sha}  P03-TASK-01/RETURN_FILES_SHA256.txt\n{value_sha}  P03-TASK-01/value.txt\n",
        encoding="utf-8",
    )

    result = module._verify_return_checksums(root)
    assert result["entries"] == 2
    assert result["matched"] == 1
    assert result["known_self_entry_mismatch"] == 1
    assert result["unexpected_mismatches"] == []


def test_calibration_metrics_are_finite_and_bounded() -> None:
    module = _load_script_module()
    condition = _condition(module, [0, 2, 2, 4])
    values = module._calibration_metrics(condition.labels, condition.logits)
    assert values["nll"] > 0
    assert 0 <= values["ece_15"] <= 1
    assert values["mean_confidence_correct"] >= values["mean_confidence_wrong"]
