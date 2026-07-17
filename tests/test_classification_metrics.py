"""P0-3 分类指标测试。"""

import numpy as np

from rsdet.evaluation.classification import (
    confusion_matrix_fixed,
    evaluate_classification,
    metrics_from_confusion,
    topk_accuracy,
)


def test_confusion_orientation_and_macro_recall() -> None:
    matrix = confusion_matrix_fixed([0, 0, 1, 1], [0, 1, 1, 1], num_classes=3)
    assert matrix.tolist() == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    metrics = metrics_from_confusion(matrix, class_names=("a", "b", "c"))
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_recall"] == 0.75
    assert metrics["n_macro_classes"] == 2
    assert metrics["per_class"][2]["recall"] is None


def test_topk_and_aircraft_subset_are_computed_from_fixed_25_classes() -> None:
    logits = np.full((3, 25), -10.0)
    logits[0, 4] = 5.0
    logits[1, 5] = 4.0
    logits[2, 0] = 3.0
    metrics, matrix = evaluate_classification([4, 6, 0], logits)

    assert metrics["accuracy"] == 2 / 3
    assert metrics["subgroups"]["aircraft20"]["n_samples"] == 2
    assert metrics["subgroups"]["aircraft20"]["macro_recall"] == 0.5
    assert matrix.shape == (25, 25)
    assert topk_accuracy(logits, [4, 6, 0], 5) >= 2 / 3


def test_frequency_tiers_use_training_counts_without_changing_predictions() -> None:
    logits = np.eye(25, dtype=np.float64)[:3]
    metrics, _ = evaluate_classification(
        [0, 1, 2], logits, training_class_counts={0: 1, 1: 10, 2: 100}
    )
    assert metrics["frequency_tiers"]["tail"]["class_ids"] == [0]
    assert metrics["frequency_tiers"]["middle"]["class_ids"] == [1]
    assert metrics["frequency_tiers"]["head"]["class_ids"] == [2]
    assert metrics["macro_recall"] == 1.0

