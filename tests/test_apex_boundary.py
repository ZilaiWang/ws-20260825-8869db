from __future__ import annotations

import numpy as np
import pytest

from rsdet.augmentation.apex_boundary import (
    assign_proposal_roles,
    rescue_indices,
    select_precision_threshold,
)
from rsdet.models.apex_boundary import ApexBoundaryClassifier

# The CLI owns the post-NMS evidence schema.  Importing it here keeps its gate
# arithmetic covered without duplicating production logic in a library module.
from scripts.run_apex_boundary_screen import _incremental_rescue_ledger


def test_proposal_roles_follow_score_order_and_fine_class() -> None:
    ground_truth = [{"category_id": 0, "bbox_xyxy": [0, 0, 10, 10]}]
    predictions = [
        {"category_id": 0, "bbox_xyxy": [0, 0, 10, 10], "score": 0.9},
        {"category_id": 0, "bbox_xyxy": [0, 0, 10, 10], "score": 0.8},
        {"category_id": 1, "bbox_xyxy": [0, 0, 10, 10], "score": 0.7},
        {"category_id": 0, "bbox_xyxy": [20, 20, 30, 30], "score": 0.6},
    ]
    roles = assign_proposal_roles(ground_truth, predictions)
    assert [row["role"] for row in roles] == [
        "canonical_tp",
        "fp_duplicate",
        "fp_cls",
        "fp_bg",
    ]
    assert [row["target"] for row in roles] == [1, 0, 0, 0]


def test_precision_threshold_maximises_true_positive_prefix() -> None:
    selected = select_precision_threshold([0.9, 0.8, 0.7, 0.6], [1, 1, 0, 1], minimum_precision=0.7)
    assert selected is not None
    assert selected.threshold == 0.6
    assert selected.true_positives == 3
    assert selected.false_positives == 1


def test_rescue_is_tail_only() -> None:
    rows = [{"score": score} for score in (0.001, 0.1, 0.5, 0.7)]
    assert rescue_indices(
        rows,
        [1.0, 0.95, 0.95, 1.0],
        detector_threshold=0.6,
        rescue_floor=0.01,
        quality_threshold=0.9,
    ) == {1, 2}


def test_apex_classifier_round_trip_and_prototype_arm() -> None:
    pytest.importorskip("sklearn")
    rows = [
        {
            "category_id": 0,
            "scale_bin": "small",
            "coarse": "ship",
            "score": score,
            "source_group": f"g{index // 2}",
        }
        for index, score in enumerate((0.8, 0.7, 0.3, 0.2, 0.75, 0.15))
    ]
    embeddings = np.asarray(
        [[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
        dtype=np.float32,
    )
    targets = [1, 1, 0, 0, 1, 0]
    model = ApexBoundaryClassifier(use_prototypes=True).fit(
        embeddings, rows, targets, [1.0] * len(rows)
    )
    probabilities = model.predict_proba(embeddings, rows)
    assert probabilities.shape == (6,)
    assert np.isfinite(probabilities).all()
    assert probabilities[0] > probabilities[2]


def test_mlp_rank_head_round_trip_probabilities() -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("torch")
    rows = [
        {
            "row_id": f"r{index}",
            "category_id": 0,
            "scale_bin": "small",
            "coarse": "ship",
            "score": 0.2 + 0.01 * index,
            "source_group": f"g{index // 2}",
            "role": "canonical_tp" if index % 2 == 0 else "jitter_hard_negative",
        }
        for index in range(12)
    ]
    targets = [1 if index % 2 == 0 else 0 for index in range(12)]
    embeddings = np.asarray(
        [[2.0, 0.2] if target else [-2.0, -0.2] for target in targets], dtype=np.float32
    )
    model = ApexBoundaryClassifier(use_prototypes=True, head="mlp_rank", random_state=7)
    model.fit(embeddings, rows, targets, [1.0] * len(rows))
    probabilities = model.predict_proba(embeddings, rows)
    assert probabilities.shape == (12,)
    assert np.isfinite(probabilities).all()
    mask = np.asarray(targets) == 1
    assert float(probabilities[mask].mean()) > float(probabilities[~mask].mean())


def test_incremental_rescue_ledger_is_coarse_specific_and_conservative() -> None:
    control = {
        "per_fine": {
            str(index): {"tp": 10, "fp": 2, "fn": 1}
            for index in (*range(4), 24)
        }
    }
    candidate = {
        "per_fine": {
            "0": {"tp": 11, "fp": 2, "fn": 0},
            "1": {"tp": 10, "fp": 1, "fn": 1},
            "2": {"tp": 11, "fp": 3, "fn": 0},
            "3": {"tp": 10, "fp": 2, "fn": 1},
            "24": {"tp": 11, "fp": 2, "fn": 0},
        }
    }
    ledger = _incremental_rescue_ledger(control, candidate, ["ship", "vehicle"])
    ship = ledger["per_coarse"]["ship"]
    assert ship["effective_added_tp"] == 2
    assert ship["effective_added_fp"] == 1
    assert ship["effective_removed_fp"] == 1
    assert ship["effective_incremental_precision"] == pytest.approx(2 / 3)
    assert ship["precision_gate_pass"] is False
    vehicle = ledger["per_coarse"]["vehicle"]
    assert vehicle["effective_incremental_precision"] == 1.0
    assert vehicle["precision_gate_pass"] is True


def test_incremental_rescue_ledger_rejects_lost_true_positive() -> None:
    control = {
        "per_fine": {str(index): {"tp": 10, "fp": 2, "fn": 1} for index in range(4)}
    }
    candidate = {
        "per_fine": {
            "0": {"tp": 11, "fp": 2, "fn": 0},
            "1": {"tp": 9, "fp": 2, "fn": 2},
            "2": {"tp": 10, "fp": 2, "fn": 1},
            "3": {"tp": 10, "fp": 2, "fn": 1},
        }
    }
    ship = _incremental_rescue_ledger(control, candidate, ["ship"])["per_coarse"]["ship"]
    assert ship["effective_incremental_precision"] == 1.0
    assert ship["effective_removed_tp"] == 1
    assert ship["precision_gate_pass"] is False
