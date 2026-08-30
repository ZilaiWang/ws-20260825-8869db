import numpy as np
import pytest

from rsdet.hera_guard.metric_aligned import MetricAlignedRole
from rsdet.hera_guard.metric_risk import (
    align_anchor_scores,
    broad_rank_roles,
    build_metric_features,
    deterministic_rank_pairs,
    metric_risk_loss,
)


def _role(index: int, role: str, group: str) -> MetricAlignedRole:
    return MetricAlignedRole(
        candidate_id=index,
        image_id=1,
        predicted_category_id=0,
        predicted_coarse="ship",
        role=role,
        target=int(role == "canonical_positive"),
        object_group_id=group,
        support_gt_index=0,
        support_category_id=0,
        support_coarse="ship",
        support_iou=0.8,
        official_match_iou=0.8 if role == "canonical_positive" else 0.0,
    )


def test_anchor_alignment_and_features() -> None:
    evidence = [
        {
            "image_id": 2,
            "category_id": 0,
            "bbox": [1, 2, 10, 20],
            "score": 0.8,
            "detector_score": 0.7,
            "source_fold": 1,
            "source_model": "Y5",
        },
        {
            "image_id": 3,
            "category_id": 24,
            "bbox": [3, 4, 8, 8],
            "score": 0.6,
            "source_fold": 2,
            "source_model": "M3",
        },
    ]
    anchor = [{**evidence[1], "score": 0.2}, {**evidence[0], "score": 0.9}]
    scores = align_anchor_scores(evidence, anchor)
    assert scores == [0.9, 0.2]
    features = np.asarray(
        build_metric_features(
            evidence,
            anchor_scores=scores,
            category_mapping={0: "ship", 24: "vehicle"},
        )
    )
    assert features.shape == (2, 25)
    assert np.isfinite(features).all()


def test_rank_pairs_are_deterministic_and_group_local() -> None:
    roles = [
        _role(0, "canonical_positive", "g0"),
        _role(1, "duplicate_negative", "g0"),
        _role(2, "background_negative", ""),
        _role(3, "canonical_positive", "g1"),
        _role(4, "cross_fine_negative", "g1"),
    ]
    pairs = deterministic_rank_pairs(
        roles, include_roles=broad_rank_roles(), max_pairs=10
    )
    assert pairs == [(0, 1), (3, 4)]


def test_metric_loss_rewards_correct_ranking() -> None:
    torch = pytest.importorskip("torch")
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    pairs = torch.tensor([[0, 1], [2, 3]])
    good = metric_risk_loss(
        torch.tensor([3.0, -2.0, 2.5, -1.0]),
        targets,
        stage="one_winner",
        rank_pairs=pairs,
        one_winner_pairs=pairs,
        soft_threshold=0.0,
    )["total"]
    bad = metric_risk_loss(
        torch.tensor([-2.0, 3.0, -1.0, 2.5]),
        targets,
        stage="one_winner",
        rank_pairs=pairs,
        one_winner_pairs=pairs,
        soft_threshold=0.0,
    )["total"]
    assert good.item() < bad.item()
