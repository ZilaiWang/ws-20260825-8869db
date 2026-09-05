from __future__ import annotations

import pytest

from rsdet.contracts import Prediction
from rsdet.submission.class_resolution_router import (
    PrimaryLabelRescue,
    ResolutionLabelRoute,
    compose_routed_predictions,
)


def test_route_preserves_branch_geometry_and_applies_postfusion_thresholds() -> None:
    route = ResolutionLabelRoute(
        primary_labels=frozenset(range(24)),
        expert_labels=frozenset({24}),
        primary_threshold=0.4,
        expert_threshold=0.6,
    )
    primary = Prediction(
        image_id=7,
        boxes_xyxy=[[1, 2, 3, 4], [10, 20, 30, 40]],
        scores=[0.5, 0.99],
        labels=[3, 24],
    )
    expert = Prediction(
        image_id=7,
        boxes_xyxy=[[5, 6, 7, 8], [50, 60, 70, 80]],
        scores=[0.8, 0.5],
        labels=[24, 24],
    )
    output = compose_routed_predictions(primary, expert, route=route)
    assert output.labels == [24, 3]
    assert output.boxes_xyxy == [[5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]]


def test_route_requires_exact_disjoint_taxonomy() -> None:
    with pytest.raises(ValueError, match="cover"):
        ResolutionLabelRoute(
            primary_labels=frozenset(range(23)),
            expert_labels=frozenset({24}),
            primary_threshold=0.4,
            expert_threshold=0.6,
        )


def test_primary_rescue_preserves_expert_priority_and_self_deduplicates() -> None:
    route = ResolutionLabelRoute(
        primary_labels=frozenset(range(24)),
        expert_labels=frozenset({24}),
        primary_threshold=0.4,
        expert_threshold=0.42,
    )
    primary = Prediction(
        image_id=3,
        boxes_xyxy=[
            [0, 0, 10, 10],
            [30, 30, 40, 40],
            [30.5, 30.5, 40.5, 40.5],
            [60, 60, 70, 70],
        ],
        scores=[0.99, 0.80, 0.75, 0.59],
        labels=[24, 24, 24, 24],
    )
    expert = Prediction(
        image_id=3,
        boxes_xyxy=[[0.5, 0.5, 10.5, 10.5]],
        scores=[0.50],
        labels=[24],
    )
    output = compose_routed_predictions(
        primary,
        expert,
        route=route,
        primary_rescue=PrimaryLabelRescue(
            labels=frozenset({24}), threshold=0.60, dedup_iou=0.70
        ),
    )
    assert output.boxes_xyxy == [
        [30.0, 30.0, 40.0, 40.0],
        [0.5, 0.5, 10.5, 10.5],
    ]
    assert output.scores == [0.8, 0.5]


def test_primary_rescue_must_target_expert_labels() -> None:
    route = ResolutionLabelRoute(
        primary_labels=frozenset(range(24)),
        expert_labels=frozenset({24}),
        primary_threshold=0.4,
        expert_threshold=0.42,
    )
    with pytest.raises(ValueError, match="owned"):
        compose_routed_predictions(
            Prediction(1, [], [], []),
            Prediction(1, [], [], []),
            route=route,
            primary_rescue=PrimaryLabelRescue(
                labels=frozenset({23}), threshold=0.6, dedup_iou=0.7
            ),
        )
    with pytest.raises(ValueError, match="disjoint"):
        ResolutionLabelRoute(
            primary_labels=frozenset(range(25)),
            expert_labels=frozenset({24}),
            primary_threshold=0.4,
            expert_threshold=0.6,
        )
