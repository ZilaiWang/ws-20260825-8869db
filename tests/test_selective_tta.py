from __future__ import annotations

import pytest

pytest.importorskip("torch")

from rsdet.submission.selective_tta import (
    accept_rotated_candidates,
    select_tta_tiles,
    tile_summary_features,
)


def test_aircraft_second_view_is_bypassed() -> None:
    rotated = [
        {"category_id": 4, "score": 0.99, "bbox_xyxy": [0, 0, 10, 10]},
        {
            "category_id": 24,
            "score": 0.20,
            "bbox_xyxy": [20, 20, 30, 30],
            "official_match_quality": 0.95,
        },
    ]
    accepted = accept_rotated_candidates([], rotated, novel_budget_by_coarse={"vehicle": 1})
    assert [row["category_id"] for row in accepted] == [24]


def test_same_fine_support_admits_ship_candidate() -> None:
    identity = [{"category_id": 0, "score": 0.2, "bbox_xyxy": [0, 0, 10, 10]}]
    rotated = [{"category_id": 0, "score": 0.3, "bbox_xyxy": [1, 1, 11, 11]}]
    accepted = accept_rotated_candidates(identity, rotated, support_iou=0.25)
    assert len(accepted) == 1
    assert accepted[0]["tta_admission"] == "same_fine_support"


def test_tile_budget_is_deterministic() -> None:
    decisions = select_tta_tiles(
        [0.9, 0.7, 0.8, 0.1], max_fraction=0.5, minimum_probability=0.5
    )
    assert [item.tile_index for item in decisions] == [0, 2]


def test_tile_summary_emphasizes_vehicle_uncertainty() -> None:
    records = [
        {"category_id": 24, "score": 0.15, "bbox_xyxy": [2, 2, 30, 30]},
        {"category_id": 5, "score": 0.9, "bbox_xyxy": [100, 100, 160, 160]},
    ]
    features = tile_summary_features(records, width=1024, height=1024)
    assert features[1] == 0.15
    assert features[3] == 1.0
    assert features[6] == 1.0
    assert features[7] == 1.0
