from __future__ import annotations

from scripts.train_y5_hard_replay import select_balanced_hard_tiles


def test_balanced_hard_tiles_round_robin_sources() -> None:
    tiles = [
        {
            "source_image_id": source,
            "candidate_score": score,
            "image": f"{source}-{score}.jpg",
        }
        for source in (1, 2)
        for score in (0.1, 0.9, 0.5)
    ]
    selected = select_balanced_hard_tiles(tiles, maximum=4)
    assert [item["source_image_id"] for item in selected] == [1, 2, 1, 2]
    assert [item["candidate_score"] for item in selected] == [0.9, 0.9, 0.5, 0.5]


def test_balanced_hard_tiles_handles_unequal_sources() -> None:
    tiles = [
        {"source_image_id": 1, "candidate_score": 0.9, "image": "1.jpg"},
        {"source_image_id": 2, "candidate_score": 0.8, "image": "2.jpg"},
        {"source_image_id": 2, "candidate_score": 0.7, "image": "3.jpg"},
    ]
    assert len(select_balanced_hard_tiles(tiles, maximum=10)) == 3
