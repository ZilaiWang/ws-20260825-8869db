"""安全切片融合的部署不变量测试。"""

import random

from rsdet.contracts import Prediction, TileRecord
from rsdet.postprocess.safe_tile_fusion import (
    _candidate_sort_key,
    _canonical,
    _fine_nms,
    _overlap,
    _restore_candidates,
    fuse_safe_tile_predictions,
)


def _tiles() -> list[TileRecord]:
    return [
        TileRecord(0, 9, 0, 0, 600, 600),
        TileRecord(1, 9, 400, 0, 600, 600),
    ]


def _fuse(predictions: list[Prediction], **kwargs: float) -> Prediction:
    return fuse_safe_tile_predictions(
        predictions,
        _tiles(),
        parent_image_id=9,
        image_width=1000,
        image_height=600,
        **kwargs,
    )


def test_same_class_cross_tile_duplicate_merges() -> None:
    fused = _fuse(
        [
            Prediction(0, [[450, 100, 550, 200]], [0.9], [4]),
            Prediction(1, [[50, 100, 150, 200]], [0.8], [4]),
        ]
    )
    assert fused.boxes_xyxy == [[450.0, 100.0, 550.0, 200.0]]
    assert fused.scores == [0.9]
    assert fused.labels == [4]


def test_cross_fine_class_overlap_is_preserved() -> None:
    fused = _fuse(
        [
            Prediction(0, [[450, 100, 550, 200]], [0.9], [4]),
            Prediction(1, [[50, 100, 150, 200]], [0.8], [5]),
        ]
    )
    assert fused.labels == [4, 5]
    assert fused.scores == [0.9, 0.8]


def test_low_score_candidate_is_removed_before_fusion() -> None:
    fused = _fuse(
        [
            Prediction(0, [[450, 100, 550, 200]], [0.04], [4]),
            Prediction(1, [[50, 100, 150, 200]], [0.8], [4]),
        ],
        score_threshold=0.051,
    )
    assert fused.scores == [0.8]
    assert fused.boxes_xyxy == [[450.0, 100.0, 550.0, 200.0]]


def test_coarse_specific_thresholds_are_applied_before_fusion() -> None:
    fused = _fuse(
        [
            Prediction(
                0,
                [[10, 10, 50, 50], [100, 10, 150, 60], [200, 10, 250, 60]],
                [0.35, 0.35, 0.35],
                [0, 4, 24],
            ),
            Prediction(1, [], [], []),
        ],
        score_threshold=0.0,
        score_threshold_by_coarse={"ship": 0.30, "aircraft": 0.40, "vehicle": 0.34},
    )
    assert fused.labels == [0, 24]
    assert fused.scores == [0.35, 0.35]


def test_coarse_specific_thresholds_require_complete_taxonomy() -> None:
    try:
        _fuse(
            [Prediction(0, [], [], []), Prediction(1, [], [], [])],
            score_threshold_by_coarse={"ship": 0.3},
        )
    except ValueError as error:
        assert "exactly ship, aircraft, vehicle" in str(error)
    else:
        raise AssertionError("incomplete coarse thresholds must fail")


def test_anchor_clustering_is_non_transitive() -> None:
    tiles = [
        TileRecord(0, 9, 0, 0, 1000, 600),
        TileRecord(1, 9, 0, 0, 1000, 600),
        TileRecord(2, 9, 0, 0, 1000, 600),
    ]
    predictions = [
        Prediction(0, [[100, 100, 200, 200]], [0.9], [4]),
        Prediction(1, [[140, 100, 240, 200]], [0.8], [4]),
        Prediction(2, [[180, 100, 280, 200]], [0.7], [4]),
    ]
    fused = fuse_safe_tile_predictions(
        predictions,
        tiles,
        parent_image_id=9,
        image_width=1000,
        image_height=600,
        merge_iou=0.4,
        merge_ios=1.0,
        fine_nms_iou=1.0,
    )
    assert fused.scores == [0.9, 0.7]


def test_complete_proposal_is_canonical_and_fields_stay_consistent() -> None:
    fused = _fuse(
        [
            Prediction(0, [[500, 100, 600, 200]], [0.95], [4]),
            Prediction(1, [[90, 100, 190, 200]], [0.8], [4]),
        ],
        border_margin=8.0,
    )
    assert fused.boxes_xyxy == [[490.0, 100.0, 590.0, 200.0]]
    assert fused.scores == [0.8]
    assert fused.labels == [4]


def test_same_tile_candidates_do_not_enter_same_duplicate_cluster() -> None:
    fused = _fuse(
        [
            Prediction(
                0,
                [[450, 100, 550, 200], [455, 100, 555, 200]],
                [0.9, 0.8],
                [4, 4],
            ),
            Prediction(1, [], [], []),
        ],
        fine_nms_iou=1.0,
    )
    assert fused.scores == [0.9, 0.8]


def _reference_fusion(
    predictions: list[Prediction],
    tiles: list[TileRecord],
    *,
    merge_iou: float,
    merge_ios: float,
    fine_nms_iou: float,
) -> Prediction:
    """Former all-pairs implementation retained only as an equivalence oracle."""

    candidates = _restore_candidates(
        predictions,
        tiles,
        parent_image_id=9,
        image_width=1000,
        image_height=1000,
        score_threshold=0.0,
        score_threshold_by_coarse=None,
        border_margin=8.0,
    )
    ordered = sorted(candidates, key=_candidate_sort_key)
    assigned = [False] * len(ordered)
    canonical = []
    for anchor_index, anchor in enumerate(ordered):
        if assigned[anchor_index]:
            continue
        assigned[anchor_index] = True
        cluster = [anchor]
        for candidate_index in range(anchor_index + 1, len(ordered)):
            if assigned[candidate_index]:
                continue
            candidate = ordered[candidate_index]
            if candidate.label != anchor.label or candidate.tile_id == anchor.tile_id:
                continue
            iou, ios = _overlap(anchor.box, candidate.box)
            if iou >= merge_iou or ios >= merge_ios:
                assigned[candidate_index] = True
                cluster.append(candidate)
        canonical.append(_canonical(cluster))
    canonical = _fine_nms(canonical, fine_nms_iou)
    return Prediction(
        9,
        [list(item.box) for item in canonical],
        [item.score for item in canonical],
        [item.label for item in canonical],
    )


def test_spatial_index_is_exactly_equivalent_to_all_pairs_reference() -> None:
    rng = random.Random(20260829)
    tiles = [
        TileRecord(index, 9, (index % 3) * 200, (index // 3) * 200, 600, 600)
        for index in range(6)
    ]
    predictions = []
    for tile in tiles:
        boxes = []
        scores = []
        labels = []
        for _ in range(90):
            x1 = rng.uniform(0.0, 540.0)
            y1 = rng.uniform(0.0, 540.0)
            width = rng.uniform(8.0, 120.0)
            height = rng.uniform(8.0, 120.0)
            boxes.append([x1, y1, min(600.0, x1 + width), min(600.0, y1 + height)])
            scores.append(rng.random())
            labels.append(rng.randrange(25))
        predictions.append(Prediction(tile.tile_id, boxes, scores, labels))

    for merge_iou, merge_ios in ((0.50, 0.75), (0.0, 0.75), (0.50, 0.0)):
        expected = _reference_fusion(
            predictions,
            tiles,
            merge_iou=merge_iou,
            merge_ios=merge_ios,
            fine_nms_iou=0.70,
        )
        actual = fuse_safe_tile_predictions(
            predictions,
            tiles,
            parent_image_id=9,
            image_width=1000,
            image_height=1000,
            score_threshold=0.0,
            merge_iou=merge_iou,
            merge_ios=merge_ios,
            fine_nms_iou=0.70,
        )
        assert actual == expected
