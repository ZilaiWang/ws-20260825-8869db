"""安全切片融合的部署不变量测试。"""

from rsdet.contracts import Prediction, TileRecord
from rsdet.postprocess.safe_tile_fusion import fuse_safe_tile_predictions


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
