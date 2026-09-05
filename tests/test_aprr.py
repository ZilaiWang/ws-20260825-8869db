from rsdet.contracts import Prediction
from rsdet.submission.aprr import AprrConfig, apply_aprr


def _prediction(rows):
    return Prediction(
        image_id=7,
        boxes_xyxy=[row[0] for row in rows],
        scores=[row[1] for row in rows],
        labels=[row[2] for row in rows],
    )


def test_aprr_emits_only_exact_primary_rows() -> None:
    primary = _prediction(
        [
            ([0, 0, 10, 10], 0.7, 5),
            ([20, 0, 30, 10], 0.4, 1),
            ([40, 0, 50, 10], 0.58, 24),
            ([60, 0, 70, 10], 0.7, 24),
        ]
    )
    ship = _prediction([([20, 0, 30, 10], 0.8, 1), ([90, 0, 100, 10], 0.9, 1)])
    vehicle = _prediction([([40, 0, 50, 10], 0.7, 24), ([80, 0, 90, 10], 0.9, 24)])
    output, stats = apply_aprr(
        primary,
        ship,
        vehicle,
        config=AprrConfig(
            primary_threshold=0.536,
            ship_support_threshold=0.5,
            vehicle_protect_threshold=0.6,
        ),
    )
    assert output.boxes_xyxy == primary.boxes_xyxy
    assert output.scores == primary.scores
    assert output.labels == primary.labels
    assert stats["ship_rescued"] == 1
    assert stats["vehicle_risk_supported"] == 1


def test_aprr_rejects_unsupported_vehicle_risk_band_and_class3_rescue() -> None:
    primary = _prediction(
        [([0, 0, 10, 10], 0.58, 24), ([20, 0, 30, 10], 0.4, 3)]
    )
    empty = _prediction([])
    output, stats = apply_aprr(
        primary,
        empty,
        empty,
        config=AprrConfig(primary_threshold=0.536, ship_support_threshold=0.5),
    )
    assert output.scores == []
    assert stats["vehicle_risk_rejected"] == 1
    assert stats["below_policy_floor"] == 1


def test_aprr_threshold_boundaries_are_inclusive() -> None:
    primary = _prediction(
        [([0, 0, 10, 10], 0.536, 0), ([20, 0, 30, 10], 0.6, 24)]
    )
    empty = _prediction([])
    output, _ = apply_aprr(
        primary,
        empty,
        empty,
        config=AprrConfig(primary_threshold=0.536, ship_support_threshold=0.5),
    )
    assert output.scores == [0.536, 0.6]
