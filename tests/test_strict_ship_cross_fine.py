from rsdet.postprocess.strict_ship_cross_fine import suppress_strict_ship_cross_fine


def _row(label: int, score: float, bbox: list[float]) -> dict:
    return {"image_id": 1, "category_id": label, "score": score, "bbox": bbox}


def test_strict_ship_cross_fine_keeps_highest_real_proposal() -> None:
    kept, audit = suppress_strict_ship_cross_fine(
        [_row(2, 0.80, [10, 10, 100, 40]), _row(3, 0.60, [11, 10, 100, 40])]
    )
    assert [row["category_id"] for row in kept] == [2]
    assert audit["suppressed_count"] == 1


def test_aircraft_vehicle_and_distinct_ships_are_untouched() -> None:
    rows = [
        _row(4, 0.80, [10, 10, 100, 40]),
        _row(5, 0.60, [10, 10, 100, 40]),
        _row(24, 0.70, [10, 10, 100, 40]),
        _row(1, 0.90, [10, 10, 30, 30]),
        _row(2, 0.85, [80, 80, 30, 30]),
    ]
    kept, audit = suppress_strict_ship_cross_fine(rows)
    assert len(kept) == len(rows)
    assert audit["suppressed_count"] == 0
